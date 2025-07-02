#!/usr/bin/env python3
# scripts/train_paired_flux.py - 茶歇领替换LoRA训练主脚本
import os
import sys
import argparse
import yaml
import logging
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import torch
import torch.multiprocessing as mp
from transformers import CLIPTokenizer

from models.flux_model import FluxLoRAModel
from data_processing.paired_dataset import PairedCollarDataset
from data_processing.data_loader import create_paired_data_loaders
from training.flux_trainer import FluxCollarTrainer
from utils.logging_utils import setup_logging

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="FLUX茶歇领替换LoRA训练")
    
    # 数据相关参数
    parser.add_argument("--train_data_dir", type=str, required=True, help="训练数据目录")
    parser.add_argument("--val_data_dir", type=str, help="验证数据目录")
    parser.add_argument("--mask_dir", type=str, required=True, help="蒙版目录")
    parser.add_argument("--output_dir", type=str, default="./outputs/collar_replacement", help="输出目录")
    
    # 模型相关参数
    parser.add_argument("--model_name", type=str, default="black-forest-labs/FLUX.1-fill-dev", help="FLUX模型名称")
    parser.add_argument("--cache_dir", type=str, default="./cache", help="模型缓存目录")
    parser.add_argument("--resume_from_checkpoint", type=str, help="恢复训练的检查点路径")
    
    # LoRA参数
    parser.add_argument("--lora_rank", type=int, default=32, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=2, help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="学习率")
    parser.add_argument("--max_steps", type=int, default=3000, help="最大训练步数")
    parser.add_argument("--warmup_steps", type=int, default=300, help="预热步数")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="梯度累积步数")
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"], help="混合精度")
    parser.add_argument("--optimizer", type=str, default="adamw_8bit", help="优化器类型")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="梯度裁剪")
    
    # 损失权重参数
    parser.add_argument("--masked_loss_weight", type=float, default=2.0, help="蒙版区域损失权重")
    parser.add_argument("--unmasked_loss_weight", type=float, default=0.1, help="非蒙版区域损失权重")
    
    # 保存和日志参数
    parser.add_argument("--save_steps", type=int, default=500, help="保存间隔")
    parser.add_argument("--eval_steps", type=int, default=250, help="验证间隔")
    parser.add_argument("--logging_steps", type=int, default=50, help="日志间隔")
    parser.add_argument("--save_total_limit", type=int, default=5, help="保存的检查点数量限制")
    
    # 数据处理参数
    parser.add_argument("--resolution", type=int, default=1024, help="图像分辨率")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载工作进程数")
    parser.add_argument("--augmentation_prob", type=float, default=0.5, help="数据增强概率")
    
    # 其他参数
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--use_wandb", action="store_true", help="使用Weights & Biases记录")
    parser.add_argument("--wandb_project", type=str, default="flux-collar-replacement", help="WandB项目名")
    parser.add_argument("--config_file", type=str, help="配置文件路径")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    return parser.parse_args()

def load_config_file(config_path: str) -> dict:
    """加载YAML配置文件"""
    if not os.path.exists(config_path):
        return {}
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config or {}

def merge_configs(args, config_file_dict: dict) -> dict:
    """合并命令行参数和配置文件"""
    config = {}
    
    # 从配置文件加载
    config.update(config_file_dict)
    
    # 命令行参数优先级更高
    for key, value in vars(args).items():
        if value is not None:
            config[key] = value
    
    return config

def validate_config(config: dict) -> bool:
    """验证配置参数"""
    required_keys = ["train_data_dir", "mask_dir", "model_name"]
    
    for key in required_keys:
        if key not in config or not config[key]:
            print(f"缺少必要参数: {key}")
            return False
    
    # 检查数据目录
    if not os.path.exists(config["train_data_dir"]):
        print(f"训练数据目录不存在: {config['train_data_dir']}")
        return False
    
    if not os.path.exists(config["mask_dir"]):
        print(f"蒙版目录不存在: {config['mask_dir']}")
        return False
    
    # 检查验证数据目录（可选）
    if config.get("val_data_dir") and not os.path.exists(config["val_data_dir"]):
        print(f"验证数据目录不存在，将不进行验证: {config['val_data_dir']}")
        config["val_data_dir"] = None
    
    # 验证数值参数
    if config.get("batch_size", 0) <= 0:
        print("batch_size必须大于0")
        return False
    
    if config.get("learning_rate", 0) <= 0:
        print("learning_rate必须大于0")
        return False
    
    if config.get("lora_rank", 0) <= 0:
        print("lora_rank必须大于0")
        return False
    
    return True

def setup_environment(config: dict):
    """设置训练环境"""
    # 设置随机种子
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    
    # 设置多进程
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    # 创建输出目录
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "samples"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
    
    # 保存配置文件
    config_save_path = os.path.join(output_dir, "training_config.yaml")
    with open(config_save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"配置已保存到: {config_save_path}")

def create_tokenizer(model_name: str, cache_dir: str = None):
    """创建tokenizer"""
    try:
        tokenizer = CLIPTokenizer.from_pretrained(
            model_name,
            subfolder="tokenizer",
            cache_dir=cache_dir
        )
        return tokenizer
    except Exception as e:
        print(f"创建tokenizer失败: {e}")
        print("尝试使用默认CLIP tokenizer...")
        
        # 备用方案：使用标准CLIP tokenizer
        tokenizer = CLIPTokenizer.from_pretrained(
            "openai/clip-vit-base-patch32",
            cache_dir=cache_dir
        )
        return tokenizer

def main():
    """主函数"""
    args = parse_args()
    
    # 加载配置文件
    config_file_dict = {}
    if args.config_file:
        config_file_dict = load_config_file(args.config_file)
    
    # 合并配置
    config = merge_configs(args, config_file_dict)
    
    # 验证配置
    if not validate_config(config):
        sys.exit(1)
    
    # 设置日志
    log_file = os.path.join(config["output_dir"], "training.log")
    logger = setup_logging(
        log_level=config.get("log_level", "INFO"),
        log_file=log_file
    )
    
    logger.info("开始FLUX茶歇领替换LoRA训练")
    logger.info(f"配置参数: {config}")
    
    # 设置环境
    setup_environment(config)
    
    # 检查GPU
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"检测到 {gpu_count} 个GPU: {gpu_name}")
        
        # 检查显存
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU显存: {gpu_memory:.1f} GB")
        
        if gpu_memory < 12:
            logger.warning("显存可能不足，建议减少batch_size或启用gradient_checkpointing")
    else:
        logger.warning("未检测到GPU，将使用CPU训练")
    
    try:
        # 创建tokenizer
        logger.info("创建tokenizer...")
        tokenizer = create_tokenizer(config["model_name"], config.get("cache_dir"))
        
        # 创建数据集和数据加载器
        logger.info("创建数据集...")
        train_dataset = PairedCollarDataset(
            data_dir=config["train_data_dir"],
            mask_dir=config["mask_dir"],
            tokenizer=tokenizer,
            resolution=config.get("resolution", 1024),
            is_training=True,
            augmentation_prob=config.get("augmentation_prob", 0.5)
        )
        
        val_dataset = None
        if config.get("val_data_dir"):
            val_dataset = PairedCollarDataset(
                data_dir=config["val_data_dir"],
                mask_dir=config["mask_dir"],
                tokenizer=tokenizer,
                resolution=config.get("resolution", 1024),
                is_training=False,
                augmentation_prob=0.0
            )
        
        # 创建数据加载器
        train_loader, val_loader = create_paired_data_loaders(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            batch_size=config["batch_size"],
            num_workers=config.get("num_workers", 4)
        )
        
        logger.info(f"训练样本: {len(train_dataset)}")
        if val_dataset:
            logger.info(f"验证样本: {len(val_dataset)}")
        
        # 显示数据样本信息
        sample_data = train_dataset.get_sample_data(3)
        logger.info("数据样本:")
        for sample in sample_data:
            logger.info(f"  - {sample['pair_id']}: {sample['target_prompt'][:50]}...")
        
        # 创建FLUX模型
        logger.info("创建FLUX LoRA模型...")
        model = FluxLoRAModel(
            model_name=config["model_name"],
            device="cuda" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.bfloat16 if config.get("mixed_precision") == "bf16" else torch.float16,
            lora_rank=config["lora_rank"],
            lora_alpha=config.get("lora_alpha", 16),
            lora_dropout=config.get("lora_dropout", 0.1),
            use_gradient_checkpointing=True,
            cache_dir=config.get("cache_dir")
        )
        
        # 创建训练器
        logger.info("创建训练器...")
        trainer = FluxCollarTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            output_dir=config["output_dir"],
            resume_from_checkpoint=config.get("resume_from_checkpoint")
        )
        
        # 开始训练
        logger.info("开始训练茶歇领替换模型...")
        trainer.train()
        
        logger.info("训练完成！")
        
    except KeyboardInterrupt:
        logger.info("训练被用户中断")
        sys.exit(0)
    except Exception as e:
        logger.error(f"训练过程中发生错误: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    # 设置多进程启动方法
    mp.set_start_method('spawn', force=True)
    main()