#!/usr/bin/env python3
# scripts/convert_checkpoint.py - 检查点转换脚本
import os
import sys
import argparse
import torch
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.flux_model import FluxLoRAModel
from utils.model_utils import get_model_size

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="FLUX LoRA检查点转换工具")
    
    parser.add_argument("--input_checkpoint", type=str, required=True, help="输入检查点路径")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--base_model", type=str, default="black-forest-labs/FLUX.1-fill-dev", help="基础模型")
    parser.add_argument("--format", type=str, default="safetensors", choices=["safetensors", "pytorch"], help="输出格式")
    parser.add_argument("--merge_weights", action="store_true", help="合并LoRA权重到基础模型")
    parser.add_argument("--float16", action="store_true", help="转换为float16精度")
    
    return parser.parse_args()

def load_checkpoint(checkpoint_path: str):
    """加载检查点"""
    print(f"加载检查点: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print(f"检查点加载成功")
    
    # 显示检查点信息
    if 'step' in checkpoint:
        print(f"  训练步数: {checkpoint['step']}")
    if 'epoch' in checkpoint:
        print(f"  训练轮数: {checkpoint['epoch']}")
    if 'best_val_loss' in checkpoint:
        print(f"  最佳验证损失: {checkpoint['best_val_loss']:.4f}")
    
    return checkpoint

def convert_to_lora_only(checkpoint, output_path: str, format_type: str = "safetensors"):
    """转换为纯LoRA权重"""
    print(f"🔧 转换为LoRA权重格式...")
    
    # 提取LoRA相关的权重
    lora_state_dict = {}
    model_state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    for key, value in model_state_dict.items():
        if 'lora_' in key or 'peft' in key:
            lora_state_dict[key] = value
    
    if not lora_state_dict:
        print("⚠️ 未找到LoRA权重，尝试查找其他适配器权重...")
        # 查找其他可能的适配器权重
        for key, value in model_state_dict.items():
            if any(pattern in key.lower() for pattern in ['adapter', 'lora', 'delta']):
                lora_state_dict[key] = value
    
    if not lora_state_dict:
        raise ValueError("检查点中未找到LoRA权重")
    
    print(f"  找到 {len(lora_state_dict)} 个LoRA参数")
    
    # 保存LoRA权重
    if format_type == "safetensors":
        try:
            from safetensors.torch import save_file
            save_file(lora_state_dict, output_path)
        except ImportError:
            print("⚠️ safetensors未安装，使用PyTorch格式保存")
            torch.save(lora_state_dict, output_path.replace('.safetensors', '.pt'))
    else:
        torch.save(lora_state_dict, output_path)
    
    print(f"LoRA权重已保存: {output_path}")

def merge_lora_weights(checkpoint, base_model_name: str, output_dir: str):
    """合并LoRA权重到基础模型"""
    print(f"合并LoRA权重到基础模型...")
    
    # 加载基础模型
    model = FluxLoRAModel(
        model_name=base_model_name,
        device='cpu'
    )
    
    # 加载LoRA权重
    model.load_state_dict(checkpoint.get('model_state_dict', checkpoint), strict=False)
    
    # 合并权重
    merged_model = model.merge_and_unload()
    
    # 保存合并后的模型
    merged_path = os.path.join(output_dir, "merged_model")
    merged_model.save_pretrained(merged_path)
    
    print(f"合并后的模型已保存: {merged_path}")
    
    return merged_path

def convert_precision(state_dict, target_dtype=torch.float16):
    """转换模型精度"""
    print(f"转换精度到: {target_dtype}")
    
    converted_dict = {}
    for key, value in state_dict.items():
        if value.dtype.is_floating_point:
            converted_dict[key] = value.to(target_dtype)
        else:
            converted_dict[key] = value
    
    return converted_dict

def validate_converted_model(model_path: str, base_model_name: str):
    """验证转换后的模型"""
    print(f"验证转换后的模型...")
    
    try:
        # 尝试加载模型
        model = FluxLoRAModel(
            model_name=base_model_name,
            device='cpu'
        )
        
        if os.path.isdir(model_path):
            model.load_lora_weights(model_path)
        else:
            checkpoint = torch.load(model_path, map_location='cpu')
            model.load_state_dict(checkpoint, strict=False)
        
        # 获取模型信息
        model_info = get_model_size(model)
        print(f"  模型参数数: {model_info['total_parameters']:,}")
        print(f"  可训练参数: {model_info['trainable_parameters']:,}")
        
        print("模型验证通过")
        return True
        
    except Exception as e:
        print(f"模型验证失败: {e}")
        return False

def main():
    """主函数"""
    args = parse_args()
    
    print("FLUX LoRA检查点转换工具")
    print("=" * 40)
    
    try:
        # 加载原始检查点
        checkpoint = load_checkpoint(args.input_checkpoint)
        
        # 创建输出目录
        os.makedirs(args.output_dir, exist_ok=True)
        
        # 转换精度（如果需要）
        if args.float16:
            print("转换为float16精度...")
            if 'model_state_dict' in checkpoint:
                checkpoint['model_state_dict'] = convert_precision(
                    checkpoint['model_state_dict'], 
                    torch.float16
                )
            else:
                checkpoint = convert_precision(checkpoint, torch.float16)
        
        if args.merge_weights:
            # 合并权重到基础模型
            merged_path = merge_lora_weights(
                checkpoint, 
                args.base_model, 
                args.output_dir
            )
            
            # 验证合并后的模型
            validate_converted_model(merged_path, args.base_model)
            
        else:
            # 转换为纯LoRA格式
            if args.format == "safetensors":
                output_file = os.path.join(args.output_dir, "pytorch_lora_weights.safetensors")
            else:
                output_file = os.path.join(args.output_dir, "pytorch_lora_weights.pt")
            
            convert_to_lora_only(checkpoint, output_file, args.format)
            
            # 验证转换后的LoRA权重
            validate_converted_model(output_file, args.base_model)
        
        # 保存元数据
        metadata = {
            'original_checkpoint': args.input_checkpoint,
            'base_model': args.base_model,
            'conversion_format': args.format,
            'merged_weights': args.merge_weights,
            'float16': args.float16
        }
        
        if 'config' in checkpoint:
            metadata['training_config'] = checkpoint['config']
        
        metadata_path = os.path.join(args.output_dir, "conversion_metadata.json")
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"元数据已保存: {metadata_path}")
        
        print("\n检查点转换完成！")
        print(f"输出目录: {args.output_dir}")
        
        # 显示使用说明
        if args.merge_weights:
            print("\n使用合并后的模型:")
            print(f"  from diffusers import FluxFillPipeline")
            print(f"  pipeline = FluxFillPipeline.from_pretrained('{os.path.join(args.output_dir, 'merged_model')}')")
        else:
            print("\n使用LoRA权重:")
            print(f"  model.load_lora_weights('{args.output_dir}')")
        
    except Exception as e:
        print(f"转换失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()