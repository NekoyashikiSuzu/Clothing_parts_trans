#!/usr/bin/env python3
# flux_lora_trainer.py - FLUX-Fill LoRA训练器
import os
import sys
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from PIL import Image
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import math
import random
from datetime import datetime
import shutil
from contextlib import contextmanager
import traceback
import gc

# 添加diffusers和相关库
from diffusers import FluxFillPipeline
from diffusers.utils import make_image_grid
import transformers
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)

class TeaBreakCollarDataset(Dataset):
    """茶歇领数据集"""
    
    def __init__(self, data_dir: str, size: int = 512, center_crop: bool = True):
        self.data_dir = Path(data_dir)
        self.size = size
        self.center_crop = center_crop
        
        # 按照README中的数据结构
        self.image_dir = self.data_dir / "tea_break_collar"
        self.mask_dir = self.data_dir / "masks"
        self.caption_dir = self.data_dir / "captions"
        
        # 验证目录存在
        for dir_path, dir_name in [(self.image_dir, "tea_break_collar"), 
                                   (self.mask_dir, "masks"), 
                                   (self.caption_dir, "captions")]:
            if not dir_path.exists():
                raise ValueError(f"数据目录不存在: {dir_path}")
        
        # 收集所有样本
        self.samples = []
        
        # 支持多种图片格式
        image_extensions = ["*.png", "*.jpg", "*.jpeg"]
        all_image_files = []
        
        for ext in image_extensions:
            all_image_files.extend(self.image_dir.glob(ext))
        
        logger.info(f"在 {self.image_dir} 中找到 {len(all_image_files)} 个图像文件")
        
        for img_file in sorted(all_image_files):
            stem = img_file.stem  # 例如: tea_break_collar_001
            
            # 查找对应的mask文件 (优先.png格式)
            mask_candidates = [
                self.mask_dir / f"{stem}_mask.png",
                self.mask_dir / f"{stem}_mask.jpg", 
                self.mask_dir / f"{stem}_mask.jpeg",
                self.mask_dir / f"{stem}.png",  # 备用命名方式
                self.mask_dir / f"{stem}.jpg"
            ]
            
            mask_file = None
            for candidate in mask_candidates:
                if candidate.exists():
                    mask_file = candidate
                    break
            
            # 查找对应的caption文件
            caption_file = self.caption_dir / f"{stem}.txt"
            
            # 检查文件是否都存在
            if mask_file is None:
                logger.warning(f"未找到mask文件: {stem}")
                continue
                
            if not caption_file.exists():
                logger.warning(f"未找到caption文件: {caption_file}")
                continue
            
            # 验证caption文件内容
            try:
                with open(caption_file, "r", encoding="utf-8") as f:
                    caption_content = f.read().strip()
                if not caption_content:
                    logger.warning(f"Caption文件为空: {caption_file}")
                    continue
            except Exception as e:
                logger.warning(f"读取caption文件失败 {caption_file}: {e}")
                continue
            
            self.samples.append({
                "image": img_file,
                "mask": mask_file,
                "caption": caption_file
            })
        
        logger.info(f"成功加载 {len(self.samples)} 个有效训练样本")
        
        if len(self.samples) == 0:
            raise ValueError(f"未找到有效的训练样本。请检查数据结构:\n"
                           f"  图片目录: {self.image_dir}\n"
                           f"  蒙版目录: {self.mask_dir}\n" 
                           f"  描述目录: {self.caption_dir}\n"
                           f"确保每个图片都有对应的mask和caption文件")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 加载图像
        image = Image.open(sample["image"]).convert("RGB")
        mask = Image.open(sample["mask"]).convert("L")
        
        # 读取描述文本
        with open(sample["caption"], "r", encoding="utf-8") as f:
            caption = f.read().strip()
        
        # 预处理图像
        image, mask = self._preprocess_images(image, mask)
        
        return {
            "pixel_values": image,
            "mask_values": mask,
            "input_ids": caption,
            "file_name": sample["image"].name
        }
    
    def _preprocess_images(self, image: Image.Image, mask: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        """预处理图像和蒙版"""
        # 调整尺寸
        if self.center_crop:
            # 中心裁剪
            image = self._center_crop_resize(image, self.size)
            mask = self._center_crop_resize(mask, self.size)
        else:
            # 直接缩放
            image = image.resize((self.size, self.size), Image.Resampling.LANCZOS)
            mask = mask.resize((self.size, self.size), Image.Resampling.LANCZOS)
        
        # 转换为tensor - 使用float32，稍后在training_step中转换为float16
        image_array = np.array(image).astype(np.float32) / 255.0
        mask_array = np.array(mask).astype(np.float32) / 255.0
        
        # 转换为PyTorch tensor格式 (C, H, W)
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float()
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0).float()
        
        # 归一化到 [-1, 1]
        image_tensor = image_tensor * 2.0 - 1.0
        
        return image_tensor, mask_tensor
    
    def _center_crop_resize(self, image: Image.Image, size: int) -> Image.Image:
        """中心裁剪并调整尺寸"""
        w, h = image.size
        min_dim = min(w, h)
        
        # 中心裁剪为正方形
        left = (w - min_dim) // 2
        top = (h - min_dim) // 2
        right = left + min_dim
        bottom = top + min_dim
        
        image = image.crop((left, top, right, bottom))
        
        # 调整到目标尺寸
        image = image.resize((size, size), Image.Resampling.LANCZOS)
        
        return image
    
    def validate_dataset(self):
        """验证数据集质量"""
        logger.info("验证数据集质量...")
        
        issues = []
        valid_samples = 0
        
        for i, sample in enumerate(self.samples[:10]):  # 检查前10个样本
            try:
                # 检查图片
                image = Image.open(sample["image"]).convert("RGB")
                if image.size[0] < 256 or image.size[1] < 256:
                    issues.append(f"图片 {sample['image'].name} 尺寸过小: {image.size}")
                
                # 检查mask
                mask = Image.open(sample["mask"]).convert("L") 
                if mask.size != image.size:
                    issues.append(f"图片和mask尺寸不匹配: {sample['image'].name}")
                
                # 检查mask是否有内容
                mask_array = np.array(mask)
                white_pixels = np.sum(mask_array > 128)
                total_pixels = mask_array.size
                mask_ratio = white_pixels / total_pixels
                
                if mask_ratio < 0.001:  # mask区域少于0.1%
                    issues.append(f"Mask区域过小: {sample['mask'].name} ({mask_ratio:.1%})")
                elif mask_ratio > 0.5:  # mask区域超过50%
                    issues.append(f"Mask区域过大: {sample['mask'].name} ({mask_ratio:.1%})")
                
                # 检查caption
                with open(sample["caption"], "r", encoding="utf-8") as f:
                    caption = f.read().strip().lower()
                
                # 检查是否包含关键词
                if "tea break collar" not in caption and "v-shaped" not in caption:
                    issues.append(f"Caption缺少关键词: {sample['caption'].name}")
                
                valid_samples += 1
                
            except Exception as e:
                issues.append(f"样本 {i+1} 验证失败: {e}")
        
        if issues:
            logger.warning("发现数据质量问题:")
            for issue in issues[:5]:  # 只显示前5个问题
                logger.warning(f"  - {issue}")
            if len(issues) > 5:
                logger.warning(f"  - ... 还有 {len(issues)-5} 个问题")
        
        logger.info(f"数据验证完成: {valid_samples}/{min(10, len(self.samples))} 个样本有效")
        return len(issues) == 0

class FluxFillLoRATrainer:
    """FLUX-Fill LoRA训练器 - 优化数据类型匹配"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # RTX 4090 优化配置
        self.target_dtype = torch.float16
        self.compute_dtype = torch.float16
        
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config["gradient_accumulation_steps"],
            mixed_precision=config["mixed_precision"],
            log_with="tensorboard" if config["logging"]["use_tensorboard"] else None,
            project_dir=config["output_dir"],
        )
        
        # 设置随机种子
        if config["seed"] is not None:
            set_seed(config["seed"])
        
        # 设置日志
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(f"开始训练 - 加速器设备: {self.accelerator.device}")
        
        # 硬件检查
        self._check_hardware()
        
        # 创建输出目录
        Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)
        
        # 保存配置
        with open(Path(config["output_dir"]) / "training_config.json", "w") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    
    def _check_hardware(self):
        """检查硬件配置"""
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            logger.info(f"GPU: {gpu_name}")
            logger.info(f"显存: {total_memory:.1f}GB")
            logger.info(f"目标数据类型: {self.target_dtype}")
            
            if "4090" in gpu_name:
                logger.info("检测到RTX 4090，启用专用优化")
        else:
            logger.warning("未检测到CUDA，训练可能很慢")
    
    def _ensure_dtype_consistency(self, model, model_name=""):
        """确保模型数据类型一致性"""
        logger.info(f"统一 {model_name} 数据类型到 {self.target_dtype}...")
        
        converted_params = 0
        converted_buffers = 0
        
        # 转换所有参数
        for name, param in model.named_parameters():
            if param.dtype != self.target_dtype:
                param.data = param.data.to(self.target_dtype)
                converted_params += 1
        
        # 转换所有缓冲区
        for name, buffer in model.named_buffers():
            if hasattr(buffer, 'dtype') and buffer.dtype != self.target_dtype:
                buffer.data = buffer.data.to(self.target_dtype)
                converted_buffers += 1
        
        logger.info(f"{model_name} 类型转换完成: {converted_params} 参数, {converted_buffers} 缓冲区")
        
        # 移动到GPU
        model = model.to(self.accelerator.device)
        
        return model
    
    @contextmanager
    def error_handler(self, operation: str):
        """增强的错误处理"""
        try:
            logger.info(f"执行: {operation}")
            yield
            logger.info(f"完成: {operation}")
        except Exception as e:
            error_msg = str(e)
            logger.error(f"错误 [{operation}]: {type(e).__name__}")
            logger.error(f"错误信息: {error_msg}")
            logger.error(f"完整错误栈:\n{traceback.format_exc()}")
            
            # 数据类型错误专项诊断
            if any(keyword in error_msg for keyword in ["Input type", "should be the same", "dtype"]):
                logger.error("*** 数据类型不匹配错误诊断 ***")
                logger.error("可能原因:")
                logger.error("1. 模型组件数据类型不一致")
                logger.error("2. 输入张量与模型参数类型不匹配")
                logger.error("3. CPU张量与GPU张量混合")
                logger.error("4. LoRA层初始化类型错误")
                
                # 提供解决建议
                logger.error("解决建议:")
                logger.error("- 检查所有模型组件是否使用相同数据类型")
                logger.error("- 确保输入数据类型与模型一致")
                logger.error("- 验证LoRA适配器类型匹配")
            
            # 清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                current_memory = torch.cuda.memory_allocated() / 1024**3
                logger.info(f"当前GPU内存使用: {current_memory:.2f}GB")
            
            raise
    
    def setup_models(self):
        """设置模型"""
        with self.error_handler("模型加载"):
            logger.info("加载FLUX-Fill模型...")
            
            # 加载预训练的FLUX-Fill管道
            self.pipe = FluxFillPipeline.from_pretrained(
                self.config["pretrained_model_name_or_path"],
                torch_dtype=self.target_dtype,  # 统一数据类型
                use_safetensors=True,
            )
            
            # 获取模型组件 - FLUX使用transformer而不是unet
            self.transformer = self.pipe.transformer
            self.vae = self.pipe.vae
            self.text_encoder = self.pipe.text_encoder
            self.text_encoder_2 = self.pipe.text_encoder_2
            self.tokenizer = self.pipe.tokenizer
            self.tokenizer_2 = self.pipe.tokenizer_2
            
            # 强制统一所有模型组件的数据类型
            self.transformer = self._ensure_dtype_consistency(self.transformer, "Transformer")
            self.vae = self._ensure_dtype_consistency(self.vae, "VAE")
            self.text_encoder = self._ensure_dtype_consistency(self.text_encoder, "TextEncoder")
            self.text_encoder_2 = self._ensure_dtype_consistency(self.text_encoder_2, "TextEncoder2")
            
            # 设置训练/评估模式
            self.transformer.train()
            self.vae.eval()
            self.text_encoder.eval()
            self.text_encoder_2.eval()
            
            # 冻结不需要训练的模型
            for param in self.vae.parameters():
                param.requires_grad = False
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            for param in self.text_encoder_2.parameters():
                param.requires_grad = False
        
        with self.error_handler("LoRA设置"):
            # 配置LoRA - 针对FLUX Transformer的目标模块
            lora_config = LoraConfig(
                r=self.config["lora"]["rank"],
                lora_alpha=self.config["lora"]["alpha"],
                target_modules=self.config["lora"]["target_modules"],
                lora_dropout=self.config["lora"]["dropout"],
                bias="none",
            )
            
            # 应用LoRA到Transformer
            self.transformer = get_peft_model(self.transformer, lora_config)
            
            # 确保LoRA层也使用正确的数据类型
            self.transformer = self._ensure_dtype_consistency(self.transformer, "LoRA-Transformer")
            
            logger.info(f"LoRA配置: rank={lora_config.r}, alpha={lora_config.lora_alpha}")
            logger.info(f"可训练参数: {sum(p.numel() for p in self.transformer.parameters() if p.requires_grad):,}")
    
    def setup_dataset_and_dataloader(self):
        """设置数据集和数据加载器"""
        logger.info("设置数据集...")
        
        # 创建数据集
        self.train_dataset = TeaBreakCollarDataset(
            data_dir=self.config["data_dir"],
            size=self.config["resolution"],
            center_crop=self.config["center_crop"]
        )
        
        # 验证数据集质量
        self.train_dataset.validate_dataset()
        
        # 创建数据加载器
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config["train_batch_size"],
            shuffle=True,
            num_workers=self.config["dataloader_num_workers"],
            pin_memory=True,
        )
        
        logger.info(f"训练样本数: {len(self.train_dataset)}")
        logger.info(f"批次大小: {self.config['train_batch_size']}")
        logger.info(f"每轮步数: {len(self.train_dataloader)}")
        
        # 显示一些样本信息
        sample_info = []
        for i in range(min(3, len(self.train_dataset.samples))):
            sample = self.train_dataset.samples[i]
            
            # 读取caption
            with open(sample["caption"], "r", encoding="utf-8") as f:
                caption = f.read().strip()
            
            sample_info.append({
                "image": sample["image"].name,
                "mask": sample["mask"].name, 
                "caption": caption[:50] + "..." if len(caption) > 50 else caption
            })
        
        logger.info("样本示例:")
        for i, info in enumerate(sample_info):
            logger.info(f"  样本 {i+1}:")
            logger.info(f"    图片: {info['image']}")
            logger.info(f"    蒙版: {info['mask']}")
            logger.info(f"    描述: {info['caption']}")
    
    def setup_optimizer_and_scheduler(self):
        """设置优化器和调度器"""
        # 优化器
        self.optimizer = AdamW(
            self.transformer.parameters(),
            lr=self.config["learning_rate"],
            betas=(self.config["adam_beta1"], self.config["adam_beta2"]),
            weight_decay=self.config["adam_weight_decay"],
            eps=self.config["adam_epsilon"],
        )
        
        # 学习率调度器
        num_training_steps = self.config["max_train_steps"]
        num_warmup_steps = self.config["lr_warmup_steps"]
        
        self.lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        
        logger.info(f"优化器: AdamW, 学习率: {self.config['learning_rate']}")
        logger.info(f"总训练步数: {num_training_steps}, 预热步数: {num_warmup_steps}")
    
    def encode_prompt(self, prompt: str):
        """编码提示词 - 支持双编码器"""
        # CLIP编码器
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,  # CLIP标准长度
            truncation=True,
            return_tensors="pt",
        )
        
        with torch.no_grad():
            text_embeddings = self.text_encoder(
                text_inputs.input_ids.to(self.accelerator.device, dtype=torch.long)
            )[0].to(self.target_dtype)
        
        # T5编码器
        text_inputs_2 = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=256,  # T5更长的序列
            truncation=True,
            return_tensors="pt",
        )
        
        with torch.no_grad():
            text_embeddings_2 = self.text_encoder_2(
                text_inputs_2.input_ids.to(self.accelerator.device, dtype=torch.long)
            )[0].to(self.target_dtype)
            
            # 提取pooled embeddings
            pooled_prompt_embeds = text_embeddings_2.mean(dim=1)
        
        return text_embeddings, pooled_prompt_embeds
    
    def training_step(self, batch):
        """单步训练 - 优化数据类型处理"""
        with self.error_handler("训练步骤"):
            # 确保所有输入数据类型一致
            pixel_values = batch["pixel_values"].to(
                self.accelerator.device, dtype=self.target_dtype
            )
            mask_values = batch["mask_values"].to(
                self.accelerator.device, dtype=self.target_dtype
            )
            prompts = batch["input_ids"]
            
            batch_size = pixel_values.shape[0]
            
            # 编码图像到潜在空间
            with torch.no_grad():
                latents = self.vae.encode(pixel_values).latent_dist.sample()
                latents = latents * self.vae.config.scaling_factor
                latents = latents.to(self.target_dtype)
            
            # 编码蒙版到潜在空间尺寸
            mask_latents = F.interpolate(
                mask_values, 
                size=(latents.shape[-2], latents.shape[-1]), 
                mode="nearest"
            ).to(self.target_dtype)
            
            # 添加噪声 - 确保类型一致
            noise = torch.randn_like(latents, dtype=self.target_dtype, device=latents.device)
            timesteps = torch.randint(
                0, self.pipe.scheduler.config.num_train_timesteps, 
                (batch_size,), device=latents.device, dtype=torch.long
            )
            
            noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timesteps)
            noisy_latents = noisy_latents.to(self.target_dtype)
            
            # 编码提示词 - FLUX双编码器
            prompt_embeds_list = []
            pooled_prompt_embeds_list = []
            
            for prompt in prompts:
                prompt_embeds, pooled_prompt_embeds = self.encode_prompt(prompt)
                prompt_embeds_list.append(prompt_embeds)
                pooled_prompt_embeds_list.append(pooled_prompt_embeds)
            
            prompt_embeds = torch.stack(prompt_embeds_list, dim=0).squeeze(1).to(self.target_dtype)
            pooled_prompt_embeds = torch.stack(pooled_prompt_embeds_list, dim=0).squeeze(1).to(self.target_dtype)
            
            # 准备输入 - FLUX Fill需要拼接蒙版
            model_input = torch.cat([noisy_latents, mask_latents], dim=1).to(self.target_dtype)
            
            # 前向传播 - FLUX transformer
            try:
                model_pred = self.transformer(
                    hidden_states=model_input,
                    timestep=timesteps, 
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    return_dict=False,
                )[0]
                
                # 确保输出类型正确
                model_pred = model_pred.to(self.target_dtype)
                
            except Exception as e:
                logger.error(f"Transformer前向传播失败: {e}")
                logger.error(f"输入形状和类型:")
                logger.error(f"  model_input: {model_input.shape}, {model_input.dtype}")
                logger.error(f"  timesteps: {timesteps.shape}, {timesteps.dtype}")
                logger.error(f"  prompt_embeds: {prompt_embeds.shape}, {prompt_embeds.dtype}")
                logger.error(f"  pooled_prompt_embeds: {pooled_prompt_embeds.shape}, {pooled_prompt_embeds.dtype}")
                raise
            
            # 计算损失
            if self.pipe.scheduler.config.prediction_type == "epsilon":
                target = noise
            elif self.pipe.scheduler.config.prediction_type == "v_prediction":
                target = self.pipe.scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"未知的预测类型: {self.pipe.scheduler.config.prediction_type}")
            
            target = target.to(self.target_dtype)
            
            # 只在蒙版区域计算损失
            mask_loss_weight = mask_latents.repeat(1, latents.shape[1], 1, 1).to(self.target_dtype)
            
            # 确保预测和目标形状匹配
            if model_pred.shape != target.shape:
                logger.warning(f"形状不匹配: pred={model_pred.shape}, target={target.shape}")
                # FLUX可能有不同的输出通道数，需要调整
                if model_pred.shape[1] > target.shape[1]:
                    model_pred = model_pred[:, :target.shape[1]]
                elif model_pred.shape[1] < target.shape[1]:
                    # 如果预测通道少，重复最后几个通道
                    channels_to_add = target.shape[1] - model_pred.shape[1]
                    model_pred = torch.cat([model_pred, model_pred[:, -channels_to_add:]], dim=1)
            
            # 计算损失 - 使用float32避免精度问题
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
            loss = loss * mask_loss_weight.float()
            loss = loss.mean()
            
            return loss
    
    def save_model(self, step: int, is_final: bool = False):
        """保存模型"""
        if is_final:
            save_dir = Path(self.config["output_dir"]) / "final_model"
        else:
            save_dir = Path(self.config["output_dir"]) / f"checkpoint-{step}"
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存LoRA权重
        self.transformer.save_pretrained(save_dir)
        
        # 保存训练配置
        with open(save_dir / "training_config.json", "w") as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"模型已保存到: {save_dir}")
        
        # 生成测试图像
        if self.config["validation"]["enabled"]:
            self.generate_validation_images(save_dir, step)
    
    def generate_validation_images(self, save_dir: Path, step: int):
        """生成验证图像"""
        logger.info("生成验证图像...")
        
        try:
            # 创建验证管道
            validation_pipe = FluxFillPipeline(
                vae=self.vae,
                text_encoder=self.text_encoder,
                text_encoder_2=self.text_encoder_2,
                tokenizer=self.tokenizer,
                tokenizer_2=self.tokenizer_2,
                transformer=self.accelerator.unwrap_model(self.transformer),
                scheduler=self.pipe.scheduler,
            )
            validation_pipe = validation_pipe.to(self.accelerator.device)
            validation_pipe.set_progress_bar_config(disable=True)
            
            # 确保验证管道的数据类型一致
            validation_pipe.transformer = self._ensure_dtype_consistency(validation_pipe.transformer, "Validation-Transformer")
            
            # 准备验证数据
            validation_dir = save_dir / "validation"
            validation_dir.mkdir(exist_ok=True)
            
            # 使用训练数据中的一些样本进行验证
            val_samples = random.sample(self.train_dataset.samples, min(4, len(self.train_dataset.samples)))
            
            for i, sample in enumerate(val_samples):
                # 加载原图和蒙版
                image = Image.open(sample["image"]).convert("RGB")
                mask = Image.open(sample["mask"]).convert("L")
                
                # 读取提示词
                with open(sample["caption"], "r", encoding="utf-8") as f:
                    prompt = f.read().strip()
                
                # 调整尺寸
                image = image.resize((512, 512), Image.Resampling.LANCZOS)
                mask = mask.resize((512, 512), Image.Resampling.LANCZOS)
                
                # 生成图像 - 使用正确的数据类型
                with torch.autocast("cuda", dtype=self.target_dtype):
                    generated_image = validation_pipe(
                        image=image,
                        mask_image=mask,
                        prompt=prompt,
                        num_inference_steps=20,
                        guidance_scale=3.5,
                        generator=torch.Generator().manual_seed(42)
                    ).images[0]
                
                # 保存结果
                comparison = make_image_grid([image, mask.convert("RGB"), generated_image], rows=1, cols=3)
                comparison.save(validation_dir / f"step_{step}_sample_{i}.png")
            
            del validation_pipe
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.warning(f"验证图像生成失败: {e}")
            logger.warning(f"错误详情: {traceback.format_exc()}")
    
    def train(self):
        """开始训练"""
        logger.info("开始训练...")
        
        # 设置模型、数据集、优化器
        self.setup_models()
        self.setup_dataset_and_dataloader()
        self.setup_optimizer_and_scheduler()
        
        # 使用accelerator准备 - 确保数据类型一致性
        with self.error_handler("Accelerator准备"):
            self.transformer, self.optimizer, self.train_dataloader, self.lr_scheduler = self.accelerator.prepare(
                self.transformer, self.optimizer, self.train_dataloader, self.lr_scheduler
            )
            
            # 再次确保准备后的模型数据类型正确
            self.transformer = self._ensure_dtype_consistency(self.transformer, "Prepared-Transformer")
        
        # 训练循环
        global_step = 0
        progress_bar = tqdm(
            range(self.config["max_train_steps"]),
            disable=not self.accelerator.is_local_main_process,
        )
        progress_bar.set_description("训练")
        
        for epoch in range(self.config["num_train_epochs"]):
            for step, batch in enumerate(self.train_dataloader):
                with self.accelerator.accumulate(self.transformer):
                    # 训练步骤
                    loss = self.training_step(batch)
                    
                    # 反向传播
                    self.accelerator.backward(loss)
                    
                    # 梯度裁剪
                    if self.config["max_grad_norm"] is not None:
                        self.accelerator.clip_grad_norm_(self.transformer.parameters(), self.config["max_grad_norm"])
                    
                    # 更新参数
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                
                # 更新进度
                progress_bar.update(1)
                global_step += 1
                
                # 记录损失
                logs = {
                    "loss": loss.detach().item(),
                    "lr": self.lr_scheduler.get_last_lr()[0],
                    "step": global_step,
                }
                progress_bar.set_postfix(**logs)
                self.accelerator.log(logs, step=global_step)
                
                # 定期检查GPU内存使用情况
                if global_step % 50 == 0 and torch.cuda.is_available():
                    current_memory = torch.cuda.memory_allocated() / 1024**3
                    max_memory = torch.cuda.max_memory_allocated() / 1024**3
                    logger.info(f"GPU内存使用: {current_memory:.2f}GB / 峰值: {max_memory:.2f}GB")
                
                # 保存检查点
                if global_step % self.config["checkpointing_steps"] == 0:
                    if self.accelerator.is_main_process:
                        self.save_model(global_step)
                
                # 达到最大步数
                if global_step >= self.config["max_train_steps"]:
                    break
            
            if global_step >= self.config["max_train_steps"]:
                break
        
        # 保存最终模型
        if self.accelerator.is_main_process:
            self.save_model(global_step, is_final=True)
        
        self.accelerator.wait_for_everyone()
        self.accelerator.end_training()
        
        logger.info("训练完成!")

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """线性学习率调度器"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def create_training_config():
    """创建训练配置"""
    config = {
        # 基本配置
        "pretrained_model_name_or_path": "./models/flux-fill",
        "data_dir": "./training_data",
        "output_dir": "./lora_output",
        "seed": 42,
        
        # 数据配置
        "resolution": 512,
        "center_crop": True,
        "train_batch_size": 1,  # RTX 4090可以使用更大的批次
        "dataloader_num_workers": 4,
        
        # LoRA配置
        "lora": {
            "rank": 16,
            "alpha": 32,
            "target_modules": [
                # 基于实际FLUX模型结构的配置
                "to_q", "to_k", "to_v",           # 主要注意力模块  
                "proj_out",                        # 输出投影
                "proj",                           # 通用投影层
                "add_q_proj", "add_k_proj", "add_v_proj"  # 额外的注意力投影
            ],
            "dropout": 0.1,
        },
        
        # 训练配置
        "num_train_epochs": 100,
        "max_train_steps": 1500,  # 150样本 * 10轮
        "gradient_accumulation_steps": 4,
        "learning_rate": 1e-4,
        "lr_warmup_steps": 100,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_weight_decay": 1e-2,
        "adam_epsilon": 1e-8,
        "max_grad_norm": 1.0,
        
        # 保存配置
        "checkpointing_steps": 500,
        "mixed_precision": "fp16",
        
        # 验证配置
        "validation": {
            "enabled": True,
            "num_samples": 4,
        },
        
        # 日志配置
        "logging": {
            "use_tensorboard": True,
            "log_interval": 10,
        }
    }
    
    return config

def main():
    """主函数"""
    print("=== FLUX-Fill LoRA训练器 ===")
    
    # 检查CUDA
    if not torch.cuda.is_available():
        print("警告: 未检测到CUDA，训练可能会很慢")
    else:
        print(f"使用GPU: {torch.cuda.get_device_name()}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    # 检查数据目录结构
    data_dir = Path("training_data")
    print(f"\n检查数据结构...")
    
    if not data_dir.exists():
        print(f"数据目录不存在: {data_dir}")
        return False
    
    # 检查各个子目录
    required_dirs = {
        "tea_break_collar": "原始图片",
        "masks": "蒙版图片", 
        "captions": "描述文本"
    }
    
    all_dirs_exist = True
    for subdir, desc in required_dirs.items():
        subdir_path = data_dir / subdir
        if not subdir_path.exists():
            print(f"{desc}目录不存在: {subdir_path}")
            all_dirs_exist = False
        else:
            # 统计文件数量
            file_count = len(list(subdir_path.glob("*")))
            print(f"{desc}: {file_count} 个文件")
            
            if file_count == 0:
                print(f"{desc}目录为空")
                all_dirs_exist = False
    
    if not all_dirs_exist:
        print("\n请确保数据结构正确后重新运行")
        return False
    
    # 检查模型
    model_dir = Path("models/flux-fill")
    if not model_dir.exists():
        print(f"\n模型目录不存在: {model_dir}")
        return False
    
    # 检查模型文件
    model_files = ["model_index.json", "transformer/config.json"]  # FLUX使用transformer而不是unet
    missing_files = []
    for file_name in model_files:
        if not (model_dir / file_name).exists():
            missing_files.append(file_name)
    
    if missing_files:
        print(f"缺少模型文件: {missing_files}")
        return False
    
    print("模型检查通过")
    
    # 创建配置
    config = create_training_config()
    
    # 显示训练配置
    print(f"\n训练配置:")
    print(f"  数据目录: {config['data_dir']}")
    print(f"  输出目录: {config['output_dir']}")
    print(f"  批次大小: {config['train_batch_size']}")
    print(f"  最大步数: {config['max_train_steps']}")
    print(f"  学习率: {config['learning_rate']}")
    print(f"  LoRA rank: {config['lora']['rank']}")
    print(f"  分辨率: {config['resolution']}")
    print(f"  混合精度: {config['mixed_precision']}")
    
    # 估算训练时间
    try:
        # 粗略估算数据样本数
        img_count = len(list((data_dir / "tea_break_collar").glob("*")))
        steps_per_epoch = max(1, img_count // config['train_batch_size'])
        total_epochs = config['max_train_steps'] // steps_per_epoch
        
        print(f"\n训练估算:")
        print(f"  样本数: ~{img_count}")
        print(f"  每轮步数: ~{steps_per_epoch}")
        print(f"  总轮数: ~{total_epochs}")
        print(f"  预计时间: {config['max_train_steps'] * 0.5 / 60:.0f}-{config['max_train_steps'] * 2 / 60:.0f} 分钟")
    except:
        pass
    
    # 询问是否开始训练
    response = input("\n是否开始训练? (y/n): ")
    if response.lower() != 'y':
        print("训练已取消")
        return False
    
    try:
        # 开始训练
        print("\n" + "="*50)
        print("开始训练...")
        print("="*50)
        
        trainer = FluxFillLoRATrainer(config)
        trainer.train()
        
        print(f"\n训练完成!")
        print(f"LoRA模型保存在: {config['output_dir']}/final_model")
        print(f"\n使用方法:")
        print(f"python lora_inference.py")
        print(f"\n或者在代码中使用:")
        print(f"""
from peft import PeftModel
from diffusers import FluxFillPipeline

# 加载基础模型
pipe = FluxFillPipeline.from_pretrained("./models/flux-fill")

# 加载LoRA
pipe.transformer = PeftModel.from_pretrained(pipe.transformer, "{config['output_dir']}/final_model")

# 使用时在提示词中包含 "V-shaped tea break collar"
""")
        
        return True
        
    except KeyboardInterrupt:
        print("\n训练被用户中断")
        return False
    except Exception as e:
        print(f"\n训练失败: {e}")
        print(f"完整错误信息:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)