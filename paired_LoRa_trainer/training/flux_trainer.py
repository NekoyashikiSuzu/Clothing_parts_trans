# training/flux_trainer.py - FLUX茶歇领替换训练器
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Optional, Any, List
from tqdm import tqdm
import logging
import numpy as np
from accelerate import Accelerator
from diffusers import FlowMatchEulerDiscreteScheduler
import wandb

from models.flux_model import FluxLoRAModel
from utils.checkpoint_utils import CheckpointManager
from utils.visualization import tensor_to_pil, save_comparison_grid

logger = logging.getLogger(__name__)

class FluxCollarTrainer:
    """FLUX茶歇领替换LoRA训练器"""
    
    def __init__(
        self,
        model: FluxLoRAModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        output_dir: str = "./outputs",
        resume_from_checkpoint: Optional[str] = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.output_dir = output_dir
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "samples"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
        
        # 初始化Accelerator
        self.accelerator = Accelerator(
            mixed_precision=config.get("mixed_precision", "bf16"),
            gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
            log_with="tensorboard" if not config.get("use_wandb", False) else "wandb",
            project_dir=output_dir
        )
        
        # 设置设备
        self.device = self.accelerator.device
        
        # 初始化噪声调度器
        self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model.model_name, 
            subfolder="scheduler"
        )
        
        # 初始化优化器
        self.optimizer = self._create_optimizer()
        
        # 初始化学习率调度器
        self.lr_scheduler = self._create_lr_scheduler()
        
        # 准备模型、优化器和数据加载器
        (
            self.model,
            self.optimizer,
            self.train_loader,
            self.val_loader,
            self.lr_scheduler
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.train_loader,
            self.val_loader,
            self.lr_scheduler
        )
        
        # 初始化检查点管理器
        self.checkpoint_manager = CheckpointManager(
            output_dir=os.path.join(output_dir, "checkpoints"),
            save_total_limit=config.get("save_total_limit", 5)
        )
        
        # 训练状态
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # 日志记录
        if self.accelerator.is_main_process:
            self.writer = SummaryWriter(os.path.join(output_dir, "logs"))
            
            # 初始化wandb（如果启用）
            if config.get("use_wandb", False):
                wandb.init(
                    project="flux-collar-replacement",
                    config=config,
                    name=f"collar_lora_{config.get('lora_rank', 32)}"
                )
        
        # 从检查点恢复
        if resume_from_checkpoint:
            self._load_checkpoint(resume_from_checkpoint)
        
        logger.info("训练器初始化完成")
    
    def _create_optimizer(self):
        """创建优化器"""
        optimizer_type = self.config.get("optimizer", "adamw")
        learning_rate = self.config.get("learning_rate", 5e-5)
        weight_decay = self.config.get("weight_decay", 0.01)
        
        if optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                betas=(0.9, 0.999),
                weight_decay=weight_decay,
                eps=1e-8
            )
        elif optimizer_type == "adamw_8bit":
            try:
                import bitsandbytes as bnb
                optimizer = bnb.optim.AdamW8bit(
                    self.model.parameters(),
                    lr=learning_rate,
                    betas=(0.9, 0.999),
                    weight_decay=weight_decay,
                    eps=1e-8
                )
            except ImportError:
                logger.warning("bitsandbytes未安装，使用标准AdamW")
                optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=learning_rate,
                    weight_decay=weight_decay
                )
        else:
            raise ValueError(f"不支持的优化器类型: {optimizer_type}")
        
        return optimizer
    
    def _create_lr_scheduler(self):
        """创建学习率调度器"""
        scheduler_type = self.config.get("lr_scheduler", "cosine_with_restarts")
        warmup_steps = self.config.get("warmup_steps", 300)
        max_steps = self.config.get("max_steps", 3000)
        
        if scheduler_type == "cosine_with_restarts":
            from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
            scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=max_steps,
                num_cycles=self.config.get("lr_num_cycles", 1)
            )
        elif scheduler_type == "linear":
            from transformers import get_linear_schedule_with_warmup
            scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=max_steps
            )
        else:
            scheduler = None
        
        return scheduler
    
    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """计算茶歇领替换的损失"""
        # 获取批次数据
        target_images = batch["target_image"]
        masks = batch["mask"]
        input_ids = batch["input_ids"]
        
        batch_size = target_images.shape[0]
        
        # 编码目标图像到潜在空间（这是我们要学习生成的茶歇领风格）
        with torch.no_grad():
            target_latents = self.model.encode_images(target_images)
        
        # 生成噪声
        noise = torch.randn_like(target_latents)
        
        # 随机时间步
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (batch_size,), device=self.device
        ).long()
        
        # 添加噪声到目标潜在表示
        noisy_latents = self.noise_scheduler.add_noise(target_latents, noise, timesteps)
        
        # 编码目标提示词（茶歇领描述）
        with torch.no_grad():
            # 使用批次中的第一个提示词作为示例获取维度
            sample_prompt = batch["target_prompt"][0] if isinstance(batch["target_prompt"], list) else "woman wearing off-shoulder top"
            encoder_hidden_states = self.model.encode_prompt(sample_prompt)
            
            # 为整个批次复制
            if encoder_hidden_states.dim() == 2:
                encoder_hidden_states = encoder_hidden_states.unsqueeze(0)
            encoder_hidden_states = encoder_hidden_states.repeat(batch_size, 1, 1)
        
        # 前向传播预测噪声
        noise_pred = self.model(
            noisy_latents=noisy_latents,
            timesteps=timesteps,
            encoder_hidden_states=encoder_hidden_states
        )
        
        # 计算基础扩散损失
        base_loss = F.mse_loss(noise_pred, noise, reduction='none')
        
        # 调整蒙版维度以匹配latent空间
        if masks.shape[-1] != base_loss.shape[-1]:
            masks_resized = F.interpolate(
                masks, 
                size=(base_loss.shape[-2], base_loss.shape[-1]), 
                mode='bilinear',
                align_corners=False
            )
        else:
            masks_resized = masks
        
        # 确保蒙版有正确的通道数
        if masks_resized.shape[1] != base_loss.shape[1]:
            masks_resized = masks_resized.repeat(1, base_loss.shape[1], 1, 1)
        
        # 计算加权损失（茶歇领区域权重更高）
        mask_weight = self.config.get("masked_loss_weight", 2.0)
        unmasked_weight = self.config.get("unmasked_loss_weight", 0.1)
        
        # 蒙版区域和非蒙版区域的加权损失
        masked_loss = base_loss * masks_resized * mask_weight
        unmasked_loss = base_loss * (1 - masks_resized) * unmasked_weight
        
        total_loss = (masked_loss + unmasked_loss).mean()
        
        return {
            'total_loss': total_loss,
            'masked_loss': masked_loss.mean(),
            'unmasked_loss': unmasked_loss.mean(),
            'base_loss': base_loss.mean()
        }
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """单步训练"""
        self.model.set_train(True)
        
        # 计算损失
        loss_dict = self._compute_loss(batch)
        total_loss = loss_dict['total_loss']
        
        # 反向传播
        self.accelerator.backward(total_loss)
        
        # 梯度裁剪
        max_grad_norm = self.config.get("max_grad_norm", 1.0)
        if max_grad_norm > 0:
            self.accelerator.clip_grad_norm_(self.model.parameters(), max_grad_norm)
        
        # 优化器步骤
        self.optimizer.step()
        if self.lr_scheduler:
            self.lr_scheduler.step()
        self.optimizer.zero_grad()
        
        # 返回损失信息
        losses = {k: v.item() for k, v in loss_dict.items()}
        losses['learning_rate'] = self.lr_scheduler.get_last_lr()[0] if self.lr_scheduler else self.optimizer.param_groups[0]['lr']
        
        return losses
    
    @torch.no_grad()
    def validate_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """验证步骤"""
        self.model.set_train(False)
        
        # 计算验证损失
        loss_dict = self._compute_loss(batch)
        
        return {k: v.item() for k, v in loss_dict.items()}
    
    @torch.no_grad()
    def generate_samples(self, num_samples: int = 3) -> List[Dict]:
        """生成样本用于验证"""
        self.model.set_train(False)
        
        samples = []
        
        # 从验证集获取样本
        for i, batch in enumerate(self.val_loader):
            if i >= num_samples:
                break
            
            # 使用训练好的模型生成茶歇领替换效果
            original_images = batch["original_image"]
            masks = batch["mask"]
            target_prompts = batch["target_prompt"]
            
            # 生成样本
            try:
                # 这里简化处理，实际应用中需要完整的推理pipeline
                generated_images = self._inference_sample(
                    original_images[0:1], 
                    masks[0:1], 
                    target_prompts[0] if isinstance(target_prompts, list) else "woman wearing off-shoulder top"
                )
                
                samples.append({
                    'original': tensor_to_pil(original_images[0]),
                    'target': tensor_to_pil(batch["target_image"][0]),
                    'generated': tensor_to_pil(generated_images[0]),
                    'mask': tensor_to_pil(masks[0]),
                    'prompt': target_prompts[0] if isinstance(target_prompts, list) else "generated sample"
                })
            except Exception as e:
                logger.warning(f"生成样本失败: {e}")
                continue
        
        return samples
    
    def _inference_sample(self, image: torch.Tensor, mask: torch.Tensor, prompt: str) -> torch.Tensor:
        """简化的推理样本生成"""
        # 这是一个简化版本，实际应该使用完整的FLUX pipeline
        # 这里返回target image作为占位符
        return image  # 实际实现中需要完整的推理流程
    
    def _save_checkpoint(self, step: int, is_best: bool = False):
        """保存检查点"""
        checkpoint_data = {
            'step': step,
            'epoch': self.epoch,
            'model_state_dict': self.accelerator.get_state_dict(self.model),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            'config': self.config,
            'best_val_loss': self.best_val_loss
        }
        
        # 保存LoRA权重
        lora_path = os.path.join(self.output_dir, f"checkpoint-{step}")
        self.accelerator.unwrap_model(self.model).save_lora_weights(lora_path)
        
        # 保存完整检查点
        checkpoint_path = self.checkpoint_manager.save_checkpoint(checkpoint_data, step, is_best)
        
        logger.info(f"检查点已保存: {checkpoint_path}")
        
        return checkpoint_path
    
    def _load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        logger.info(f"从检查点恢复: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 恢复模型状态
        self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint['model_state_dict'])
        
        # 恢复优化器状态
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 恢复学习率调度器
        if self.lr_scheduler and checkpoint.get('lr_scheduler_state_dict'):
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        
        # 恢复训练状态
        self.global_step = checkpoint['step']
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        logger.info(f"检查点恢复完成，从步骤 {self.global_step} 继续训练")
    
    def train(self):
        """主训练循环"""
        logger.info("开始茶歇领替换LoRA训练")
        
        max_steps = self.config.get("max_steps", 3000)
        save_steps = self.config.get("save_steps", 500)
        eval_steps = self.config.get("eval_steps", 250)
        logging_steps = self.config.get("logging_steps", 50)
        
        progress_bar = tqdm(
            total=max_steps,
            initial=self.global_step,
            desc="训练茶歇领替换",
            disable=not self.accelerator.is_main_process
        )
        
        while self.global_step < max_steps:
            epoch_loss = 0
            num_batches = 0
            
            for batch in self.train_loader:
                # 训练步骤
                losses = self.train_step(batch)
                epoch_loss += losses['total_loss']
                num_batches += 1
                
                self.global_step += 1
                progress_bar.update(1)
                
                # 日志记录
                if self.global_step % logging_steps == 0 and self.accelerator.is_main_process:
                    for key, value in losses.items():
                        self.writer.add_scalar(f"train/{key}", value, self.global_step)
                        if self.config.get("use_wandb", False):
                            wandb.log({f"train/{key}": value}, step=self.global_step)
                    
                    progress_bar.set_postfix({
                        'loss': f"{losses['total_loss']:.4f}",
                        'lr': f"{losses['learning_rate']:.2e}"
                    })
                
                # 验证
                if self.global_step % eval_steps == 0:
                    val_losses = self._validate()
                    
                    if self.accelerator.is_main_process:
                        for key, value in val_losses.items():
                            self.writer.add_scalar(f"val/{key}", value, self.global_step)
                            if self.config.get("use_wandb", False):
                                wandb.log({f"val/{key}": value}, step=self.global_step)
                        
                        # 生成样本
                        samples = self.generate_samples(3)
                        if samples:
                            self._log_samples(samples, self.global_step)
                        
                        # 检查是否是最佳模型
                        is_best = val_losses['total_loss'] < self.best_val_loss
                        if is_best:
                            self.best_val_loss = val_losses['total_loss']
                
                # 保存检查点
                if self.global_step % save_steps == 0 and self.accelerator.is_main_process:
                    is_best = hasattr(self, '_last_val_loss') and self._last_val_loss == self.best_val_loss
                    self._save_checkpoint(self.global_step, is_best)
                
                if self.global_step >= max_steps:
                    break
            
            self.epoch += 1
        
        # 保存最终模型
        if self.accelerator.is_main_process:
            final_path = self._save_checkpoint(self.global_step, is_best=True)
            logger.info(f"训练完成！最终模型保存在: {final_path}")
        
        progress_bar.close()
    
    def _validate(self) -> Dict[str, float]:
        """验证模型"""
        self.model.set_train(False)
        
        val_losses = []
        for batch in self.val_loader:
            losses = self.validate_step(batch)
            val_losses.append(losses)
        
        # 计算平均损失
        avg_losses = {}
        for key in val_losses[0].keys():
            avg_losses[key] = np.mean([loss[key] for loss in val_losses])
        
        self._last_val_loss = avg_losses['total_loss']
        return avg_losses
    
    def _log_samples(self, samples: List[Dict], step: int):
        """记录生成样本"""
        try:
            # 保存样本对比图
            save_path = os.path.join(self.output_dir, "samples", f"step_{step}.png")
            save_comparison_grid(samples, save_path)
            
            # 记录到tensorboard
            if hasattr(self, 'writer'):
                import torchvision.transforms as T
                for i, sample in enumerate(samples[:3]):
                    # 转换PIL图像为tensor
                    to_tensor = T.ToTensor()
                    
                    self.writer.add_image(f"samples/original_{i}", to_tensor(sample['original']), step)
                    self.writer.add_image(f"samples/target_{i}", to_tensor(sample['target']), step)
                    self.writer.add_image(f"samples/generated_{i}", to_tensor(sample['generated']), step)
            
            logger.info(f"样本已保存: {save_path}")
            
        except Exception as e:
            logger.warning(f"保存样本失败: {e}")