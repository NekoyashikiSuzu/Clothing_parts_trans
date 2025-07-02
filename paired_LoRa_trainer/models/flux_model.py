# models/flux_model.py - FLUX模型封装
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union
from diffusers import FluxFillPipeline, FluxTransformer2DModel
from diffusers.utils import logging
from peft import LoraConfig, get_peft_model, TaskType
import sys

logger = logging.get_logger(__name__)

class FluxLoRAModel(nn.Module):
    """FLUX Fill模型的LoRA封装"""
    
    def __init__(
        self,
        model_name: str = "black-forest-labs/FLUX.1-fill-dev",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        lora_rank: int = 32,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        lora_target_modules: Optional[List[str]] = None,
        use_gradient_checkpointing: bool = True,
        cache_dir: Optional[str] = "./cache"
    ):
        super().__init__()
        
        self.model_name = model_name
        self.device = device
        self.torch_dtype = torch_dtype
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        
        # 默认LoRA目标模块（FLUX Transformer的注意力层）
        if lora_target_modules is None:
            self.lora_target_modules = [
                "to_k", "to_q", "to_v", "to_out.0",
                "ff.net.0.proj", "ff.net.2"
            ]
        else:
            self.lora_target_modules = lora_target_modules
        
        logger.info(f"初始化FLUX模型: {model_name}")
        
        # 加载FLUX Fill pipeline
        self.pipeline = FluxFillPipeline.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir,
            device_map=None  # 手动管理设备
        )
        
        # 提取transformer模型
        self.transformer = self.pipeline.transformer
        
        # 移动到设备
        self.pipeline = self.pipeline.to(device)
        
        # 启用梯度检查点
        if use_gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()
        
        # 设置LoRA
        self._setup_lora()
        
        # Python 3.12优化
        if sys.version_info >= (3, 12):
            try:
                # 尝试使用torch.compile优化
                self.transformer = torch.compile(self.transformer, mode="reduce-overhead")
                logger.info("启用torch.compile优化")
            except Exception as e:
                logger.warning(f"torch.compile优化失败: {e}")
    
    def _setup_lora(self):
        """设置LoRA配置"""
        logger.info(f"配置LoRA: rank={self.lora_rank}, alpha={self.lora_alpha}")
        
        # 创建LoRA配置
        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            target_modules=self.lora_target_modules,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type=TaskType.DIFFUSION  # 扩散模型任务
        )
        
        # 应用LoRA到transformer
        self.transformer = get_peft_model(self.transformer, lora_config)
        
        # 只训练LoRA参数
        self.transformer.train()
        for param in self.transformer.parameters():
            param.requires_grad = False
        
        # 启用LoRA参数训练
        for name, param in self.transformer.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
        
        # 统计可训练参数
        trainable_params = sum(p.numel() for p in self.transformer.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.transformer.parameters())
        
        logger.info(f"LoRA可训练参数: {trainable_params:,} / {total_params:,} "
                   f"({100 * trainable_params / total_params:.2f}%)")
    
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """将图像编码到潜在空间"""
        with torch.no_grad():
            # FLUX使用VAE编码
            latents = self.pipeline.vae.encode(images).latent_dist.sample()
            # FLUX的缩放因子
            latents = latents * self.pipeline.vae.config.scaling_factor
            return latents
    
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """将潜在表示解码为图像"""
        with torch.no_grad():
            # 反缩放
            latents = latents / self.pipeline.vae.config.scaling_factor
            # VAE解码
            images = self.pipeline.vae.decode(latents).sample
            return images
    
    def encode_prompt(self, prompt: str) -> torch.Tensor:
        """编码提示词"""
        with torch.no_grad():
            # FLUX使用T5文本编码器
            text_inputs = self.pipeline.tokenizer(
                prompt,
                return_tensors="pt",
                padding="max_length",
                max_length=self.pipeline.tokenizer.model_max_length,
                truncation=True
            )
            
            text_input_ids = text_inputs.input_ids.to(self.device)
            
            # 获取文本嵌入
            prompt_embeds = self.pipeline.text_encoder(text_input_ids)[0]
            
            return prompt_embeds
    
    def forward(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        added_cond_kwargs: Optional[Dict] = None
    ) -> torch.Tensor:
        """前向传播"""
        
        # FLUX transformer前向传播
        noise_pred = self.transformer(
            hidden_states=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False
        )[0]
        
        return noise_pred
    
    def save_lora_weights(self, save_path: str):
        """保存LoRA权重"""
        logger.info(f"保存LoRA权重到: {save_path}")
        self.transformer.save_pretrained(save_path)
    
    def load_lora_weights(self, load_path: str):
        """加载LoRA权重"""
        logger.info(f"加载LoRA权重从: {load_path}")
        
        # 重新创建LoRA模型并加载权重
        from peft import PeftModel
        self.transformer = PeftModel.from_pretrained(
            self.transformer.base_model, 
            load_path
        )
        
        # 合并LoRA权重到pipeline中
        self.pipeline.transformer = self.transformer
    
    def merge_and_unload(self):
        """合并LoRA权重并卸载"""
        logger.info("合并LoRA权重到基础模型")
        merged_model = self.transformer.merge_and_unload()
        self.pipeline.transformer = merged_model
        return merged_model
    
    def get_scheduler(self):
        """获取噪声调度器"""
        return self.pipeline.scheduler
    
    def prepare_mask_and_masked_image(self, image: torch.Tensor, mask: torch.Tensor):
        """准备蒙版和被蒙版的图像（用于inpainting）"""
        # 确保mask是二值的
        mask = (mask > 0.5).float()
        
        # 创建被蒙版的图像
        masked_image = image * (1 - mask)
        
        return mask, masked_image
    
    def set_train(self, mode: bool = True):
        """设置训练模式"""
        if mode:
            self.transformer.train()
            # 确保只有LoRA参数可训练
            for name, param in self.transformer.named_parameters():
                if "lora_" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        else:
            self.transformer.eval()
    
    def parameters(self):
        """返回可训练参数"""
        return (p for p in self.transformer.parameters() if p.requires_grad)
    
    def named_parameters(self):
        """返回命名的可训练参数"""
        return ((n, p) for n, p in self.transformer.named_parameters() if p.requires_grad)