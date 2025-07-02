# configs/training_config.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class TrainingConfig:
    # 基础训练参数
    learning_rate: float = 5e-5
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    max_steps: int = 3000
    warmup_steps: int = 300
    
    # 优化器配置
    optimizer: str = "adamw_8bit"
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    
    # 学习率调度
    lr_scheduler: str = "cosine_with_restarts"
    lr_warmup_steps: int = 300
    lr_num_cycles: int = 1
    
    # 保存和验证
    save_steps: int = 500
    logging_steps: int = 50
    eval_steps: int = 250
    save_total_limit: int = 5
    
    # 混合精度
    mixed_precision: str = "bf16"  # fp16, bf16, no
    
    # 遮罩训练权重
    masked_loss_weight: float = 2.0
    unmasked_loss_weight: float = 0.1
    
    # 噪声调度
    noise_offset: float = 0.0
    snr_gamma: Optional[float] = 5.0
    
    # 数据增强
    enable_augmentation: bool = True
    
    # 随机种子
    seed: int = 42