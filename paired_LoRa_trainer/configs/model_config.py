# configs/model_config.py 
from dataclasses import dataclass
from typing import Optional, List
import sys

@dataclass
class ModelConfig:
    # FLUX模型配置
    model_name: str = "black-forest-labs/FLUX.1-fill-dev"
    model_revision: Optional[str] = None
    cache_dir: Optional[str] = "./cache"
    
    # LoRA配置
    lora_rank: int = 32
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = None
    
    # 量化配置
    use_8bit: bool = True
    use_gradient_checkpointing: bool = True
    
    # FLUX特定配置
    resolution: int = 1024
    vae_scale_factor: int = 8
    scheduler_name: str = "FlowMatchEulerDiscreteScheduler"
    
    # Python 3.12 特定优化
    torch_dtype: str = "bfloat16"  # Python 3.12对bfloat16支持更好
    compile_model: bool = False    # 是否使用torch.compile
    use_flash_attention: bool = False  # 在Python 3.12上可能不兼容
    
    def __post_init__(self):
        if self.lora_target_modules is None:
            # FLUX的注意力层目标模块
            self.lora_target_modules = [
                "to_k", "to_q", "to_v", "to_out.0",
                "ff.net.0.proj", "ff.net.2"
            ]
        
        # Python 3.12 特定调整
        if sys.version_info >= (3, 12):
            # 在Python 3.12上启用编译优化
            self.compile_model = True
            # 禁用可能不兼容的功能
            self.use_flash_attention = False
            print("应用Python 3.12优化设置")