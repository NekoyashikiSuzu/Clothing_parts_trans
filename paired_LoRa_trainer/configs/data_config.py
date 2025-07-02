# configs/data_config.py
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class DataConfig:
    # 数据路径
    train_data_dir: str = "./data/train"
    validation_data_dir: str = "./data/val"
    mask_data_dir: str = "./data/masks"
    
    # 图像配置
    resolution: int = 1024
    center_crop: bool = False
    random_flip: bool = True
    
    # 数据加载
    num_workers: int = 4
    pin_memory: bool = True
    
    # 提示词配置
    caption_column: str = "caption"
    max_caption_length: int = 77
    
    # 服装部件类别
    clothing_parts: List[str] = None
    
    # 遮罩配置
    mask_threshold: float = 0.5
    mask_blur_radius: int = 2
    
    # 数据增强概率
    augmentation_prob: float = 0.5
    color_jitter_prob: float = 0.3
    
    def __post_init__(self):
        if self.clothing_parts is None:
            self.clothing_parts = [
                "collar", "sleeve", "neckline", "button", 
                "pocket", "hem", "cuff", "lapel"
            ]