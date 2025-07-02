# training/training_utils.py - 训练工具函数
import os
import torch
import random
import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

def setup_training_environment(config: Dict[str, Any]):
    """设置训练环境"""
    # 设置随机种子
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    # 设置CUDA优化
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # 显存优化
        torch.cuda.empty_cache()
        
        # 显示GPU信息
        gpu_count = torch.cuda.device_count()
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            logger.info(f"训练环境设置完成，随机种子: {seed}")

def compute_training_stats(losses: Dict[str, float], step: int) -> Dict[str, Any]:
    """计算训练统计信息"""
    stats = {
        'step': step,
        'total_loss': losses.get('total_loss', 0.0),
        'masked_loss': losses.get('masked_loss', 0.0),
        'unmasked_loss': losses.get('unmasked_loss', 0.0),
        'learning_rate': losses.get('learning_rate', 0.0)
    }
    
    # 计算损失比例
    if stats['total_loss'] > 0:
        stats['masked_loss_ratio'] = stats['masked_loss'] / stats['total_loss']
        stats['unmasked_loss_ratio'] = stats['unmasked_loss'] / stats['total_loss']
    else:
        stats['masked_loss_ratio'] = 0.0
        stats['unmasked_loss_ratio'] = 0.0
    
    return stats

def estimate_training_time(current_step: int, max_steps: int, step_time: float) -> str:
    """估算剩余训练时间"""
    remaining_steps = max_steps - current_step
    remaining_time = remaining_steps * step_time
    
    hours = int(remaining_time // 3600)
    minutes = int((remaining_time % 3600) // 60)
    
    if hours > 0:
        return f"{hours}小时{minutes}分钟"
    else:
        return f"{minutes}分钟"

def check_memory_usage():
    """检查显存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        
        return {
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'utilization': allocated / torch.cuda.get_device_properties(0).total_memory * 1024**3
        }
    
    return None

def save_training_config(config: Dict[str, Any], save_path: str):
    """保存训练配置"""
    import yaml
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    logger.info(f"训练配置已保存: {save_path}")

def load_training_config(config_path: str) -> Dict[str, Any]:
    """加载训练配置"""
    import yaml
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"训练配置已加载: {config_path}")
    return config(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    
    logger.info