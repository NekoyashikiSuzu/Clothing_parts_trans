# utils/model_utils.py - 模型工具函数
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import logging
import os

logger = logging.getLogger(__name__)

def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """统计模型参数数量"""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())

def get_model_size(model: nn.Module) -> Dict[str, Any]:
    """获取模型大小信息"""
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    
    # 估算模型大小（以MB为单位）
    param_size = total_params * 4 / 1024 / 1024  # 假设float32
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'trainable_ratio': trainable_params / total_params if total_params > 0 else 0,
        'estimated_size_mb': param_size
    }

def print_model_info(model: nn.Module, model_name: str = "Model"):
    """打印模型信息"""
    info = get_model_size(model)
    
    logger.info(f"{model_name} 信息:")
    logger.info(f"  总参数数: {info['total_parameters']:,}")
    logger.info(f"  可训练参数: {info['trainable_parameters']:,}")
    logger.info(f"  可训练比例: {info['trainable_ratio']:.2%}")
    logger.info(f"  估算大小: {info['estimated_size_mb']:.1f} MB")

def load_model_safely(model_path: str, device: str = "cpu") -> Optional[Dict[str, Any]]:
    """安全加载模型权重"""
    try:
        if not os.path.exists(model_path):
            logger.error(f"模型文件不存在: {model_path}")
            return None
        
        checkpoint = torch.load(model_path, map_location=device)
        logger.info(f"成功加载模型: {model_path}")
        
        return checkpoint
        
    except Exception as e:
        logger.error(f"加载模型失败 {model_path}: {e}")
        return None

def save_model_safely(model: nn.Module, save_path: str, additional_info: Optional[Dict] = None):
    """安全保存模型"""
    try:
        # 确保保存目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 准备保存的数据
        save_data = {
            'model_state_dict': model.state_dict(),
            'model_info': get_model_size(model)
        }
        
        if additional_info:
            save_data.update(additional_info)
        
        torch.save(save_data, save_path)
        logger.info(f"模型已保存: {save_path}")
        
    except Exception as e:
        logger.error(f"保存模型失败 {save_path}: {e}")

def freeze_model_layers(model: nn.Module, freeze_patterns: list):
    """冻结模型指定层"""
    frozen_count = 0
    
    for name, param in model.named_parameters():
        for pattern in freeze_patterns:
            if pattern in name:
                param.requires_grad = False
                frozen_count += 1
                break
    
    logger.info(f"冻结了 {frozen_count} 个参数")

def unfreeze_model_layers(model: nn.Module, unfreeze_patterns: list):
    """解冻模型指定层"""
    unfrozen_count = 0
    
    for name, param in model.named_parameters():
        for pattern in unfreeze_patterns:
            if pattern in name:
                param.requires_grad = True
                unfrozen_count += 1
                break
    
    logger.info(f"解冻了 {unfrozen_count} 个参数")

def get_learning_rates(optimizer: torch.optim.Optimizer) -> Dict[str, float]:
    """获取优化器中各参数组的学习率"""
    lrs = {}
    
    for i, param_group in enumerate(optimizer.param_groups):
        lrs[f'group_{i}'] = param_group['lr']
    
    return lrs

def clip_gradients(model: nn.Module, max_norm: float = 1.0) -> float:
    """梯度裁剪"""
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    return total_norm.item()

def check_model_device(model: nn.Module) -> str:
    """检查模型所在设备"""
    devices = {param.device for param in model.parameters()}
    
    if len(devices) == 1:
        return str(list(devices)[0])
    else:
        logger.warning(f"模型参数分布在多个设备上: {devices}")
        return "mixed"

def convert_model_dtype(model: nn.Module, dtype: torch.dtype):
    """转换模型数据类型"""
    model = model.to(dtype)
    logger.info(f"模型数据类型已转换为: {dtype}")
    return model

def enable_gradient_checkpointing(model: nn.Module):
    """启用梯度检查点以节省显存"""
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        logger.info("已启用梯度检查点")
    else:
        logger.warning("模型不支持梯度检查点")