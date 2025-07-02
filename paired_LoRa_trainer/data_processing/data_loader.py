# data_processing/data_loader.py - 数据加载器
import torch
from torch.utils.data import DataLoader
from typing import Optional

def create_paired_data_loaders(
    train_dataset,
    val_dataset: Optional = None,
    batch_size: int = 2,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True
):
    """创建成对数据的DataLoader"""
    
    # 训练数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=persistent_workers and num_workers > 0,
        collate_fn=custom_collate_fn
    )
    
    # 验证数据加载器
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            persistent_workers=persistent_workers and num_workers > 0,
            collate_fn=custom_collate_fn
        )
    
    return train_loader, val_loader

def custom_collate_fn(batch):
    """自定义批次整理函数"""
    # 处理可能的数据类型不一致问题
    collated = {}
    
    # 图像相关的张量数据
    tensor_keys = ['original_image', 'target_image', 'mask', 'input_ids', 'attention_mask', 
                   'original_input_ids', 'original_attention_mask']
    
    for key in tensor_keys:
        if key in batch[0]:
            collated[key] = torch.stack([item[key] for item in batch])
    
    # 文本数据（保持为列表）
    text_keys = ['original_prompt', 'target_prompt', 'pair_id']
    
    for key in text_keys:
        if key in batch[0]:
            collated[key] = [item[key] for item in batch]
    
    return collated