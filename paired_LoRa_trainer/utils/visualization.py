# utils/visualization.py - 可视化工具
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Optional
import os

def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """将张量转换为PIL图像"""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    if tensor.dim() == 3:
        # 处理CHW格式
        if tensor.shape[0] in [1, 3]:
            tensor = tensor.permute(1, 2, 0)
    
    # 归一化到[0,255]
    if tensor.dtype == torch.float32 or tensor.dtype == torch.float16 or tensor.dtype == torch.bfloat16:
        if tensor.min() >= -1 and tensor.max() <= 1:
            # 假设是[-1,1]范围
            tensor = (tensor + 1) / 2
        elif tensor.max() <= 1:
            # 假设是[0,1]范围
            pass
        tensor = (tensor * 255).clamp(0, 255)
    
    # 转换为numpy
    array = tensor.cpu().numpy().astype(np.uint8)
    
    # 处理单通道图像
    if array.ndim == 3 and array.shape[2] == 1:
        array = array.squeeze(2)
    
    if array.ndim == 2:
        return Image.fromarray(array, mode='L')
    else:
        return Image.fromarray(array, mode='RGB')

def pil_to_tensor(image: Image.Image, normalize: bool = True) -> torch.Tensor:
    """将PIL图像转换为张量"""
    # 转换为RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # 转换为numpy数组
    array = np.array(image)
    
    # 转换为张量 (H, W, C) -> (C, H, W)
    tensor = torch.from_numpy(array).permute(2, 0, 1).float()
    
    if normalize:
        tensor = tensor / 255.0
        # 归一化到[-1, 1]
        tensor = tensor * 2.0 - 1.0
    
    return tensor

def save_comparison_grid(
    samples: List[Dict], 
    save_path: str, 
    grid_cols: int = 4,
    image_size: tuple = (256, 256),
    add_labels: bool = True
):
    """保存对比网格图像"""
    if not samples:
        return
    
    # 计算网格大小
    num_samples = len(samples)
    grid_rows = (num_samples + grid_cols - 1) // grid_cols
    
    # 创建网格图像
    grid_width = grid_cols * image_size[0]
    grid_height = grid_rows * image_size[1]
    
    if add_labels:
        grid_height += 30  # 为标签留出空间
    
    grid_image = Image.new('RGB', (grid_width, grid_height), 'white')
    
    # 尝试加载字体
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except:
        font = ImageFont.load_default()
    
    draw = ImageDraw.Draw(grid_image)
    
    for i, sample in enumerate(samples):
        row = i // grid_cols
        col = i % grid_cols
        
        x = col * image_size[0]
        y = row * image_size[1]
        
        if add_labels:
            y += 30
        
        # 处理样本中的图像
        images_in_sample = []
        
        # 收集所有图像
        if 'original' in sample:
            images_in_sample.append(('Original', sample['original']))
        if 'target' in sample:
            images_in_sample.append(('Target', sample['target']))
        if 'generated' in sample:
            images_in_sample.append(('Generated', sample['generated']))
        if 'mask' in sample:
            images_in_sample.append(('Mask', sample['mask']))
        
        # 创建子网格
        sub_cols = len(images_in_sample)
        sub_width = image_size[0] // sub_cols
        sub_height = image_size[1]
        
        for j, (label, img) in enumerate(images_in_sample):
            sub_x = x + j * sub_width
            sub_y = y
            
            # 调整图像大小
            if isinstance(img, torch.Tensor):
                img = tensor_to_pil(img)
            
            img_resized = img.resize((sub_width, sub_height), Image.LANCZOS)
            grid_image.paste(img_resized, (sub_x, sub_y))
            
            # 添加标签
            if add_labels:
                text_x = sub_x + sub_width // 2
                text_y = y - 25
                
                # 计算文本大小以居中
                bbox = draw.textbbox((0, 0), label, font=font)
                text_width = bbox[2] - bbox[0]
                
                draw.text(
                    (text_x - text_width // 2, text_y), 
                    label, 
                    fill='black', 
                    font=font
                )
        
        # 添加样本提示词（如果有）
        if add_labels and 'prompt' in sample:
            prompt_text = sample['prompt'][:50] + "..." if len(sample['prompt']) > 50 else sample['prompt']
            
            text_x = x + image_size[0] // 2
            text_y = y - 10
            
            bbox = draw.textbbox((0, 0), prompt_text, font=font)
            text_width = bbox[2] - bbox[0]
            
            draw.text(
                (text_x - text_width // 2, text_y), 
                prompt_text, 
                fill='blue', 
                font=font
            )
    
    # 保存网格图像
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    grid_image.save(save_path)

def create_training_visualization(
    step: int,
    losses: Dict[str, float],
    samples: List[Dict],
    save_dir: str
):
    """创建训练可视化"""
    # 保存样本网格
    if samples:
        sample_path = os.path.join(save_dir, f"samples_step_{step}.png")
        save_comparison_grid(samples, sample_path)
    
    # 可以扩展添加损失曲线图等

def visualize_mask_overlay(
    image: Image.Image, 
    mask: Image.Image, 
    alpha: float = 0.5,
    mask_color: tuple = (255, 0, 0)
) -> Image.Image:
    """在图像上叠加蒙版可视化"""
    # 确保图像格式一致
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    if mask.mode != 'L':
        mask = mask.convert('L')
    
    # 调整蒙版大小
    if mask.size != image.size:
        mask = mask.resize(image.size, Image.NEAREST)
    
    # 创建彩色蒙版
    colored_mask = Image.new('RGB', image.size, mask_color)
    
    # 创建Alpha通道
    mask_array = np.array(mask)
    alpha_channel = (mask_array * alpha).astype(np.uint8)
    alpha_mask = Image.fromarray(alpha_channel, 'L')
    
    # 合成图像
    result = Image.composite(colored_mask, image, alpha_mask)
    
    return result

def create_before_after_comparison(
    before_image: Image.Image,
    after_image: Image.Image,
    mask: Optional[Image.Image] = None,
    save_path: Optional[str] = None
) -> Image.Image:
    """创建前后对比图"""
    # 确保图像大小一致
    if before_image.size != after_image.size:
        after_image = after_image.resize(before_image.size, Image.LANCZOS)
    
    width, height = before_image.size
    
    # 创建对比图像
    if mask:
        comparison = Image.new('RGB', (width * 3, height), 'white')
        comparison.paste(before_image, (0, 0))
        comparison.paste(mask.convert('RGB'), (width, 0))
        comparison.paste(after_image, (width * 2, 0))
        
        # 添加标签
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        draw = ImageDraw.Draw(comparison)
        draw.text((width//2 - 30, 10), "Before", fill='black', font=font)
        draw.text((width + width//2 - 20, 10), "Mask", fill='black', font=font)
        draw.text((width*2 + width//2 - 25, 10), "After", fill='black', font=font)
    else:
        comparison = Image.new('RGB', (width * 2, height), 'white')
        comparison.paste(before_image, (0, 0))
        comparison.paste(after_image, (width, 0))
        
        # 添加标签
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        draw = ImageDraw.Draw(comparison)
        draw.text((width//2 - 30, 10), "Before", fill='black', font=font)
        draw.text((width + width//2 - 25, 10), "After", fill='black', font=font)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        comparison.save(save_path)
    
    return comparison