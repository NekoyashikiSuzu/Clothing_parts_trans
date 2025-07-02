# data_processing/paired_dataset.py - 成对数据集类
import os
import numpy as np
from typing import Dict, List, Optional, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import CLIPTokenizer

class PairedCollarDataset(Dataset):
    """茶歇领替换成对数据集
    
    数据结构:
    data/
    ├── train/
    │   ├── 001_original.jpg     # 变换前图片
    │   ├── 001_target.jpg       # 变换后图片
    │   ├── 001_original.txt     # 原始描述
    │   ├── 001_target.txt       # 目标描述
    │   └── ...
    ├── val/
    │   └── (same structure)
    └── masks/
        ├── 001_mask.png        # 变换区域蒙版
        └── ...
    """
    
    def __init__(
        self,
        data_dir: str,
        mask_dir: str,
        tokenizer: CLIPTokenizer,
        resolution: int = 1024,
        max_caption_length: int = 77,
        is_training: bool = True,
        augmentation_prob: float = 0.5
    ):
        self.data_dir = data_dir
        self.mask_dir = mask_dir
        self.tokenizer = tokenizer
        self.resolution = resolution
        self.max_caption_length = max_caption_length
        self.is_training = is_training
        self.augmentation_prob = augmentation_prob
        
        # 加载数据对
        self.data_pairs = self._load_data_pairs()
        
        # 创建数据变换
        self.transforms = self._create_transforms()
        
        print(f"加载了 {len(self.data_pairs)} 个数据对")
    
    def _create_transforms(self):
        """创建数据变换管道"""
        if self.is_training:
            # 训练时的数据增强
            transform_list = [
                A.Resize(self.resolution, self.resolution, interpolation=Image.LANCZOS),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(
                    brightness=0.1, 
                    contrast=0.1, 
                    saturation=0.1, 
                    hue=0.05, 
                    p=self.augmentation_prob
                ),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ToTensorV2()
            ]
        else:
            # 验证时只做基础变换
            transform_list = [
                A.Resize(self.resolution, self.resolution, interpolation=Image.LANCZOS),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ToTensorV2()
            ]
        
        # 图像变换（支持多个目标图像）
        image_transform = A.Compose(
            transform_list,
            additional_targets={'target': 'image'}
        )
        
        # 蒙版变换（只做几何变换）
        mask_transform = A.Compose([
            A.Resize(self.resolution, self.resolution, interpolation=Image.NEAREST),
            A.HorizontalFlip(p=0.5) if self.is_training else A.NoOp(),
            ToTensorV2()
        ])
        
        return image_transform, mask_transform
    
    def _load_data_pairs(self):
        """加载训练数据对"""
        pairs = []
        
        if not os.path.exists(self.data_dir):
            raise ValueError(f"数据目录不存在: {self.data_dir}")
        
        # 扫描所有original文件
        for filename in os.listdir(self.data_dir):
            if '_original.' in filename and filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                # 提取base_id
                for ext in ['.jpg', '.jpeg', '.png']:
                    if filename.endswith(f'_original{ext}'):
                        base_id = filename[:-len(f'_original{ext}')]
                        break
                else:
                    continue
                
                # 构建文件路径
                original_img = os.path.join(self.data_dir, filename)
                
                # 查找对应的target图片
                target_img = None
                for ext in ['.jpg', '.jpeg', '.png']:
                    potential_target = os.path.join(self.data_dir, f"{base_id}_target{ext}")
                    if os.path.exists(potential_target):
                        target_img = potential_target
                        break
                
                if not target_img:
                    print(f"跳过: 未找到target图片 for {base_id}")
                    continue
                
                # 文本文件路径
                original_txt = os.path.join(self.data_dir, f"{base_id}_original.txt")
                target_txt = os.path.join(self.data_dir, f"{base_id}_target.txt")
                
                # 蒙版文件路径
                mask_img = os.path.join(self.mask_dir, f"{base_id}_mask.png")
                if not os.path.exists(mask_img):
                    # 尝试其他蒙版命名格式
                    for mask_name in [f"{base_id}.png", f"mask_{base_id}.png"]:
                        potential_mask = os.path.join(self.mask_dir, mask_name)
                        if os.path.exists(potential_mask):
                            mask_img = potential_mask
                            break
                    else:
                        print(f"跳过: 未找到蒙版文件 for {base_id}")
                        continue
                
                # 检查文本文件
                if not os.path.exists(original_txt) or not os.path.exists(target_txt):
                    print(f"跳过: 缺少文本文件 for {base_id}")
                    continue
                
                # 读取提示词
                try:
                    with open(original_txt, 'r', encoding='utf-8') as f:
                        original_prompt = f.read().strip()
                    with open(target_txt, 'r', encoding='utf-8') as f:
                        target_prompt = f.read().strip()
                    
                    if not original_prompt or not target_prompt:
                        print(f"跳过: 空的提示词文件 for {base_id}")
                        continue
                        
                except Exception as e:
                    print(f"跳过: 读取文本文件失败 {base_id}: {e}")
                    continue
                
                pairs.append({
                    'id': base_id,
                    'original_image': original_img,
                    'target_image': target_img,
                    'mask_image': mask_img,
                    'original_prompt': original_prompt,
                    'target_prompt': target_prompt
                })
        
        return pairs
    
    def _preprocess_mask(self, mask_array: np.ndarray) -> np.ndarray:
        """预处理蒙版，确保茶歇领替换的精确性"""
        # 转换为浮点数
        mask_float = mask_array.astype(np.float32) / 255.0
        
        # 应用轻微的高斯模糊来软化边缘
        from scipy import ndimage
        mask_blurred = ndimage.gaussian_filter(mask_float, sigma=1.0)
        
        # 确保值在[0,1]范围内
        mask_processed = np.clip(mask_blurred, 0, 1)
        
        return mask_processed
    
    def __len__(self):
        return len(self.data_pairs)
    
    def __getitem__(self, idx):
        pair = self.data_pairs[idx]
        
        try:
            # 加载图像
            original_img = Image.open(pair['original_image']).convert('RGB')
            target_img = Image.open(pair['target_image']).convert('RGB')
            mask_img = Image.open(pair['mask_image']).convert('L')
            
            # 转换为numpy数组
            original_array = np.array(original_img)
            target_array = np.array(target_img)
            mask_array = np.array(mask_img)
            
            # 预处理蒙版
            mask_array = self._preprocess_mask(mask_array)
            
            # 确保所有图像尺寸一致
            if original_array.shape[:2] != target_array.shape[:2]:
                raise ValueError(f"原始图和目标图尺寸不匹配: {pair['id']}")
            if original_array.shape[:2] != mask_array.shape[:2]:
                raise ValueError(f"图像和蒙版尺寸不匹配: {pair['id']}")
            
            # 应用变换 - 确保几何变换同步
            image_transform, mask_transform = self.transforms
            
            # 设置相同的随机种子确保几何变换一致
            if self.is_training:
                # 生成随机种子
                random_seed = np.random.randint(0, 2**32)
                
                # 图像变换（original和target同步）
                A.random.seed(random_seed)
                image_transformed = image_transform(
                    image=original_array, 
                    target=target_array
                )
                original_transformed = image_transformed['image']
                target_transformed = image_transformed['target']
                
                # 蒙版变换（使用相同随机种子）
                A.random.seed(random_seed)
                mask_transformed = mask_transform(image=mask_array)['image']
            else:
                # 验证时不使用随机变换
                image_transformed = image_transform(
                    image=original_array, 
                    target=target_array
                )
                original_transformed = image_transformed['image']
                target_transformed = image_transformed['target']
                
                mask_transformed = mask_transform(image=mask_array)['image']
            
            # 处理蒙版张量 - 确保单通道
            if mask_transformed.dim() == 3 and mask_transformed.shape[0] == 3:
                # 如果是3通道，取第一个通道
                mask_transformed = mask_transformed[0:1]
            elif mask_transformed.dim() == 3 and mask_transformed.shape[0] == 1:
                # 已经是单通道，保持不变
                pass
            else:
                # 添加通道维度
                mask_transformed = mask_transformed.unsqueeze(0)
            
            # 归一化蒙版到[0,1]
            mask_tensor = mask_transformed.float()
            if mask_tensor.max() > 1.0:
                mask_tensor = mask_tensor / 255.0
            
            # 编码目标提示词（用于生成茶歇领风格）
            text_inputs = self.tokenizer(
                pair['target_prompt'],
                truncation=True,
                padding="max_length",
                max_length=self.max_caption_length,
                return_tensors="pt"
            )
            
            # 也编码原始提示词（可能用于对比学习）
            original_text_inputs = self.tokenizer(
                pair['original_prompt'],
                truncation=True,
                padding="max_length",
                max_length=self.max_caption_length,
                return_tensors="pt"
            )
            
            return {
                'original_image': original_transformed,
                'target_image': target_transformed,
                'mask': mask_tensor,
                'input_ids': text_inputs.input_ids.squeeze(),
                'attention_mask': text_inputs.attention_mask.squeeze(),
                'original_input_ids': original_text_inputs.input_ids.squeeze(),
                'original_attention_mask': original_text_inputs.attention_mask.squeeze(),
                'original_prompt': pair['original_prompt'],
                'target_prompt': pair['target_prompt'],
                'pair_id': pair['id']
            }
            
        except Exception as e:
            print(f"加载数据失败 {pair['id']}: {e}")
            # 返回下一个数据项
            return self.__getitem__((idx + 1) % len(self.data_pairs))
    
    def get_sample_data(self, num_samples: int = 3) -> Dict:
        """获取样本数据用于验证"""
        samples = []
        indices = np.random.choice(len(self), min(num_samples, len(self)), replace=False)
        
        for idx in indices:
            sample = self.__getitem__(idx)
            samples.append({
                'pair_id': sample['pair_id'],
                'original_prompt': sample['original_prompt'],
                'target_prompt': sample['target_prompt'],
                'has_mask': torch.sum(sample['mask']) > 0
            })
        
        return samples