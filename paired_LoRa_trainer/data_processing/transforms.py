# data_processing/transforms.py - 数据变换
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

class ClothingTransforms:
    """服装图像变换类"""
    
    def __init__(self, resolution: int = 1024, is_training: bool = True):
        self.resolution = resolution
        self.is_training = is_training
        
        # 创建图像变换pipeline
        self.image_transforms = self._create_image_transforms()
        self.mask_transforms = self._create_mask_transforms()
    
    def _create_image_transforms(self):
        """创建图像变换"""
        if self.is_training:
            # 训练时的变换
            transforms = [
                A.Resize(self.resolution, self.resolution, interpolation=Image.LANCZOS),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(
                    brightness=0.1, 
                    contrast=0.1, 
                    saturation=0.05,  # 保守的饱和度调整
                    hue=0.02,         # 很小的色相调整
                    p=0.3
                ),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ToTensorV2()
            ]
        else:
            # 验证时的变换
            transforms = [
                A.Resize(self.resolution, self.resolution, interpolation=Image.LANCZOS),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ToTensorV2()
            ]
        
        return A.Compose(transforms, additional_targets={'target': 'image'})
    
    def _create_mask_transforms(self):
        """创建蒙版变换"""
        transforms = [
            A.Resize(self.resolution, self.resolution, interpolation=Image.NEAREST),
        ]
        
        if self.is_training:
            transforms.append(A.HorizontalFlip(p=0.5))
        
        transforms.append(ToTensorV2())
        
        return A.Compose(transforms)
    
    def __call__(self, original_image, target_image, mask_image):
        """应用变换"""
        # 应用图像变换（同步几何变换）
        image_result = self.image_transforms(
            image=original_image, 
            target=target_image
        )
        
        # 应用蒙版变换
        mask_result = self.mask_transforms(image=mask_image)
        
        return {
            'original': image_result['image'],
            'target': image_result['target'],
            'mask': mask_result['image']
        }