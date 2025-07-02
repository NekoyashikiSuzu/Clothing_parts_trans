# models/loss_functions.py - 损失函数
import torch
import torch.nn.functional as F

class FluxDiffusionLoss:
    """FLUX扩散模型损失函数"""
    
    def __init__(self, masked_weight: float = 2.0, unmasked_weight: float = 0.1):
        self.masked_weight = masked_weight
        self.unmasked_weight = unmasked_weight
    
    def __call__(self, noise_pred, noise_target, timesteps, masks=None):
        """计算损失"""
        # 基础MSE损失
        loss = F.mse_loss(noise_pred, noise_target, reduction='none')
        
        if masks is not None:
            # 调整蒙版维度以匹配loss
            if masks.shape != loss.shape:
                masks = F.interpolate(
                    masks, 
                    size=loss.shape[-2:], 
                    mode='bilinear',
                    align_corners=False
                )
                
                # 确保通道数匹配
                if masks.shape[1] != loss.shape[1]:
                    masks = masks.repeat(1, loss.shape[1], 1, 1)
            
            # 加权损失
            masked_loss = loss * masks * self.masked_weight
            unmasked_loss = loss * (1 - masks) * self.unmasked_weight
            
            total_loss = masked_loss + unmasked_loss
        else:
            total_loss = loss
        
        return total_loss.mean()

class FluxFlowMatchingLoss:
    """FLUX Flow Matching损失函数"""
    
    def __init__(self, mask_weight: float = 2.0):
        self.mask_weight = mask_weight
    
    def __call__(self, pred_flow, target_flow, timesteps, masks=None):
        """Flow Matching损失计算"""
        # Flow Matching使用MSE损失
        loss = F.mse_loss(pred_flow, target_flow, reduction='none')
        
        if masks is not None:
            # 对蒙版区域加权
            weighted_loss = loss * (1 + masks * (self.mask_weight - 1))
            return weighted_loss.mean()
        
        return loss.mean()