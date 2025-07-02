# utils/checkpoint_utils.py - 检查点管理工具
import os
import torch
import json
import glob
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)

class CheckpointManager:
    """检查点管理器"""
    
    def __init__(self, output_dir: str, save_total_limit: int = 5):
        self.output_dir = output_dir
        self.save_total_limit = save_total_limit
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
    
    def save_checkpoint(self, checkpoint_data: Dict, step: int, is_best: bool = False) -> str:
        """保存检查点"""
        checkpoint_name = f"checkpoint-{step}.pt"
        checkpoint_path = os.path.join(self.output_dir, checkpoint_name)
        
        # 保存检查点
        torch.save(checkpoint_data, checkpoint_path)
        
        # 保存元数据
        metadata = {
            'step': step,
            'is_best': is_best,
            'checkpoint_path': checkpoint_path,
            'config': checkpoint_data.get('config', {})
        }
        
        metadata_path = checkpoint_path.replace('.pt', '_metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # 如果是最佳模型，创建符号链接或复制
        if is_best:
            best_path = os.path.join(self.output_dir, "best_model.pt")
            if os.path.exists(best_path):
                os.remove(best_path)
            
            # Windows不支持符号链接，直接复制
            import shutil
            shutil.copy2(checkpoint_path, best_path)
            
            logger.info(f"保存最佳模型: {best_path}")
        
        # 清理旧检查点
        self._cleanup_old_checkpoints()
        
        logger.info(f"检查点已保存: {checkpoint_path}")
        return checkpoint_path
    
    def _cleanup_old_checkpoints(self):
        """清理旧的检查点"""
        if self.save_total_limit <= 0:
            return
        
        # 获取所有检查点文件
        checkpoint_pattern = os.path.join(self.output_dir, "checkpoint-*.pt")
        checkpoint_files = glob.glob(checkpoint_pattern)
        
        if len(checkpoint_files) <= self.save_total_limit:
            return
        
        # 按修改时间排序
        checkpoint_files.sort(key=os.path.getmtime)
        
        # 删除最旧的检查点
        files_to_delete = checkpoint_files[:-self.save_total_limit]
        
        for file_path in files_to_delete:
            try:
                os.remove(file_path)
                
                # 同时删除元数据文件
                metadata_path = file_path.replace('.pt', '_metadata.json')
                if os.path.exists(metadata_path):
                    os.remove(metadata_path)
                
                logger.info(f"删除旧检查点: {file_path}")
                
            except Exception as e:
                logger.warning(f"删除检查点失败 {file_path}: {e}")
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict:
        """加载检查点"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
        
        logger.info(f"加载检查点: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        return checkpoint
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """获取最新的检查点路径"""
        checkpoint_pattern = os.path.join(self.output_dir, "checkpoint-*.pt")
        checkpoint_files = glob.glob(checkpoint_pattern)
        
        if not checkpoint_files:
            return None
        
        # 按修改时间排序，获取最新的
        latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
        return latest_checkpoint
    
    def get_best_checkpoint(self) -> Optional[str]:
        """获取最佳检查点路径"""
        best_path = os.path.join(self.output_dir, "best_model.pt")
        
        if os.path.exists(best_path):
            return best_path
        
        return None
    
    def list_checkpoints(self) -> List[Dict]:
        """列出所有检查点"""
        checkpoint_pattern = os.path.join(self.output_dir, "checkpoint-*.pt")
        checkpoint_files = glob.glob(checkpoint_pattern)
        
        checkpoints = []
        
        for checkpoint_file in checkpoint_files:
            # 提取步数
            basename = os.path.basename(checkpoint_file)
            step_str = basename.replace('checkpoint-', '').replace('.pt', '')
            
            try:
                step = int(step_str)
            except ValueError:
                continue
            
            # 获取元数据
            metadata_path = checkpoint_file.replace('.pt', '_metadata.json')
            metadata = {}
            
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                except Exception as e:
                    logger.warning(f"读取元数据失败 {metadata_path}: {e}")
            
            checkpoints.append({
                'step': step,
                'path': checkpoint_file,
                'metadata': metadata,
                'size_mb': os.path.getsize(checkpoint_file) / 1024 / 1024,
                'modified_time': os.path.getmtime(checkpoint_file)
            })
        
        # 按步数排序
        checkpoints.sort(key=lambda x: x['step'])
        
        return checkpoints