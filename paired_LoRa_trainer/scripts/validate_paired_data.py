#!/usr/bin/env python3
# scripts/validate_paired_data.py - 成对数据验证脚本
import os
import argparse
import sys
from pathlib import Path
from PIL import Image
import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def parse_args():
    parser = argparse.ArgumentParser(description="验证茶歇领替换训练数据")
    parser.add_argument("--train_data_dir", type=str, required=True, help="训练数据目录")
    parser.add_argument("--val_data_dir", type=str, help="验证数据目录")
    parser.add_argument("--mask_dir", type=str, required=True, help="蒙版目录")
    parser.add_argument("--min_pairs", type=int, default=10, help="最小数据对数量")
    parser.add_argument("--check_image_quality", action="store_true", help="检查图像质量")
    parser.add_argument("--verbose", action="store_true", help="详细输出")
    return parser.parse_args()

def check_image_quality(image_path: str) -> tuple:
    """检查图像质量"""
    try:
        image = Image.open(image_path)
        
        # 检查基本属性
        width, height = image.size
        mode = image.mode
        
        # 转换为RGB检查
        if mode != 'RGB':
            image = image.convert('RGB')
        
        # 转换为numpy数组进行质量检查
        img_array = np.array(image)
        
        # 检查是否过暗或过亮
        mean_brightness = np.mean(img_array)
        std_brightness = np.std(img_array)
        
        issues = []
        
        # 分辨率检查
        if width < 512 or height < 512:
            issues.append(f"分辨率过低: {width}x{height}")
        
        # 亮度检查
        if mean_brightness < 30:
            issues.append("图像过暗")
        elif mean_brightness > 225:
            issues.append("图像过亮")
        
        # 对比度检查
        if std_brightness < 10:
            issues.append("对比度过低")
        
        # 检查是否是单色图像
        if len(np.unique(img_array)) < 10:
            issues.append("可能是单色图像")
        
        return True, issues, (width, height, mean_brightness, std_brightness)
        
    except Exception as e:
        return False, [f"无法打开图像: {e}"], None

def check_mask_quality(mask_path: str, image_size: tuple) -> tuple:
    """检查蒙版质量"""
    try:
        mask = Image.open(mask_path).convert('L')
        
        # 检查尺寸匹配
        if mask.size != image_size:
            return False, [f"蒙版尺寸不匹配: {mask.size} vs {image_size}"], None
        
        # 转换为numpy数组
        mask_array = np.array(mask)
        
        issues = []
        
        # 检查蒙版值分布
        unique_values = np.unique(mask_array)
        white_pixels = np.sum(mask_array > 200)
        black_pixels = np.sum(mask_array < 50)
        total_pixels = mask_array.size
        
        # 检查是否全黑或全白
        if len(unique_values) == 1:
            if unique_values[0] == 0:
                issues.append("蒙版全黑")
            elif unique_values[0] == 255:
                issues.append("蒙版全白")
            else:
                issues.append("蒙版单色")
        
        # 检查蒙版区域比例
        white_ratio = white_pixels / total_pixels
        if white_ratio < 0.01:
            issues.append("蒙版区域过小 (<1%)")
        elif white_ratio > 0.8:
            issues.append("蒙版区域过大 (>80%)")
        
        # 检查是否有过渡区域
        gray_pixels = np.sum((mask_array > 50) & (mask_array < 200))
        gray_ratio = gray_pixels / total_pixels
        
        stats = {
            'white_ratio': white_ratio,
            'black_ratio': black_pixels / total_pixels,
            'gray_ratio': gray_ratio,
            'unique_values': len(unique_values)
        }
        
        return True, issues, stats
        
    except Exception as e:
        return False, [f"无法打开蒙版: {e}"], None

def validate_text_file(text_path: str) -> tuple:
    """验证文本文件"""
    try:
        with open(text_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        issues = []
        
        if not content:
            issues.append("文本文件为空")
            return False, issues, None
        
        # 检查长度
        if len(content) < 10:
            issues.append("描述过短")
        elif len(content) > 500:
            issues.append("描述过长")
        
        # 检查是否包含茶歇领相关关键词
        collar_keywords = [
            'collar', 'neckline', 'off-shoulder', 'tea break', 
            'off shoulder', 'bardot', 'boat neck', 'scoop neck'
        ]
        
        content_lower = content.lower()
        has_collar_keyword = any(keyword in content_lower for keyword in collar_keywords)
        
        stats = {
            'length': len(content),
            'has_collar_keyword': has_collar_keyword,
            'word_count': len(content.split())
        }
        
        return True, issues, stats
        
    except Exception as e:
        return False, [f"无法读取文本文件: {e}"], None

def validate_data_directory(data_dir: str, mask_dir: str, data_type: str, check_quality: bool = False, verbose: bool = False):
    """验证数据目录"""
    print(f"\n验证{data_type}数据: {data_dir}")
    
    if not os.path.exists(data_dir):
        print(f"数据目录不存在: {data_dir}")
        return [], []
    
    valid_pairs = []
    issues = []
    
    # 找到所有original文件
    original_files = [f for f in os.listdir(data_dir) if '_original.' in f and f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"找到 {len(original_files)} 个original文件")
    
    for original_file in original_files:
        # 提取base_id
        for ext in ['.jpg', '.jpeg', '.png']:
            if original_file.endswith(f'_original{ext}'):
                base_id = original_file[:-len(f'_original{ext}')]
                break
        else:
            issues.append(f"无法解析文件名: {original_file}")
            continue
        
        if verbose:
            print(f"\n检查数据对: {base_id}")
        
        # 检查对应文件
        files_to_check = {
            'original_image': original_file,
            'target_image': None,
            'original_txt': f"{base_id}_original.txt",
            'target_txt': f"{base_id}_target.txt",
            'mask': f"{base_id}_mask.png"
        }
        
        # 查找target图片
        for ext in ['.jpg', '.jpeg', '.png']:
            target_file = f"{base_id}_target{ext}"
            if os.path.exists(os.path.join(data_dir, target_file)):
                files_to_check['target_image'] = target_file
                break
        
        # 检查所有文件是否存在
        missing_files = []
        for file_type, filename in files_to_check.items():
            if filename is None:
                missing_files.append("target_image")
                continue
                
            if file_type == 'mask':
                file_path = os.path.join(mask_dir, filename)
            else:
                file_path = os.path.join(data_dir, filename)
            
            if not os.path.exists(file_path):
                missing_files.append(file_type)
        
        if missing_files:
            issue_msg = f"{base_id}: 缺少 {', '.join(missing_files)}"
            issues.append(issue_msg)
            if verbose:
                print(f"  {issue_msg}")
            continue
        
        # 检查文件内容
        pair_issues = []
        pair_valid = True
        
        try:
            # 检查原始图像
            original_path = os.path.join(data_dir, files_to_check['original_image'])
            valid, img_issues, img_stats = check_image_quality(original_path)
            if not valid:
                pair_issues.extend([f"原始图像: {issue}" for issue in img_issues])
                pair_valid = False
            elif check_quality and img_issues:
                pair_issues.extend([f"原始图像质量: {issue}" for issue in img_issues])
            
            original_size = img_stats[:2] if img_stats else None
            
            # 检查目标图像
            target_path = os.path.join(data_dir, files_to_check['target_image'])
            valid, img_issues, img_stats = check_image_quality(target_path)
            if not valid:
                pair_issues.extend([f"目标图像: {issue}" for issue in img_issues])
                pair_valid = False
            elif check_quality and img_issues:
                pair_issues.extend([f"目标图像质量: {issue}" for issue in img_issues])
            
            target_size = img_stats[:2] if img_stats else None
            
            # 检查图像尺寸是否一致
            if original_size and target_size and original_size != target_size:
                pair_issues.append(f"图像尺寸不匹配: {original_size} vs {target_size}")
                pair_valid = False
            
            # 检查蒙版
            mask_path = os.path.join(mask_dir, files_to_check['mask'])
            image_size = original_size if original_size else (1024, 1024)
            valid, mask_issues, mask_stats = check_mask_quality(mask_path, image_size)
            if not valid:
                pair_issues.extend([f"蒙版: {issue}" for issue in mask_issues])
                pair_valid = False
            elif check_quality and mask_issues:
                pair_issues.extend([f"蒙版质量: {issue}" for issue in mask_issues])
            
            # 检查文本文件
            for txt_type in ['original_txt', 'target_txt']:
                txt_path = os.path.join(data_dir, files_to_check[txt_type])
                valid, txt_issues, txt_stats = validate_text_file(txt_path)
                if not valid:
                    pair_issues.extend([f"{txt_type}: {issue}" for issue in txt_issues])
                    pair_valid = False
                elif check_quality and txt_issues:
                    pair_issues.extend([f"{txt_type}质量: {issue}" for issue in txt_issues])
            
            if pair_issues:
                issue_msg = f"{base_id}: " + "; ".join(pair_issues)
                issues.append(issue_msg)
                if verbose:
                    print(f"  {issue_msg}")
            
            if pair_valid:
                valid_pairs.append(base_id)
                if verbose:
                    print(f"  数据对有效")
                    if check_quality and mask_stats:
                        print(f"    蒙版区域比例: {mask_stats['white_ratio']:.1%}")
                
        except Exception as e:
            issue_msg = f"{base_id}: 验证过程出错 - {e}"
            issues.append(issue_msg)
            if verbose:
                print(f"  {issue_msg}")
    
    print(f"有效数据对: {len(valid_pairs)}")
    if issues:
        print(f"问题数据: {len(issues)}")
        if not verbose:
            for issue in issues[:3]:  # 只显示前3个问题
                print(f"   - {issue}")
            if len(issues) > 3:
                print(f"   ... 还有 {len(issues)-3} 个问题 (使用 --verbose 查看全部)")
    
    return valid_pairs, issues

def main():
    args = parse_args()
    
    print("FLUX茶歇领替换数据验证")
    print("=" * 50)
    
    # 验证训练数据
    train_valid, train_issues = validate_data_directory(
        args.train_data_dir, args.mask_dir, "训练", 
        args.check_image_quality, args.verbose
    )
    
    # 验证验证数据（如果提供）
    val_valid, val_issues = [], []
    if args.val_data_dir:
        val_valid, val_issues = validate_data_directory(
            args.val_data_dir, args.mask_dir, "验证", 
            args.check_image_quality, args.verbose
        )
    
    # 总结报告
    print(f"\n验证总结")
    print("=" * 50)
    print(f"训练数据对: {len(train_valid)}")
    if args.val_data_dir:
        print(f"验证数据对: {len(val_valid)}")
    print(f"总数据对: {len(train_valid) + len(val_valid)}")
    
    total_issues = len(train_issues) + len(val_issues)
    if total_issues == 0:
        print("所有数据验证通过!")
    else:
        print(f"发现 {total_issues} 个问题")
    
    # 数据量评估
    total_pairs = len(train_valid) + len(val_valid)
    if total_pairs < args.min_pairs:
        print(f"\n数据量不足: {total_pairs} < {args.min_pairs}")
        print("建议至少准备10对以上的数据进行训练")
        return False
    elif total_pairs < 20:
        print(f"\n数据量较少: {total_pairs}")
        print("建议增加数据量以获得更好的训练效果")
    elif total_pairs < 50:
        print(f"\n数据量适中: {total_pairs}")
        print("可以开始训练，建议准备更多数据以提升效果")
    else:
        print(f"\n数据量充足: {total_pairs}")
        print("数据量充足，可以开始训练!")
    
    # 训练建议
    print(f"\n茶歇领替换训练建议:")
    print("1. 确保original和target图像除了领子外其他部分尽量相似")
    print("2. 蒙版要精确标注茶歇领区域，边缘要平滑")
    print("3. 提示词要详细描述领子类型和风格")
    print("4. 建议使用较小的学习率(5e-5)和适中的蒙版权重(2.0)")
    
    print(f"\n使用以下命令开始训练:")
    print(f"python scripts/train_paired_flux.py \\")
    print(f"  --train_data_dir {args.train_data_dir} \\")
    if args.val_data_dir:
        print(f"  --val_data_dir {args.val_data_dir} \\")
    print(f"  --mask_dir {args.mask_dir} \\")
    print(f"  --model_name black-forest-labs/FLUX.1-fill-dev \\")
    print(f"  --output_dir ./outputs/collar_replacement \\")
    print(f"  --batch_size 2 \\")
    print(f"  --learning_rate 5e-5 \\")
    print(f"  --max_steps 3000 \\")
    print(f"  --masked_loss_weight 2.0")
    
    return total_issues == 0 and total_pairs >= args.min_pairs

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

"""
# 基本验证
python scripts/validate_paired_data.py --train_data_dir data/train --mask_dir data/masks

# 详细验证（包含图像质量检查）
python scripts/validate_paired_data.py \
  --train_data_dir data/train \
  --val_data_dir data/val \
  --mask_dir data/masks \
  --min_pairs 10 \
  --check_image_quality \
  --verbose
"""