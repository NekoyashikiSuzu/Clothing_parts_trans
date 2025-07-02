# FLUX茶歇领替换LoRA训练项目

## 🎯 项目概述

本项目基于FLUX-Fill模型训练LoRA，实现图片中模特衣服领子的精确替换（如将普通领子替换为茶歇领），同时保持图片整体风格协调统一。

## 📁 数据集结构

```
data/
├── train/                    # 训练数据
│   ├── 001_original.jpg     # 变换前图片（普通领子）
│   ├── 001_target.jpg       # 变换后图片（茶歇领）
│   ├── 001_original.txt     # 原始风格描述
│   ├── 001_target.txt       # 目标风格描述
│   ├── 002_original.jpg
│   ├── 002_target.jpg
│   ├── 002_original.txt
│   ├── 002_target.txt
│   └── ...
├── val/                     # 验证数据（结构同train）
│   └── (same structure as train)
└── masks/                   # 变换区域蒙版
    ├── 001_mask.png        # 对应001的领子区域蒙版
    ├── 002_mask.png        # 对应002的领子区域蒙版
    └── ...
```

## 🏷️ 数据要求

### 图片要求
- **分辨率**: 1024x1024或更高
- **格式**: JPG/PNG
- **质量**: 清晰、光线均匀、无模糊
- **内容**: 包含清晰可见的服装领子部分

### 提示词示例
```
# 001_original.txt
woman wearing white shirt with regular collar, fashion photography, high quality, detailed clothing, professional lighting

# 001_target.txt  
woman wearing white off-shoulder top with tea break neckline, elegant style, fashion photography, high quality, detailed clothing, professional lighting
```

### 蒙版要求
- **白色区域**: 需要替换的领子部分
- **黑色区域**: 保持不变的背景和其他部分
- **灰色区域**: 柔和过渡的边缘区域
- **精度**: 蒙版边缘应该精确贴合领子轮廓

## 🚀 快速开始

### 1. 环境准备
```bash
# 创建Python 3.12环境
conda create -n flux_collar python=3.12
conda activate flux_collar

# 安装依赖
pip install -r requirements_py312.txt
```

### 2. 项目初始化
```bash
# Windows用户
setup_paired_project_windows.bat

# 或手动创建目录结构
mkdir data\train data\val data\masks data\test_images
mkdir models outputs logs cache
```

### 3. 数据准备
将准备好的成对数据按照上述结构放入对应目录，确保：
- 每个original图片都有对应的target图片
- 每对图片都有对应的蒙版文件
- 文件命名严格按照规范：`{id}_original.jpg`, `{id}_target.jpg`, `{id}_mask.png`

### 4. 数据验证
```bash
python scripts/validate_paired_data.py --train_data_dir data/train --mask_dir data/masks --min_pairs 10
```

### 5. 开始训练
```bash
# Windows用户
train_paired_flux_windows.bat

# 或手动运行
python scripts/train_paired_flux.py \
  --train_data_dir data/train \
  --val_data_dir data/val \
  --mask_dir data/masks \
  --model_name "black-forest-labs/FLUX.1-fill-dev" \
  --output_dir outputs/collar_replacement_lora \
  --batch_size 2 \
  --learning_rate 5e-5 \
  --max_steps 3000 \
  --lora_rank 32 \
  --mask_weight 2.0
```

## 🎨 推理使用

### 单张图片推理
```bash
python scripts/inference_paired_flux.py \
  --model_path outputs/collar_replacement_lora/checkpoint-1500 \
  --input_image test_image.jpg \
  --prompt "woman wearing off-shoulder top with tea break neckline, elegant style" \
  --mask_image mask.png \
  --output_path result.jpg
```

### 批量推理
```bash
# Windows用户
inference_paired_windows.bat
```

## ⚙️ 训练参数说明

### 核心参数
- `--batch_size`: 批次大小，建议2-4（取决于显存）
- `--learning_rate`: 学习率，茶歇领替换推荐5e-5
- `--max_steps`: 训练步数，推荐2000-5000步
- `--lora_rank`: LoRA秩，推荐16-64
- `--mask_weight`: 蒙版区域权重，推荐1.5-3.0

### 高级参数
- `--mixed_precision`: 混合精度，推荐bf16
- `--gradient_accumulation_steps`: 梯度累积，推荐4-8
- `--warmup_steps`: 预热步数，推荐总步数的10%
- `--save_steps`: 保存间隔，推荐500步

## 📊 训练监控

### TensorBoard
```bash
tensorboard --logdir logs/
```

### 关键指标
- `train/loss`: 训练损失，应逐步下降
- `train/masked_loss`: 蒙版区域损失
- `val/fid_score`: 验证集FID分数
- `learning_rate`: 学习率变化

## 🔧 故障排除

### 常见问题
1. **显存不足**: 减少batch_size，启用gradient_checkpointing
2. **数据加载慢**: 检查硬盘性能，调整num_workers
3. **训练不收敛**: 调整学习率，检查数据质量
4. **生成效果差**: 增加训练步数，提高数据质量

### 性能优化
- 使用SSD存储训练数据
- 启用mixed_precision训练
- 合理设置gradient_accumulation_steps
- 使用torch.compile（Python 3.12+）

## 📈 效果评估

### 定量指标
- FID分数：衡量生成图片质量
- LPIPS距离：衡量感知相似度
- 蒙版区域SSIM：衡量替换精度

### 定性评估
- 茶歇领形状准确性
- 服装材质一致性
- 光照阴影自然性
- 整体风格协调性

## 🎯 最佳实践

### 数据准备
1. 确保original和target图片除了领子外其他部分尽量相似
2. 蒙版边缘要精确，避免包含不相关区域
3. 提示词要详细描述风格特征
4. 数据质量比数量更重要，建议精选50-200对高质量数据

### 训练策略
1. 从较小的学习率开始，观察损失变化
2. 定期检查验证集效果，避免过拟合
3. 保存多个检查点，便于选择最佳模型
4. 使用不同的随机种子进行多次训练

### 推理优化
1. 根据具体需求调整guidance_scale
2. 实验不同的推理步数找到速度质量平衡点
3. 对于批量处理，考虑使用更大的batch_size

## 📝 更新日志

- v1.0: 初始版本，支持基础茶歇领替换
- v1.1: 优化蒙版处理，提升边缘质量
- v1.2: 添加数据增强，改善泛化能力

## 🤝 贡献指南

欢迎提交问题和改进建议！在提交前请确保：
1. 遵循代码风格规范
2. 添加必要的测试
3. 更新相关文档

## 📄 许可证

本项目遵循MIT许可证。请注意FLUX模型本身可能有其他许可证要求。

# 训练参数
python scripts/train_paired_flux.py \
  --model_name "./models/FLUX.1-fill-dev" \
  --train_data_dir data/train \
  --mask_dir data/masks \
  --output_dir outputs/collar_replacement

# 模型位置
flux_collar_replacement/
├── models/
│   └── FLUX.1-fill-dev/           # 从HuggingFace下载的完整模型
│       ├── model_index.json
│       ├── scheduler/
│       ├── text_encoder/
│       ├── tokenizer/
│       ├── transformer/
│       ├── vae/
│       └── README.md
├── data/
├── scripts/
└── ...
