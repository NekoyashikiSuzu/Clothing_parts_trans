@echo off
REM setup_paired_project_windows.bat - 项目结构设置脚本
setlocal enabledelayedexpansion

echo 📁 FLUX茶歇领替换项目设置 (Windows)
echo =======================================

echo 正在创建项目目录结构...

REM 创建主要目录
mkdir configs 2>NUL
mkdir data\train 2>NUL
mkdir data\val 2>NUL
mkdir data\masks 2>NUL
mkdir data\test_images 2>NUL
mkdir models 2>NUL
mkdir outputs 2>NUL
mkdir scripts 2>NUL
mkdir training 2>NUL
mkdir utils 2>NUL
mkdir logs 2>NUL
mkdir cache 2>NUL
mkdir data_processing 2>NUL

echo ✅ 目录结构已创建

REM 创建示例配置文件
echo.
echo 📝 创建配置文件...

(
echo # configs/paired_training.yaml - 茶歇领替换训练配置
echo model:
echo   model_name: "black-forest-labs/FLUX.1-fill-dev"
echo   lora_rank: 32
echo   lora_alpha: 16
echo   lora_dropout: 0.1
echo   torch_dtype: "bfloat16"
echo   compile_model: false
echo.
echo training:
echo   batch_size: 2
echo   learning_rate: 5.0e-5
echo   max_steps: 3000
echo   save_steps: 500
echo   eval_steps: 250
echo   logging_steps: 50
echo   warmup_steps: 300
echo   mixed_precision: "bf16"
echo   masked_loss_weight: 2.0
echo   unmasked_loss_weight: 0.1
echo   gradient_accumulation_steps: 4
echo.
echo data:
echo   train_data_dir: "./data/train"
echo   validation_data_dir: "./data/val"
echo   mask_data_dir: "./data/masks"
echo   resolution: 1024
echo   augmentation_prob: 0.5
) > configs\paired_training.yaml

echo ✅ 配置文件已创建

REM 创建数据准备指南
echo.
echo 📚 创建数据准备指南...

(
echo 茶歇领替换数据准备指南
echo ========================
echo.
echo 1. 图片要求:
echo    - 分辨率: 1024x1024 或更高
echo    - 格式: JPG/PNG
echo    - 质量: 清晰、光线均匀
echo    - 内容: 包含清晰可见的服装领子部分
echo.
echo 2. 命名规范:
echo    - 001_original.jpg  ^(变换前 - 普通领子^)
echo    - 001_target.jpg    ^(变换后 - 茶歇领^)
echo    - 001_original.txt  ^(原始描述^)
echo    - 001_target.txt    ^(目标描述^)
echo    - 001_mask.png      ^(蒙版文件^)
echo.
echo 3. 提示词示例:
echo.
echo    original.txt:
echo    "woman wearing white shirt with regular collar, fashion photography, high quality, detailed clothing"
echo.
echo    target.txt:
echo    "woman wearing white off-shoulder top with tea break neckline, elegant style, fashion photography, high quality, detailed clothing"
echo.
echo 4. 蒙版要求:
echo    - 白色区域: 需要变换的领子部分
echo    - 黑色区域: 保持不变的部分
echo    - 灰色区域: 渐变过渡区域
echo    - 精度要求: 边缘要精确贴合领子轮廓
echo.
echo 5. 数据量建议:
echo    - 最少: 10对 ^(测试^)
echo    - 推荐: 50-100对 ^(生产^)
echo    - 高质量: 100-200对 ^(专业^)
echo    - 质量 ^> 数量
echo.
echo 6. 茶歇领风格类型:
echo    - 一字肩 ^(off-shoulder^)
echo    - 露肩 ^(bardot^)
echo    - 船领 ^(boat neck^)
echo    - 深V领 ^(deep V-neck^)
echo    - 方领 ^(square neck^)
) > data\DATA_PREPARATION_GUIDE.txt

echo ✅ 数据指南已创建

REM 创建示例数据说明
echo.
echo 📋 创建示例数据说明文件...

(
echo # 茶歇领替换示例数据说明
echo.
echo ## 数据文件示例
echo.
echo ```
echo data/
echo ├── train/
echo │   ├── 001_original.jpg     # 女性穿普通领衬衫
echo │   ├── 001_target.jpg       # 同一女性穿茶歇领上衣
echo │   ├── 001_original.txt     # "woman wearing white shirt with regular collar"
echo │   ├── 001_target.txt       # "woman wearing white off-shoulder top with tea break neckline"
echo │   ├── 002_original.jpg
echo │   ├── 002_target.jpg
echo │   ├── 002_original.txt
echo │   ├── 002_target.txt
echo │   └── ...
echo ├── val/
echo │   └── ^(same structure^)
echo └── masks/
echo     ├── 001_mask.png         # 领子区域白色蒙版
echo     ├── 002_mask.png
echo     └── ...
echo ```
echo.
echo ## 质量要求
echo.
echo 1. **图像对一致性**: 除了领子外，其他部分^(姿势、光线、背景^)应尽量相似
echo 2. **蒙版精确性**: 蒙版应该精确标注需要替换的领子区域
echo 3. **风格描述**: 提示词要准确描述领子类型和整体风格
echo 4. **颜色一致性**: 建议使用相同或相似颜色的服装进行替换
) > data\README_DATA_FORMAT.md

echo ✅ 示例说明已创建

REM 创建项目根目录README
echo.
echo 📚 创建项目README...

(
echo # FLUX茶歇领替换LoRA训练项目
echo.
echo 基于FLUX-Fill模型训练LoRA，实现精确的茶歇领替换效果。
echo.
echo ## 快速开始
echo.
echo 1. 安装依赖: `pip install -r requirements_py312.txt`
echo 2. 准备数据: 查看 `data\DATA_PREPARATION_GUIDE.txt`
echo 3. 验证数据: `python scripts\validate_paired_data.py --train_data_dir data\train --mask_dir data\masks`
echo 4. 开始训练: `train_paired_flux_windows.bat`
echo 5. 推理测试: `inference_paired_windows.bat`
echo.
echo ## 目录结构
echo.
echo - `configs/`: 配置文件
echo - `data/`: 训练数据
echo - `models/`: 模型文件
echo - `scripts/`: 脚本文件
echo - `outputs/`: 训练输出
echo.
echo 详细说明请查看 `README_PAIRED_TRAINING.md`
) > README.md

echo ✅ 项目README已创建

REM 显示项目结构
echo.
echo 📁 项目结构创建完成:
echo.
tree /F /A . 2>NUL || (
    echo 项目目录:
    echo ├── configs\
    echo │   └── paired_training.yaml
    echo ├── data\
    echo │   ├── train\          ^(放置训练数据^)
    echo │   ├── val\            ^(放置验证数据^)
    echo │   ├── masks\          ^(放置蒙版文件^)
    echo │   ├── test_images\    ^(放置测试图片^)
    echo │   ├── DATA_PREPARATION_GUIDE.txt
    echo │   └── README_DATA_FORMAT.md
    echo ├── models\             ^(放置FLUX模型^)
    echo ├── outputs\            ^(训练输出^)
    echo ├── scripts\            ^(脚本文件^)
    echo ├── training\           ^(训练模块^)
    echo ├── utils\              ^(工具模块^)
    echo ├── data_processing\    ^(数据处理^)
    echo ├── logs\               ^(日志文件^)
    echo ├── cache\              ^(模型缓存^)
    echo └── README.md
)

echo.
echo 🎯 下一步操作:
echo.
echo 1. 📥 下载FLUX模型到 models\ 目录
echo    git clone https://huggingface.co/black-forest-labs/FLUX.1-fill-dev models\FLUX.1-fill-dev
echo.
echo 2. 📸 准备训练数据到 data\train\ 目录
echo    - 按照命名规范: xxx_original.jpg, xxx_target.jpg
echo    - 创建对应的提示词文件: xxx_original.txt, xxx_target.txt
echo    - 创建蒙版文件到 data\masks\: xxx_mask.png
echo.
echo 3. 🔍 验证数据质量
echo    python scripts\validate_paired_data.py --train_data_dir data\train --mask_dir data\masks --verbose
echo.
echo 4. 🚀 开始训练
echo    quick_start_paired_windows.bat
echo.
echo 💡 详细说明请查看:
echo    - README_PAIRED_TRAINING.md ^(完整文档^)
echo    - data\DATA_PREPARATION_GUIDE.txt ^(数据准备^)
echo    - configs\paired_training.yaml ^(配置参数^)

echo.
echo 🎉 项目初始化完成！
echo 请按照上述步骤准备数据并开始训练。

pause