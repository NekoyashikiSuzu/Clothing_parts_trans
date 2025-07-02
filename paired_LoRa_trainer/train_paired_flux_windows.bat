@echo off
REM train_paired_flux_windows.bat - Windows茶歇领替换训练脚本
setlocal enabledelayedexpansion

echo 🎨 FLUX茶歇领替换LoRA训练启动脚本
echo ========================================

REM 检查Python环境
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python未安装或未添加到PATH
    echo 请先安装Python 3.12并添加到系统PATH
    pause
    exit /b 1
)

echo ✅ Python环境检查通过

REM 检查必要目录
if not exist "data\train" (
    echo ❌ 训练数据目录不存在: data\train
    echo 请先运行setup_paired_project_windows.bat创建目录结构
    pause
    exit /b 1
)

if not exist "data\masks" (
    echo ❌ 蒙版目录不存在: data\masks
    echo 请确保蒙版文件已放置在data\masks目录中
    pause
    exit /b 1
)

echo ✅ 目录结构检查通过

REM 检查是否有训练数据
set "train_files=0"
for %%f in (data\train\*_original.*) do (
    set /a train_files+=1
)

if %train_files% lss 5 (
    echo ❌ 训练数据不足: 发现 %train_files% 个original文件
    echo 建议至少准备10对训练数据
    echo 请检查data\train目录中的数据文件
    pause
    exit /b 1
)

echo ✅ 发现 %train_files% 个训练数据文件

REM 验证数据
echo.
echo 📋 验证训练数据...
python scripts/validate_paired_data.py --train_data_dir data/train --mask_dir data/masks --min_pairs 5

if %errorlevel% neq 0 (
    echo ❌ 数据验证失败，请检查数据格式和完整性
    echo.
    echo 💡 数据要求:
    echo   - 每个xxx_original.jpg都要有对应的xxx_target.jpg
    echo   - 每个xxx都要有对应的xxx_mask.png
    echo   - 每个图片都要有对应的.txt描述文件
    pause
    exit /b 1
)

echo ✅ 数据验证通过

REM 检查GPU
python -c "import torch; print('GPU可用:', torch.cuda.is_available()); print('GPU数量:', torch.cuda.device_count()); print('GPU名称:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')" 2>nul

REM 设置训练参数
echo.
echo ⚙️ 配置训练参数...

REM 可自定义的参数
set MODEL_NAME=./models/FLUX.1-fill-dev
set OUTPUT_DIR=outputs/collar_replacement_%date:~0,4%%date:~5,2%%date:~8,2%_%time:~0,2%%time:~3,2%
set BATCH_SIZE=2
set LEARNING_RATE=5e-5
set MAX_STEPS=3000
set LORA_RANK=32
set LORA_ALPHA=16
set MASK_WEIGHT=2.0
set MIXED_PRECISION=bf16

REM 处理输出目录名中的空格和冒号
set OUTPUT_DIR=%OUTPUT_DIR: =%
set OUTPUT_DIR=%OUTPUT_DIR::=%

echo 📊 训练配置:
echo   - 模型: %MODEL_NAME%
echo   - 输出目录: %OUTPUT_DIR%
echo   - 批次大小: %BATCH_SIZE%
echo   - 学习率: %LEARNING_RATE%
echo   - 最大步数: %MAX_STEPS%
echo   - LoRA秩: %LORA_RANK%
echo   - 蒙版权重: %MASK_WEIGHT%
echo   - 混合精度: %MIXED_PRECISION%

echo.
echo 🚀 开始训练茶歇领替换模型...
echo 训练过程中可以按Ctrl+C中断
echo.

REM 执行训练
python scripts/train_paired_flux.py ^
  --train_data_dir data/train ^
  --val_data_dir data/val ^
  --mask_dir data/masks ^
  --model_name "%MODEL_NAME%" ^
  --output_dir "%OUTPUT_DIR%" ^
  --batch_size %BATCH_SIZE% ^
  --learning_rate %LEARNING_RATE% ^
  --max_steps %MAX_STEPS% ^
  --lora_rank %LORA_RANK% ^
  --lora_alpha %LORA_ALPHA% ^
  --mixed_precision %MIXED_PRECISION% ^
  --gradient_accumulation_steps 4 ^
  --save_steps 500 ^
  --eval_steps 250 ^
  --logging_steps 50 ^
  --masked_loss_weight %MASK_WEIGHT% ^
  --warmup_steps 300 ^
  --weight_decay 0.01 ^
  --max_grad_norm 1.0 ^
  --augmentation_prob 0.5 ^
  --seed 42

if %errorlevel% equ 0 (
    echo.
    echo 🎉 训练完成！
    echo.
    echo 📁 输出文件位置:
    echo   - 模型检查点: %OUTPUT_DIR%\checkpoints\
    echo   - 训练样本: %OUTPUT_DIR%\samples\
    echo   - 训练日志: %OUTPUT_DIR%\logs\
    echo.
    echo 💡 使用以下命令进行推理测试:
    echo python scripts/inference_paired_flux.py ^
      --model_path "%OUTPUT_DIR%\checkpoint-1500" ^
      --input_image "test_image.jpg" ^
      --prompt "woman wearing off-shoulder top with tea break neckline, elegant style" ^
      --mask_image "test_mask.png" ^
      --output_path "result.jpg"
) else (
    echo.
    echo ❌ 训练失败，错误代码: %errorlevel%
    echo 请检查上方的错误信息
)

echo.
pause