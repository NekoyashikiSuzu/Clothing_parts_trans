@echo off
REM quick_start_paired_windows.bat - 茶歇领替换快速启动脚本
setlocal enabledelayedexpansion

echo 🚀 FLUX茶歇领替换 - 快速启动向导
echo ====================================

REM 检查Python环境
echo 🔍 检查Python环境...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python未安装或未添加到PATH
    echo.
    echo 💡 请按以下步骤安装Python:
    echo 1. 访问 https://www.python.org/downloads/
    echo 2. 下载Python 3.12
    echo 3. 安装时勾选"Add Python to PATH"
    pause
    exit /b 1
)

python -c "import sys; print('Python版本:', sys.version)" 2>nul
echo ✅ Python环境检查通过

REM 检查必要包
echo.
echo 🔍 检查必要的Python包...
python -c "import torch; print('PyTorch版本:', torch.__version__)" 2>nul
if %errorlevel% neq 0 (
    echo ❌ PyTorch未安装
    echo.
    echo 💡 正在安装依赖包...
    pip install -r requirements_py312.txt
    if %errorlevel% neq 0 (
        echo ❌ 依赖包安装失败
        echo 请手动运行: pip install -r requirements_py312.txt
        pause
        exit /b 1
    )
)

python -c "import diffusers; print('Diffusers版本:', diffusers.__version__)" 2>nul
if %errorlevel% neq 0 (
    echo ❌ Diffusers未安装，正在安装...
    pip install diffusers[torch] transformers accelerate
)

echo ✅ Python包检查通过

REM 检查项目结构
echo.
echo 🔍 检查项目结构...
if not exist "data" (
    echo 📁 创建项目目录结构...
    call setup_paired_project_windows.bat
)

echo ✅ 项目结构检查通过

REM 检查GPU
echo.
echo 🔍 检查GPU环境...
python -c "import torch; print('CUDA可用:', torch.cuda.is_available()); print('GPU数量:', torch.cuda.device_count()); print('GPU名称:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '无GPU')" 2>nul

REM 检查数据
echo.
echo 🔍 检查训练数据...
set "HAS_DATA=false"
if exist "data\train\*_original.*" (
    set "HAS_DATA=true"
)

if "%HAS_DATA%"=="false" (
    echo ❌ 未找到训练数据
    echo.
    echo 💡 数据准备指南:
    echo ================
    echo.
    echo 请按照以下结构准备数据:
    echo.
    echo data\
    echo ├── train\
    echo │   ├── 001_original.jpg     # 普通领子的图片
    echo │   ├── 001_target.jpg       # 茶歇领的图片
    echo │   ├── 001_original.txt     # 普通领子的描述
    echo │   ├── 001_target.txt       # 茶歇领的描述
    echo │   └── ...
    echo ├── val\                     # 验证数据(可选)
    echo │   └── (同train结构)
    echo └── masks\
    echo     ├── 001_mask.png         # 领子区域蒙版
    echo     └── ...
    echo.
    echo 📝 提示词示例:
    echo ===============
    echo.
    echo 001_original.txt:
    echo woman wearing white shirt with regular collar, fashion photography, high quality
    echo.
    echo 001_target.txt:
    echo woman wearing white off-shoulder top with tea break neckline, elegant style, fashion photography, high quality
    echo.
    echo 🎭 蒙版要求:
    echo ============
    echo - 白色: 需要替换的领子区域
    echo - 黑色: 保持不变的区域
    echo - 灰色: 过渡区域
    echo.
    set /p CONTINUE="数据准备完成后按任意键继续，或按N退出 (Enter/N): "
    if /i "!CONTINUE!"=="N" (
        echo 👋 请准备好数据后重新运行此脚本
        pause
        exit /b 0
    )
)

REM 验证数据
echo.
echo 📋 验证训练数据...
python scripts\validate_paired_data.py --train_data_dir data\train --mask_dir data\masks --min_pairs 5 --verbose

if %errorlevel% neq 0 (
    echo.
    echo ❌ 数据验证失败
    echo 请根据上方提示修复数据问题后重新运行
    pause
    exit /b 1
)

echo ✅ 数据验证通过

REM 选择操作
echo.
echo 🎯 选择要执行的操作:
echo ====================
echo 1. 开始训练茶歇领替换模型
echo 2. 使用现有模型进行推理
echo 3. 查看训练进度和日志
echo 4. 退出
echo.

set /p CHOICE="请选择操作 (1-4): "

if "%CHOICE%"=="1" goto :train
if "%CHOICE%"=="2" goto :inference
if "%CHOICE%"=="3" goto :logs
if "%CHOICE%"=="4" goto :exit

echo ❌ 无效选择，请重新运行脚本
pause
exit /b 1

:train
echo.
echo 🏃 启动训练...
echo ================
echo.
echo 💡 训练参数说明:
echo - 批次大小: 2 (适合12GB显存)
echo - 学习率: 5e-5 (适合茶歇领替换)
echo - 训练步数: 3000 (约1-2小时)
echo - LoRA秩: 32 (平衡质量和大小)
echo - 蒙版权重: 2.0 (重点关注领子区域)
echo.
echo 训练期间可以按Ctrl+C中断...
echo.
pause

call train_paired_flux_windows.bat
goto :end

:inference
echo.
echo 🎨 启动推理...
echo ================
echo.

REM 检查是否有训练好的模型
set "MODEL_FOUND=false"
if exist "outputs\*\checkpoints\checkpoint-*.pt" (
    set "MODEL_FOUND=true"
)

if "%MODEL_FOUND%"=="false" (
    echo ❌ 未找到训练好的模型
    echo 请先进行训练或提供模型路径
    pause
    goto :end
)

echo 将使用最新的训练模型进行推理...
echo.
pause

call inference_paired_windows.bat
goto :end

:logs
echo.
echo 📊 查看训练进度...
echo ===================
echo.

if exist "outputs" (
    echo 📁 训练输出目录:
    for /d %%d in (outputs\*) do (
        echo   %%d
        if exist "%%d\logs" (
            echo     - 日志目录: %%d\logs
        )
        if exist "%%d\checkpoints" (
            echo     - 检查点: %%d\checkpoints
        )
        if exist "%%d\samples" (
            echo     - 生成样本: %%d\samples
        )
    )
    
    echo.
    echo 💡 查看训练进度的方法:
    echo 1. 查看日志文件: outputs\[训练目录]\logs\training.log
    echo 2. 启动TensorBoard: tensorboard --logdir outputs\[训练目录]\logs
    echo 3. 查看生成样本: outputs\[训练目录]\samples\
    
    set /p OPEN_OUTPUT="是否打开输出目录? (y/N): "
    if /i "!OPEN_OUTPUT!"=="y" (
        explorer outputs
    )
) else (
    echo ❌ 未找到训练输出
    echo 请先开始训练
)

pause
goto :end

:exit
echo 👋 感谢使用FLUX茶歇领替换工具！
pause
exit /b 0

:end
echo.
echo 🎉 操作完成！
echo.
echo 💡 更多帮助:
echo ============
echo - 查看README_PAIRED_TRAINING.md了解详细说明
echo - 训练问题: 检查data\DATA_PREPARATION_GUIDE.txt
echo - 技术支持: 查看项目文档或提交issue
echo.
echo 🔄 重新运行此脚本可以执行其他操作
pause