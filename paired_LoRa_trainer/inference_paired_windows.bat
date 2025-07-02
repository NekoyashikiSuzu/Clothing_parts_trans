@echo off
REM inference_paired_windows.bat - Windows茶歇领替换推理脚本
setlocal enabledelayedexpansion

echo 🎨 FLUX茶歇领替换推理脚本
echo ==============================

REM 检查Python环境
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python未安装或未添加到PATH
    pause
    exit /b 1
)

REM 查找最新的训练模型
set "MODEL_PATH="
set "LATEST_STEP=0"

if exist "outputs" (
    for /d %%d in (outputs\*) do (
        if exist "%%d\checkpoints" (
            for /f %%f in ('dir /b "%%d\checkpoints\checkpoint-*.pt" 2^>nul') do (
                set "FILENAME=%%f"
                set "STEPNUM=!FILENAME:checkpoint-=!"
                set "STEPNUM=!STEPNUM:.pt=!"
                if !STEPNUM! gtr !LATEST_STEP! (
                    set "LATEST_STEP=!STEPNUM!"
                    set "MODEL_PATH=%%d\checkpoints\checkpoint-!STEPNUM!"
                )
            )
        )
    )
)

if "%MODEL_PATH%"=="" (
    echo ❌ 未找到训练好的模型
    echo 请先运行train_paired_flux_windows.bat进行训练
    pause
    exit /b 1
)

echo ✅ 找到模型: %MODEL_PATH% (步骤: %LATEST_STEP%)

REM 检查测试图像目录
if not exist "data\test_images" (
    echo 📁 创建测试图像目录...
    mkdir data\test_images
    echo ✅ 请将测试图像放入 data\test_images\ 目录
)

REM 检查是否有测试图像
set "TEST_COUNT=0"
for %%f in (data\test_images\*.jpg data\test_images\*.jpeg data\test_images\*.png) do (
    set /a TEST_COUNT+=1
)

if %TEST_COUNT% equ 0 (
    echo ❌ 未找到测试图像
    echo 请将测试图像放入 data\test_images\ 目录
    echo 支持的格式: .jpg, .jpeg, .png
    pause
    exit /b 1
)

echo ✅ 找到 %TEST_COUNT% 个测试图像

REM 创建输出目录
if not exist "outputs\inference_results" (
    mkdir outputs\inference_results
)

echo.
echo 🎯 推理参数配置:
echo ================

REM 可配置的推理参数
set PROMPT=woman wearing off-shoulder top with tea break neckline, elegant style, fashion photography, high quality, detailed clothing
set NUM_INFERENCE_STEPS=28
set GUIDANCE_SCALE=3.5
set STRENGTH=0.8
set SEED=42

echo 提示词: %PROMPT%
echo 推理步数: %NUM_INFERENCE_STEPS%
echo 引导强度: %GUIDANCE_SCALE%
echo 变换强度: %STRENGTH%
echo 随机种子: %SEED%

echo.
echo 🚀 开始批量推理...
echo.

REM 处理每个测试图像
set "PROCESSED=0"
set "FAILED=0"

for %%f in (data\test_images\*.jpg data\test_images\*.jpeg data\test_images\*.png) do (
    set "INPUT_FILE=%%f"
    set "BASENAME=%%~nf"
    set "OUTPUT_FILE=outputs\inference_results\!BASENAME!_tea_break_collar.png"
    set "COMPARISON_FILE=outputs\inference_results\!BASENAME!_comparison.png"
    
    echo 📸 处理: !BASENAME!
    
    REM 检查是否有对应的蒙版文件
    set "MASK_FILE="
    if exist "data\masks\!BASENAME!_mask.png" (
        set "MASK_FILE=data\masks\!BASENAME!_mask.png"
        echo   🎭 使用蒙版: !MASK_FILE!
    ) else if exist "data\masks\!BASENAME!.png" (
        set "MASK_FILE=data\masks\!BASENAME!.png"
        echo   🎭 使用蒙版: !MASK_FILE!
    ) else (
        echo   🎭 未找到蒙版，将处理整个图像
    )
    
    REM 执行推理
    if "!MASK_FILE!"=="" (
        python scripts\inference_paired_flux.py ^
          --model_path "%MODEL_PATH%" ^
          --input_image "!INPUT_FILE!" ^
          --prompt "%PROMPT%" ^
          --output_path "!OUTPUT_FILE!" ^
          --num_inference_steps %NUM_INFERENCE_STEPS% ^
          --guidance_scale %GUIDANCE_SCALE% ^
          --strength %STRENGTH% ^
          --seed %SEED% ^
          --resize_input
    ) else (
        python scripts\inference_paired_flux.py ^
          --model_path "%MODEL_PATH%" ^
          --input_image "!INPUT_FILE!" ^
          --prompt "%PROMPT%" ^
          --mask_image "!MASK_FILE!" ^
          --output_path "!OUTPUT_FILE!" ^
          --num_inference_steps %NUM_INFERENCE_STEPS% ^
          --guidance_scale %GUIDANCE_SCALE% ^
          --strength %STRENGTH% ^
          --seed %SEED% ^
          --resize_input
    )