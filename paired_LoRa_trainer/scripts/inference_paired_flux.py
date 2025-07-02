#!/usr/bin/env python3
# scripts/inference_paired_flux.py - 茶歇领替换推理脚本
import os
import sys
import argparse
from pathlib import Path
from PIL import Image
import torch
import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.flux_model import FluxLoRAModel
from utils.visualization import tensor_to_pil

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="FLUX茶歇领替换推理")
    
    parser.add_argument("--model_path", type=str, required=True, help="训练好的LoRA模型路径")
    parser.add_argument("--base_model", type=str, default="black-forest-labs/FLUX.1-fill-dev", help="基础FLUX模型")
    parser.add_argument("--input_image", type=str, required=True, help="输入图像路径")
    parser.add_argument("--prompt", type=str, required=True, help="茶歇领风格提示词")
    parser.add_argument("--mask_image", type=str, help="蒙版图像路径（可选，如果不提供则处理整个图像）")
    parser.add_argument("--output_path", type=str, default="output.png", help="输出图像路径")
    
    # 推理参数
    parser.add_argument("--num_inference_steps", type=int, default=28, help="推理步数")
    parser.add_argument("--guidance_scale", type=float, default=3.5, help="引导强度")
    parser.add_argument("--strength", type=float, default=0.8, help="变换强度（0-1）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    # 图像参数
    parser.add_argument("--width", type=int, default=1024, help="输出宽度")
    parser.add_argument("--height", type=int, default=1024, help="输出高度")
    parser.add_argument("--resize_input", action="store_true", help="是否调整输入图像大小")
    
    # 高级参数
    parser.add_argument("--cache_dir", type=str, default="./cache", help="模型缓存目录")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device", type=str, default="auto", help="设备类型")
    
    return parser.parse_args()

def load_and_preprocess_image(image_path: str, target_size: tuple = None, resize: bool = False) -> Image.Image:
    """加载并预处理图像"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像文件不存在: {image_path}")
    
    image = Image.open(image_path).convert('RGB')
    
    if resize and target_size:
        print(f"调整图像大小: {image.size} -> {target_size}")
        image = image.resize(target_size, Image.LANCZOS)
    elif target_size:
        # 检查尺寸是否匹配
        if image.size != target_size:
            print(f"图像尺寸 {image.size} 与目标尺寸 {target_size} 不匹配")
            print("建议使用 --resize_input 参数自动调整尺寸")
    
    return image

def load_mask_image(mask_path: str, target_size: tuple) -> Image.Image:
    """加载蒙版图像"""
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"蒙版文件不存在: {mask_path}")
    
    mask = Image.open(mask_path).convert('L')
    
    if mask.size != target_size:
        print(f"调整蒙版大小: {mask.size} -> {target_size}")
        mask = mask.resize(target_size, Image.NEAREST)
    
    return mask

def create_default_mask(image_size: tuple) -> Image.Image:
    """创建默认蒙版（处理整个图像）"""
    mask = Image.new('L', image_size, 255)  # 白色蒙版
    return mask

def validate_inputs(args):
    """验证输入参数"""
    if not os.path.exists(args.input_image):
        raise FileNotFoundError(f"输入图像不存在: {args.input_image}")
    
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"模型路径不存在: {args.model_path}")
    
    if args.mask_image and not os.path.exists(args.mask_image):
        raise FileNotFoundError(f"蒙版图像不存在: {args.mask_image}")
    
    # 创建输出目录
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"创建输出目录: {output_dir}")

def setup_device_and_dtype(args):
    """设置设备和数据类型"""
    # 设置设备
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    # 设置数据类型
    if args.torch_dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif args.torch_dtype == "float16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32
    
    print(f"使用设备: {device}")
    print(f"使用数据类型: {torch_dtype}")
    
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    
    return device, torch_dtype

def main():
    """主函数"""
    args = parse_args()
    
    print("FLUX茶歇领替换推理")
    print("=" * 50)
    
    try:
        # 验证输入
        validate_inputs(args)
        
        # 设置设备和数据类型
        device, torch_dtype = setup_device_and_dtype(args)
        
        # 设置随机种子
        torch.manual_seed(args.seed)
        print(f"随机种子: {args.seed}")
        
        # 加载输入图像
        print(f"加载输入图像: {args.input_image}")
        target_size = (args.width, args.height)
        input_image = load_and_preprocess_image(
            args.input_image, 
            target_size, 
            args.resize_input
        )
        
        # 加载或创建蒙版
        if args.mask_image:
            print(f"加载蒙版图像: {args.mask_image}")
            mask_image = load_mask_image(args.mask_image, target_size)
        else:
            print("使用默认蒙版（处理整个图像）")
            mask_image = create_default_mask(target_size)
        
        # 加载FLUX模型
        print(f"加载FLUX模型: {args.base_model}")
        model = FluxLoRAModel(
            model_name=args.base_model,
            device=device,
            torch_dtype=torch_dtype,
            cache_dir=args.cache_dir
        )
        
        # 加载LoRA权重
        print(f"加载LoRA权重: {args.model_path}")
        model.load_lora_weights(args.model_path)
        
        # 设置推理模式
        model.set_train(False)
        
        print(f"提示词: '{args.prompt}'")
        print(f"推理参数:")
        print(f"  - 推理步数: {args.num_inference_steps}")
        print(f"  - 引导强度: {args.guidance_scale}")
        print(f"  - 变换强度: {args.strength}")
        
        # 执行推理
        print("开始生成茶歇领替换效果...")
        
        with torch.no_grad():
            # 获取pipeline
            pipeline = model.pipeline
            
            # 设置生成器
            generator = torch.Generator(device=device).manual_seed(args.seed)
            
            # 根据是否有蒙版选择推理方式
            if args.mask_image:
                # 使用inpainting模式进行茶歇领替换
                result = pipeline(
                    prompt=args.prompt,
                    image=input_image,
                    mask_image=mask_image,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    strength=args.strength,
                    generator=generator,
                    width=args.width,
                    height=args.height
                )
            else:
                # 使用img2img模式
                result = pipeline(
                    prompt=args.prompt,
                    image=input_image,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    strength=args.strength,
                    generator=generator,
                    width=args.width,
                    height=args.height
                )
            
            generated_image = result.images[0]
        
        # 保存结果
        generated_image.save(args.output_path)
        print(f"茶歇领替换完成！结果保存在: {args.output_path}")
        
        # 显示结果信息
        print(f"输出图像信息:")
        print(f"  - 尺寸: {generated_image.size}")
        print(f"  - 模式: {generated_image.mode}")
        print(f"  - 文件大小: {os.path.getsize(args.output_path) / 1024:.1f} KB")
        
        # 可选：创建对比图
        if args.mask_image:
            create_comparison_image(
                input_image, 
                mask_image, 
                generated_image, 
                args.output_path.replace('.png', '_comparison.png')
            )
        
    except KeyboardInterrupt:
        print("\n推理被用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"推理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def create_comparison_image(original: Image.Image, mask: Image.Image, 
                          generated: Image.Image, save_path: str):
    """创建对比图像"""
    try:
        # 创建对比网格
        width, height = original.size
        comparison = Image.new('RGB', (width * 3, height), 'white')
        
        # 放置图像
        comparison.paste(original, (0, 0))
        comparison.paste(mask.convert('RGB'), (width, 0))
        comparison.paste(generated, (width * 2, 0))
        
        # 保存对比图
        comparison.save(save_path)
        print(f"对比图已保存: {save_path}")
        
    except Exception as e:
        print(f"创建对比图失败: {e}")

if __name__ == "__main__":
    main()