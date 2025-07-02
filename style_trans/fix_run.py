#!/usr/bin/env python3
# fixed_flux_runner.py - 修复版FLUX运行器
import sys
import time
import torch
from pathlib import Path
from datetime import datetime
from PIL import Image

def main():
    """修复版主函数"""
    print("=== 修复版FLUX茶歇领替换工具 ===")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*50)
    
    try:
        # 检查环境
        print("检查环境...")
        if not torch.cuda.is_available():
            print("CUDA不可用")
            return False
        
        print(f"GPU: {torch.cuda.get_device_name()}")
        memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"显存: {memory_gb:.1f}GB")
        
        # 检查文件
        print("\n检查文件...")
        image_path = "input_images/051.png"
        mask_path = "masks/051_mask.png"
        
        if not Path(image_path).exists():
            print(f"图片不存在: {image_path}")
            return False
        
        if not Path(mask_path).exists():
            print(f"Mask不存在: {mask_path}")
            return False
        
        print(f"图片: {image_path}")
        print(f"Mask: {mask_path}")
        
        # 设置GPU优化
        print("\n设置GPU优化...")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.cuda.set_per_process_memory_fraction(0.95)
        
        # 加载模型
        print("\n加载模型...")
        from diffusers import FluxFillPipeline
        
        model_path = "models/flux-fill"
        if not Path(model_path).exists():
            print(f"模型不存在: {model_path}")
            return False
        
        start_load = time.time()
        pipe = FluxFillPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            use_safetensors=True
        )
        
        print(f"移动到GPU...")
        pipe = pipe.to("cuda")
        
        if hasattr(pipe, 'disable_attention_slicing'):
            pipe.disable_attention_slicing()
            print("禁用注意力切片")
        
        if hasattr(pipe, 'disable_vae_slicing'):
            pipe.disable_vae_slicing()
            print("禁用VAE切片")
        
        load_time = time.time() - start_load
        print(f"模型加载完成 ({load_time:.1f}s)")
        
        # GPU内存状态
        allocated = torch.cuda.memory_allocated() / (1024**3)
        print(f"GPU内存: {allocated:.1f}GB")
        
        # 加载图片
        print("\n加载图片...")
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        print(f"原始尺寸: {image.size}")
        
        # 调整尺寸（确保是8的倍数）
        w, h = image.size
        w = (w // 8) * 8
        h = (h // 8) * 8
        if (w, h) != image.size:
            image = image.resize((w, h), Image.Resampling.LANCZOS)
            mask = mask.resize((w, h), Image.Resampling.LANCZOS)
            print(f"调整尺寸: {image.size}")
        
        # 准备提示词
        prompt = """A female model stands with hands in pockets, she wears a white ribbed short-sleeve knit top with elegant sweetheart neckline, fitted silhouette, and the hem of top is tucked into the pants waistband. She pairs it with high-waisted white trousers. The background is beige. Same white ribbed knit fabric and texture as original top, professional photography, high quality."""
        
        print(f"提示词长度: {len(prompt)} 字符")
        
        # 生成参数
        gen_params = {
            "num_inference_steps": 28,
            "guidance_scale": 3.5,
            "strength": 0.85,
        }
        
        print(f"生成参数: {gen_params}")
        
        # 开始生成（不使用negative_prompt）
        print("\n开始生成...")
        start_gen = time.time()
        
        generator = torch.Generator("cuda").manual_seed(42)
        
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = pipe(
                image=image,
                mask_image=mask,
                prompt=prompt,
                generator=generator,
                **gen_params
            ).images[0]
        
        gen_time = time.time() - start_gen
        print(f"生成完成 ({gen_time:.1f}s)")
        
        # 保存结果
        output_path = "output_images/051_fixed_result.jpg"
        Path(output_path).parent.mkdir(exist_ok=True)
        result.save(output_path, quality=95)
        
        # 检查结果
        if Path(output_path).exists():
            file_size = Path(output_path).stat().st_size / (1024*1024)
            print(f"保存成功: {output_path} ({file_size:.1f}MB)")
            
            # 最终GPU内存
            final_memory = torch.cuda.memory_allocated() / (1024**3)
            max_memory = torch.cuda.max_memory_allocated() / (1024**3)
            print(f"GPU内存: {final_memory:.1f}GB (峰值: {max_memory:.1f}GB)")
            
            print(f"\n处理成功!")
            print(f"输出文件: {output_path}")
            print(f"总耗时: 模型加载 {load_time:.1f}s + 生成 {gen_time:.1f}s = {load_time + gen_time:.1f}s")
            
            return True
        else:
            print("文件保存失败")
            return False
        
    except Exception as e:
        print(f"\n处理失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # 清理GPU内存
        if 'torch' in locals() and torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    success = main()
    
    if success:
        print(f"\n恭喜！茶歇领替换成功完成！")
        print(f"请查看 output_images/051_fixed_result.jpg")
    else:
        print(f"\n处理失败，请检查错误信息")
    
    input("\n按任意键退出...")
    sys.exit(0 if success else 1)