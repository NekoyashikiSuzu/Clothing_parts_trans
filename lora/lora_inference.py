# ..\lora_inference.py - LoRA推理
import sys
import torch
from pathlib import Path
from PIL import Image
import time

def main():
    """使用训练好的LoRA进行茶歇领替换"""
    print("=== FLUX-Fill LoRA 推理工具 ===")
    
    try:
        from diffusers import FluxFillPipeline
        from peft import PeftModel
        
        # 检查LoRA模型
        lora_path = "lora_output/final_model"
        if not Path(lora_path).exists():
            print(f"错误: LoRA模型不存在: {lora_path}")
            print("请先运行 flux_lora_trainer.py 训练LoRA模型")
            return False
        
        # 加载基础模型
        print("加载基础FLUX-Fill模型...")
        pipe = FluxFillPipeline.from_pretrained(
            "models/flux-fill",
            torch_dtype=torch.bfloat16,
            use_safetensors=True
        )
        
        # 加载LoRA
        print("加载训练好的LoRA...")
        pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path)
        
        # 移动到GPU
        pipe = pipe.to("cuda")
        
        # 优化设置
        if hasattr(pipe, 'disable_attention_slicing'):
            pipe.disable_attention_slicing()
        if hasattr(pipe, 'disable_vae_slicing'):
            pipe.disable_vae_slicing()
        
        print("模型加载完成")
        
        # 处理图像
        image_path = "input_images/051.png"
        mask_path = "masks/051_mask.png"
        
        if not Path(image_path).exists():
            print(f"错误: 图像不存在: {image_path}")
            return False
        
        if not Path(mask_path).exists():
            print(f"错误: 蒙版不存在: {mask_path}")
            return False
        
        # 加载图像
        print("加载图像...")
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        # 调整尺寸
        w, h = image.size
        w = (w // 8) * 8
        h = (h // 8) * 8
        image = image.resize((w, h), Image.Resampling.LANCZOS)
        mask = mask.resize((w, h), Image.Resampling.LANCZOS)
        
        # 使用训练好的LoRA生成茶歇领
        print("生成茶歇领...")
        start_time = time.time()
        
        # 重点：使用训练时的关键词
        prompt = "A female model with V-shaped tea break collar, white ribbed knit fabric, elegant neckline design, professional photography"
        
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = pipe(
                image=image,
                mask_image=mask,
                prompt=prompt,
                num_inference_steps=28,
                guidance_scale=4.0,
                strength=0.9,
                generator=torch.Generator("cuda").manual_seed(42)
            ).images[0]
        
        gen_time = time.time() - start_time
        print(f"生成完成 ({gen_time:.1f}s)")
        
        # 保存结果
        output_path = "output_images/051_lora_tea_break_collar.jpg"
        Path(output_path).parent.mkdir(exist_ok=True)
        result.save(output_path, quality=95)
        
        print(f"✅ 茶歇领替换成功!")
        print(f"输出文件: {output_path}")
        print(f"处理时间: {gen_time:.1f}秒")
        
        # 显示GPU内存使用情况
        if torch.cuda.is_available():
            memory_used = torch.cuda.max_memory_allocated() / 1e9
            print(f"峰值显存使用: {memory_used:.1f}GB")
        
        return True
        
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请确保安装了以下库:")
        print("pip install diffusers peft accelerate transformers")
        return False
    except Exception as e:
        print(f"推理失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def batch_inference():
    """批量推理"""
    print("=== LoRA 批量推理 ===")
    
    try:
        from diffusers import FluxFillPipeline
        from peft import PeftModel
        
        # 加载模型
        print("加载模型...")
        lora_path = "lora_output/final_model"
        pipe = FluxFillPipeline.from_pretrained(
            "models/flux-fill",
            torch_dtype=torch.bfloat16,
            use_safetensors=True
        )
        pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path)
        pipe = pipe.to("cuda")
        
        # 获取所有输入图像
        input_dir = Path("input_images")
        mask_dir = Path("masks")
        output_dir = Path("output_images")
        
        image_files = list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg"))
        
        if not image_files:
            print("未找到输入图像")
            return False
        
        print(f"找到 {len(image_files)} 张图像")
        
        successful = 0
        total_time = 0
        
        for i, img_file in enumerate(image_files, 1):
            print(f"\n处理 {i}/{len(image_files)}: {img_file.name}")
            
            # 查找对应的mask
            mask_file = mask_dir / f"{img_file.stem}_mask.png"
            if not mask_file.exists():
                mask_file = mask_dir / f"{img_file.stem}_mask{img_file.suffix}"
            
            if not mask_file.exists():
                print(f"跳过: 未找到mask")
                continue
            
            try:
                # 加载图像
                image = Image.open(img_file).convert('RGB')
                mask = Image.open(mask_file).convert('L')
                
                # 调整尺寸
                w, h = image.size
                w = (w // 8) * 8
                h = (h // 8) * 8
                image = image.resize((w, h), Image.Resampling.LANCZOS)
                mask = mask.resize((w, h), Image.Resampling.LANCZOS)
                
                # 生成
                start_time = time.time()
                prompt = "A female model with V-shaped tea break collar, elegant knit fabric, professional photography"
                
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    result = pipe(
                        image=image,
                        mask_image=mask,
                        prompt=prompt,
                        num_inference_steps=28,
                        guidance_scale=4.0,
                        strength=0.9,
                        generator=torch.Generator("cuda").manual_seed(42)
                    ).images[0]
                
                gen_time = time.time() - start_time
                total_time += gen_time
                
                # 保存
                output_path = output_dir / f"{img_file.stem}_lora_tea_break.jpg"
                result.save(output_path, quality=95)
                
                successful += 1
                print(f"成功: {output_path.name} ({gen_time:.1f}s)")
                
            except Exception as e:
                print(f"失败: {e}")
        
        print(f"\n批量处理完成:")
        print(f"成功: {successful}/{len(image_files)}")
        if successful > 0:
            print(f"平均耗时: {total_time/successful:.1f}秒")
        
        return successful > 0
        
    except Exception as e:
        print(f"批量推理失败: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--batch":
        success = batch_inference()
    else:
        success = main()
    
    sys.exit(0 if success else 1)