#!/usr/bin/env python3
# run_rtx4090.py - RTX 4090专用启动器
import sys
import torch
from pathlib import Path
import time

# 导入RTX 4090优化配置
from rtx4090_config import (
    setup_rtx4090_environment, 
    apply_rtx4090_optimizations,
    RTX4090ModelConfig, 
    RTX4090GenerationConfig,
    RTX4090ImageConfig
)

class RTX4090NecklineChanger:
    """RTX 4090优化版本的茶歇领替换器"""
    
    def __init__(self):
        print("初始化RTX 4090专用版本...")
        
        # 设置优化环境
        setup_rtx4090_environment()
        
        # 强制使用CUDA
        self.device = "cuda"
        
        # 加载模型
        self._load_optimized_model()
        
        print("RTX 4090初始化完成")
    
    def _load_optimized_model(self):
        """加载并优化模型"""
        try:
            from diffusers import FluxFillPipeline
            
            model_path = "models/flux-fill"
            print(f"加载模型: {model_path}")
            
            # RTX 4090优化的加载参数
            load_kwargs = {
                "torch_dtype": torch.bfloat16,  # 使用bfloat16获得最佳性能
                "use_safetensors": True,
                "device_map": "auto",
            }
            
            # 加载管道
            print("正在加载FLUX-Fill模型...")
            self.pipe = FluxFillPipeline.from_pretrained(model_path, **load_kwargs)
            
            # 移动到GPU
            self.pipe = self.pipe.to(self.device)
            
            # 应用RTX 4090特定优化
            apply_rtx4090_optimizations(self.pipe)
            
            print("模型加载和优化完成")
            
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise
    
    def process_image(self, image_path: str, mask_path: str, 
                     prompt: str, output_path: str = None):
        """处理图像"""
        try:
            from PIL import Image
            import numpy as np
            
            print(f"处理图像: {Path(image_path).name}")
            
            # 加载图像
            image = Image.open(image_path).convert('RGB')
            mask = Image.open(mask_path).convert('L')
            
            print(f"原始尺寸: {image.size}")
            
            # RTX 4090可以处理更大尺寸
            max_size = RTX4090ImageConfig.MAX_IMAGE_SIZE[0]
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = (int(image.width * ratio), int(image.height * ratio))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
                mask = mask.resize(new_size, Image.Resampling.LANCZOS)
                print(f"调整尺寸: {new_size}")
            
            # 确保尺寸是8的倍数
            w, h = image.size
            w = (w // 8) * 8
            h = (h // 8) * 8
            if (w, h) != image.size:
                image = image.resize((w, h), Image.Resampling.LANCZOS)
                mask = mask.resize((w, h), Image.Resampling.LANCZOS)
            
            print(f"最终尺寸: {image.size}")
            
            # 开始生成
            print("开始生成...")
            start_time = time.time()
            
            # 使用RTX 4090优化参数
            gen_config = RTX4090GenerationConfig.DEFAULT_PARAMS
            
            # 负向提示词
            negative_prompt = "low quality, blurry, artifacts, deformed, different style, different fabric, different color"
            
            with torch.autocast("cuda", dtype=torch.bfloat16):
                result = self.pipe(
                    image=image,
                    mask_image=mask,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=gen_config["num_inference_steps"],
                    guidance_scale=gen_config["guidance_scale"],
                    strength=gen_config["strength"],
                    generator=torch.Generator("cuda").manual_seed(gen_config["seed"])
                ).images[0]
            
            # 处理时间
            processing_time = time.time() - start_time
            print(f" 生成完成，耗时: {processing_time:.2f}秒")
            
            # 保存结果
            if output_path is None:
                output_path = f"output_images/{Path(image_path).stem}_rtx4090_result.jpg"
            
            Path(output_path).parent.mkdir(exist_ok=True)
            result.save(output_path, quality=RTX4090ImageConfig.OUTPUT_QUALITY)
            
            print(f"结果已保存: {output_path}")
            return True, output_path, f"{processing_time:.2f}s"
            
        except Exception as e:
            print(f"处理失败: {e}")
            import traceback
            traceback.print_exc()
            return False, str(e), None
    
    def clear_memory(self):
        """清理显存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

def main():
    """主函数"""
    print("=== RTX 4090 FLUX茶歇领替换工具 ===")
    
    # 检查GPU
    if not torch.cuda.is_available():
        print("未检测到CUDA GPU")
        return False
    
    gpu_name = torch.cuda.get_device_name()
    if "4090" not in gpu_name:
        print(f"检测到GPU: {gpu_name}")
        print("此工具专为RTX 4090优化，其他GPU可能效果不佳")
        response = input("是否继续? (y/n): ")
        if response.lower() != 'y':
            return False
    
    try:
        # 初始化处理器
        changer = RTX4090NecklineChanger()
        
        # 处理你的图片
        image_path = "input_images/051.png"
        mask_path = "masks/051_mask.png"
        
        # 检查文件是否存在
        if not Path(image_path).exists():
            print(f"图片不存在: {image_path}")
            return False
        
        if not Path(mask_path).exists():
            print(f"Mask不存在: {mask_path}")
            return False
        
        # RTX 4090优化的提示词
        prompt = """A female model stands with hands in pockets, she wears a white ribbed short-sleeve knit top with elegant sweetheart neckline, fitted silhouette, and the hem of top is tucked into the pants waistband. She pairs it with high-waisted white trousers. The background is beige. Same white ribbed knit fabric and texture as original top, professional photography, high quality, detailed."""
        
        print("\n开始RTX 4090优化处理...")
        
        success, result, time_str = changer.process_image(
            image_path=image_path,
            mask_path=mask_path,
            prompt=prompt
        )
        
        if success:
            print(f"\n处理成功!")
            print(f"输出文件: {result}")
            print(f"处理时间: {time_str}")
            
            # 显示显存使用情况
            if torch.cuda.is_available():
                memory_used = torch.cuda.max_memory_allocated() / 1e9
                memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"显存使用: {memory_used:.1f}GB / {memory_total:.1f}GB ({memory_used/memory_total*100:.1f}%)")
        else:
            print(f"\n处理失败: {result}")
        
        # 清理显存
        changer.clear_memory()
        return success
        
    except KeyboardInterrupt:
        print("\n 用户中断")
        return False
    except Exception as e:
        print(f"\n程序异常: {e}")
        import traceback
        traceback.print_exc()
        return False

def batch_process():
    """批量处理模式"""
    print("=== RTX 4090 批量处理模式 ===")
    
    try:
        changer = RTX4090NecklineChanger()
        
        input_dir = Path("input_images")
        mask_dir = Path("masks")
        output_dir = Path("output_images")
        
        # 获取所有图片文件
        image_files = []
        for ext in ['.png', '.jpg', '.jpeg']:
            image_files.extend(input_dir.glob(f"*{ext}"))
        
        if not image_files:
            print("未找到图片文件")
            return False
        
        print(f"找到 {len(image_files)} 张图片")
        
        successful = 0
        total_time = 0
        
        for i, img_file in enumerate(image_files, 1):
            print(f"\n--- 处理 {i}/{len(image_files)}: {img_file.name} ---")
            
            # 查找对应mask
            mask_file = mask_dir / f"{img_file.stem}_mask.png"
            if not mask_file.exists():
                mask_file = mask_dir / f"{img_file.stem}_mask{img_file.suffix}"
            
            if not mask_file.exists():
                print(f" 跳过: 未找到mask文件")
                continue
            
            # 生成提示词（可以根据文件名或其他信息自定义）
            prompt = f"elegant sweetheart neckline, same fabric and style as original garment, high quality"
            
            # 处理
            success, result, time_str = changer.process_image(
                image_path=str(img_file),
                mask_path=str(mask_file),
                prompt=prompt,
                output_path=str(output_dir / f"{img_file.stem}_rtx4090_sweetheart.jpg")
            )
            
            if success:
                successful += 1
                if time_str:
                    total_time += float(time_str.replace('s', ''))
                print(f"成功: {result}")
            else:
                print(f"失败: {result}")
            
            # 清理显存
            changer.clear_memory()
        
        print(f"\n=== 批量处理完成 ===")
        print(f"成功: {successful}/{len(image_files)}")
        if successful > 0:
            print(f"平均耗时: {total_time/successful:.2f}秒")
        
        return successful > 0
        
    except Exception as e:
        print(f"批量处理失败: {e}")
        return False

if __name__ == "__main__":
    # 选择模式
    if len(sys.argv) > 1 and sys.argv[1] == "--batch":
        success = batch_process()
    else:
        success = main()
    
    sys.exit(0 if success else 1)