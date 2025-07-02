#!/usr/bin/env python3
# run_with_logs.py - 带详细日志的运行器
import sys
import time
import torch
from pathlib import Path

# 导入详细日志系统
from detailed_logger import DetailedLogger, log_step, log_gpu_memory, log_exception, timing

class LoggedNecklineChanger:
    """带详细日志的茶歇领替换器"""
    
    def __init__(self):
        # 初始化日志系统
        self.logger = DetailedLogger()
        
        log_step("初始化开始")
        log_gpu_memory("初始化前")
        
        try:
            self._setup_environment()
            self._load_model()
            log_step("初始化完成")
        except Exception as e:
            log_exception(e, "初始化失败")
            raise
    
    @timing("环境设置")
    def _setup_environment(self):
        """设置环境"""
        log_step("设置CUDA环境")
        
        if torch.cuda.is_available():
            self.device = "cuda"
            self.logger.logger.info(f"使用GPU: {torch.cuda.get_device_name()}")
            
            # 设置GPU内存
            torch.cuda.set_per_process_memory_fraction(0.95)
            
            # 启用优化
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            
            log_gpu_memory("环境设置后")
        else:
            self.device = "cpu"
            self.logger.logger.warning("CUDA不可用，使用CPU")
    
    @timing("模型加载")
    def _load_model(self):
        """加载模型"""
        log_step("开始加载FLUX模型")
        
        try:
            from diffusers import FluxFillPipeline
            
            model_path = "models/flux-fill"
            
            # 检查模型路径
            if not Path(model_path).exists():
                raise FileNotFoundError(f"模型路径不存在: {model_path}")
            
            log_step("模型路径检查通过", model_path)
            
            # 加载参数
            load_kwargs = {
                "torch_dtype": torch.bfloat16 if self.device == "cuda" else torch.float32,
                "use_safetensors": True,
            }
            
            self.logger.logger.info(f"加载参数: {load_kwargs}")
            
            # 记录加载前内存
            log_gpu_memory("模型加载前")
            self.logger.log_process_info("模型加载前")
            
            # 加载模型
            log_step("正在加载FluxFillPipeline")
            self.pipe = FluxFillPipeline.from_pretrained(model_path, **load_kwargs)
            
            log_gpu_memory("模型加载后")
            
            # 移动到设备
            log_step("移动模型到GPU")
            self.pipe = self.pipe.to(self.device)
            
            log_gpu_memory("移动到GPU后")
            
            # 应用优化
            self._apply_optimizations()
            
            log_step("模型加载完成")
            
        except Exception as e:
            log_exception(e, "模型加载")
            raise
    
    @timing("模型优化")
    def _apply_optimizations(self):
        """应用模型优化"""
        log_step("应用内存优化")
        
        try:
            if self.device == "cuda":
                # 对于RTX 4090，可以禁用一些内存节省功能
                if hasattr(self.pipe, 'disable_attention_slicing'):
                    self.pipe.disable_attention_slicing()
                    self.logger.logger.info("禁用注意力切片 (大显存模式)")
                
                if hasattr(self.pipe, 'disable_vae_slicing'):
                    self.pipe.disable_vae_slicing()
                    self.logger.logger.info("禁用VAE切片 (大显存模式)")
            
            log_gpu_memory("优化后")
            
        except Exception as e:
            log_exception(e, "模型优化")
            # 优化失败不是致命错误，继续运行
    
    @timing("图像预处理")
    def _preprocess_images(self, image_path, mask_path):
        """预处理图像"""
        log_step("加载和预处理图像")
        
        try:
            from PIL import Image
            
            # 加载图像
            self.logger.logger.info(f"加载图像: {image_path}")
            image = Image.open(image_path).convert('RGB')
            
            self.logger.logger.info(f"加载mask: {mask_path}")
            mask = Image.open(mask_path).convert('L')
            
            original_size = image.size
            self.logger.logger.info(f"原始尺寸: {original_size}")
            
            # 调整尺寸
            max_size = 1024  # 可以根据GPU调整
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = (int(image.width * ratio), int(image.height * ratio))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
                mask = mask.resize(new_size, Image.Resampling.LANCZOS)
                self.logger.logger.info(f"调整尺寸: {original_size} -> {new_size}")
            
            # 确保尺寸是8的倍数
            w, h = image.size
            w = (w // 8) * 8
            h = (h // 8) * 8
            if (w, h) != image.size:
                image = image.resize((w, h), Image.Resampling.LANCZOS)
                mask = mask.resize((w, h), Image.Resampling.LANCZOS)
                self.logger.logger.info(f"对齐到8的倍数: {image.size}")
            
            log_step("图像预处理完成", f"最终尺寸: {image.size}")
            return image, mask
            
        except Exception as e:
            log_exception(e, "图像预处理")
            raise
    
    @timing("图像生成")
    def _generate_image(self, image, mask, prompt):
        """生成图像"""
        log_step("开始图像生成")
        log_gpu_memory("生成前")
        
        try:
            # 生成参数
            gen_params = {
                "num_inference_steps": 28,
                "guidance_scale": 3.5,
                "strength": 0.85,
            }
            
            self.logger.logger.info(f"生成参数: {gen_params}")
            self.logger.logger.info(f"提示词: {prompt}")
            
            # 负向提示词
            negative_prompt = "low quality, blurry, artifacts, deformed, different style, different fabric, different color"
            
            # 设置随机种子
            generator = torch.Generator(self.device).manual_seed(42)
            
            # 开始生成
            with torch.autocast(self.device, dtype=torch.bfloat16 if self.device == "cuda" else torch.float32):
                log_step("调用pipe生成")
                result = self.pipe(
                    image=image,
                    mask_image=mask,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    generator=generator,
                    **gen_params
                ).images[0]
            
            log_gpu_memory("生成后")
            log_step("图像生成完成")
            
            return result
            
        except Exception as e:
            log_exception(e, "图像生成")
            log_gpu_memory("生成失败时")
            raise
    
    @timing("结果保存")
    def _save_result(self, result, output_path):
        """保存结果"""
        log_step("保存结果")
        
        try:
            Path(output_path).parent.mkdir(exist_ok=True)
            result.save(output_path, quality=95)
            
            # 检查文件是否成功创建
            if Path(output_path).exists():
                file_size = Path(output_path).stat().st_size
                self.logger.logger.info(f"文件保存成功: {output_path} ({file_size/1024/1024:.2f}MB)")
                return True
            else:
                self.logger.logger.error("文件保存失败: 文件未创建")
                return False
                
        except Exception as e:
            log_exception(e, "结果保存")
            return False
    
    def process_image(self, image_path, mask_path, prompt, output_path=None):
        """处理图像的主函数"""
        log_step("开始处理图像", f"输入: {Path(image_path).name}")
        
        start_time = time.time()
        
        try:
            # 检查输入文件
            if not Path(image_path).exists():
                raise FileNotFoundError(f"图像文件不存在: {image_path}")
            
            if not Path(mask_path).exists():
                raise FileNotFoundError(f"Mask文件不存在: {mask_path}")
            
            log_step("输入文件检查通过")
            
            # 预处理图像
            image, mask = self._preprocess_images(image_path, mask_path)
            
            # 生成图像
            result = self._generate_image(image, mask, prompt)
            
            # 保存结果
            if output_path is None:
                output_path = f"output_images/{Path(image_path).stem}_logged_result.jpg"
            
            success = self._save_result(result, output_path)
            
            # 计算总时间
            total_time = time.time() - start_time
            
            if success:
                log_step("处理完成", f"总耗时: {total_time:.2f}秒")
                self.logger.logger.info(f"成功输出: {output_path}")
                return True, output_path, f"{total_time:.2f}s"
            else:
                log_step("处理失败", "保存文件失败")
                return False, "保存失败", None
            
        except Exception as e:
            total_time = time.time() - start_time
            log_exception(e, f"图像处理 (耗时: {total_time:.2f}秒)")
            return False, str(e), None
        
        finally:
            # 清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                log_gpu_memory("处理结束后")

def main():
    """主函数"""
    try:
        print("=== 带详细日志的FLUX茶歇领替换工具 ===")
        
        # 创建处理器
        changer = LoggedNecklineChanger()
        
        # 处理图像
        image_path = "input_images/051.png"
        mask_path = "masks/051_mask.png"
        prompt = """A female model stands with hands in pockets, she wears a white ribbed short-sleeve knit top with elegant sweetheart neckline, fitted silhouette, and the hem of top is tucked into the pants waistband. She pairs it with high-waisted white trousers. The background is beige. Same white ribbed knit fabric and texture as original top, professional photography, high quality."""
        
        log_step("开始主处理流程")
        
        success, result, time_str = changer.process_image(
            image_path=image_path,
            mask_path=mask_path,
            prompt=prompt
        )
        
        if success:
            print(f"\n处理成功!")
            print(f"输出: {result}")
            print(f"耗时: {time_str}")
        else:
            print(f"\n处理失败: {result}")
        
        # 生成调试报告
        log_step("生成调试报告")
        report_file = changer.logger.create_debug_report()
        
        print(f"\n调试报告已生成: {report_file}")
        print(f"  - {changer.logger.log_file}")
        print(f"  - {changer.logger.json_file}")
        print(f"  - {report_file}")
        
        return success
        
    except KeyboardInterrupt:
        print("\n用户中断")
        return False
    except Exception as e:
        print(f"\n程序异常: {e}")
        
        # 如果有日志系统，记录异常
        try:
            log_exception(e, "主程序")
        except:
            pass
        
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    
    print(f"\n日志文件位置: debug_logs/")
    
    sys.exit(0 if success else 1)