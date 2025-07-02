# neckline_changer.py
import torch
from diffusers import FluxFillPipeline
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
import cv2
import os
from pathlib import Path
import logging
from typing import Optional, Tuple, Dict, List
import json
import time
from tqdm import tqdm

# 导入配置
from config import (
    PathConfig, ModelConfig, GenerationConfig, ImageProcessingConfig,
    PromptConfig, LoggingConfig, PerformanceConfig, ValidationConfig,
    AppConfig, ConfigManager, check_environment
)

# 配置日志
def setup_logging():

    PathConfig.LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = PathConfig.LOG_DIR / LoggingConfig.LOG_FILE
    
    # 配置日志处理器
    handlers = []
    
    # 文件处理器
    try:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        handlers.append(file_handler)
    except Exception as e:
        print(f"警告: 无法创建日志文件 {log_file}: {e}")
    
    # 控制台处理器
    if LoggingConfig.CONSOLE_LOGGING:
        console_handler = logging.StreamHandler()
        handlers.append(console_handler)
    
    # 如果没有任何处理器，至少添加控制台输出
    if not handlers:
        handlers.append(logging.StreamHandler())
    
    logging.basicConfig(
        level=getattr(logging, LoggingConfig.LOG_LEVEL),
        format=LoggingConfig.LOG_FORMAT,
        datefmt=LoggingConfig.DATE_FORMAT,
        handlers=handlers
    )
    
    return logging.getLogger(__name__)

class NecklineChanger:
    def __init__(self, model_path: Optional[str] = None, config_override: Optional[Dict] = None):
        """
        初始化茶歇领替换工具
        
        Args:
            model_path: 模型路径，None则使用配置文件中的默认值
            config_override: 配置覆盖字典
        """

        self._ensure_directories()
        
        self.logger = setup_logging()
        self.logger.info(f"初始化 {AppConfig.APP_NAME} v{AppConfig.VERSION}")
        
        # 加载配置
        self._load_configuration(config_override)
        
        # 设置设备
        self.device = self._setup_device()
        
        # 加载模型
        model_id = model_path or ModelConfig.FLUX_MODEL_ID
        self._load_model(model_id)
        
        # 初始化统计
        self.stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "processing_time": 0.0
        }
        
        self.logger.info("初始化完成")
    
    def _ensure_directories(self):

        try:
            PathConfig.create_directories()
        except Exception as e:
            print(f"警告: 创建目录时出现问题: {e}")
            # 尝试创建最基本的目录
            try:
                Path("logs").mkdir(exist_ok=True)
                Path("output_images").mkdir(exist_ok=True)
            except Exception as inner_e:
                print(f"无法创建基本目录: {inner_e}")
    
    def _load_configuration(self, config_override: Optional[Dict] = None):
        """加载配置"""
        self.logger.info("加载配置...")
        
        # 加载用户自定义配置
        user_config = ConfigManager.load_user_config()
        
        # 合并配置
        self.config = {
            "generation": GenerationConfig.DEFAULT_PARAMS.copy(),
            "image_processing": {
                "max_size": ImageProcessingConfig.MAX_IMAGE_SIZE,
                "output_quality": ImageProcessingConfig.OUTPUT_QUALITY,
                "mask_blur": ImageProcessingConfig.MASK_BLUR_RADIUS,
                "output_format": ImageProcessingConfig.OUTPUT_FORMAT,
            },
            "paths": {
                "input_dir": PathConfig.INPUT_DIR,
                "mask_dir": PathConfig.MASK_DIR,
                "output_dir": PathConfig.OUTPUT_DIR,
            }
        }
        
        # 应用用户配置
        if user_config:
            self._merge_config(self.config, user_config)
        
        # 应用运行时覆盖
        if config_override:
            self._merge_config(self.config, config_override)
        
        self.logger.info("配置加载完成")
    
    def _merge_config(self, base_config: Dict, override_config: Dict):
        """递归合并配置"""
        for key, value in override_config.items():
            if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
                self._merge_config(base_config[key], value)
            else:
                base_config[key] = value
    
    def _setup_device(self) -> str:
        """设置计算设备"""
        if ModelConfig.DEVICE == "cuda" and torch.cuda.is_available():
            device = "cuda"
            self.logger.info(f"使用GPU: {torch.cuda.get_device_name()}")
            
            # 设置GPU内存
            if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                torch.cuda.set_per_process_memory_fraction(PerformanceConfig.GPU_MEMORY_FRACTION)
        else:
            device = "cpu"
            self.logger.warning("使用CPU")
        
        return device
    
    def _load_model(self, model_path: str):
        """加载FLUX模型"""
        try:
            # 检查是否为本地路径
            if Path(model_path).exists():
                self.logger.info(f"使用本地模型: {model_path}")
                local_model = True
            else:
                self.logger.info(f"使用在线模型: {model_path}")
                local_model = False
            
            self.logger.info(f"加载模型: {model_path}")
            
            # 模型加载参数
            load_kwargs = {
                "torch_dtype": torch.bfloat16 if self.device == "cuda" and ModelConfig.USE_FP16 else torch.float32,
                "use_safetensors": True,
            }
            
            # 只对在线模型使用variant参数
            if not local_model and self.device == "cuda" and ModelConfig.USE_FP16:
                load_kwargs["variant"] = ModelConfig.MODEL_VARIANT
            
            # 加载管道
            self.pipe = FluxFillPipeline.from_pretrained(model_path, **load_kwargs)
            self.pipe = self.pipe.to(self.device)
            
            # 内存优化
            if self.device == "cuda":
                if ModelConfig.USE_CPU_OFFLOAD:
                    self.pipe.enable_model_cpu_offload()
                if ModelConfig.USE_ATTENTION_SLICING:
                    self.pipe.enable_attention_slicing()
            
            self.logger.info("模型加载成功")
            
        except Exception as e:
            self.logger.error(f"模型加载失败: {str(e)}")
            
            # 如果是本地模型失败，尝试在线模型
            if Path(model_path).exists():
                self.logger.info("本地模型加载失败，尝试在线模型")
                try:
                    # 导入配置，避免变量作用域问题
                    from config import ModelConfig as MC
                    self._load_model(MC.FLUX_MODEL_ID_ONLINE)
                    return
                except Exception as online_e:
                    self.logger.error(f"在线模型也失败: {str(online_e)}")
            
            # 尝试备用模型
            for backup_model in ModelConfig.BACKUP_MODELS:
                try:
                    self.logger.info(f"尝试备用模型: {backup_model}")
                    self._load_model(backup_model)
                    return
                except Exception as backup_e:
                    self.logger.warning(f"备用模型 {backup_model} 也失败: {str(backup_e)}")
            
            raise RuntimeError("所有模型都加载失败")
    
    def validate_inputs(self, image_path: str, mask_path: str) -> bool:
        """验证输入文件"""
        try:
            # 检查文件存在
            if not Path(image_path).exists():
                self.logger.error(f"图片文件不存在: {image_path}")
                return False
            
            if not Path(mask_path).exists():
                self.logger.error(f"Mask文件不存在: {mask_path}")
                return False
            
            # 检查文件大小
            if ValidationConfig.MAX_FILE_SIZE:
                for file_path in [image_path, mask_path]:
                    file_size = Path(file_path).stat().st_size
                    if file_size > ValidationConfig.MAX_FILE_SIZE:
                        self.logger.error(f"文件过大: {file_path}")
                        return False
            
            # 检查图片完整性
            if ValidationConfig.CHECK_IMAGE_INTEGRITY:
                try:
                    with Image.open(image_path) as img:
                        img.verify()
                    with Image.open(mask_path) as mask:
                        mask.verify()
                except Exception as e:
                    self.logger.error(f"图片完整性检查失败: {e}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"输入验证失败: {e}")
            return False
    
    def preprocess_image(self, image_path: str) -> Image.Image:
        """预处理图片"""
        image = Image.open(image_path).convert('RGB')
        
        # 尺寸调整
        w, h = image.size
        max_w, max_h = self.config["image_processing"]["max_size"]
        
        # 如果图片过大，等比缩放
        if w > max_w or h > max_h:
            ratio = min(max_w / w, max_h / h)
            new_w, new_h = int(w * ratio), int(h * ratio)
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            self.logger.info(f"图片已缩放: {w}x{h} -> {new_w}x{new_h}")
        
        # 确保尺寸是8的倍数
        w, h = image.size
        new_w = (w // ImageProcessingConfig.SIZE_MULTIPLE) * ImageProcessingConfig.SIZE_MULTIPLE
        new_h = (h // ImageProcessingConfig.SIZE_MULTIPLE) * ImageProcessingConfig.SIZE_MULTIPLE
        
        if new_w != w or new_h != h:
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        return image
    
    def preprocess_mask(self, mask_path: str, target_size: Tuple[int, int]) -> Image.Image:
        """预处理mask"""
        mask = Image.open(mask_path).convert('L')
        
        # 调整尺寸
        mask = mask.resize(target_size, Image.Resampling.LANCZOS)
        
        # 应用模糊
        blur_radius = self.config["image_processing"]["mask_blur"]
        if blur_radius > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        
        # 增强对比度
        enhancer = ImageEnhance.Contrast(mask)
        mask = enhancer.enhance(ImageProcessingConfig.MASK_CONTRAST_ENHANCE)
        
        # 验证mask面积
        if ValidationConfig.MIN_MASK_AREA or ValidationConfig.MAX_MASK_AREA_RATIO:
            mask_array = np.array(mask)
            mask_area = np.sum(mask_array > 128)
            total_area = mask_array.size
            
            if mask_area < ValidationConfig.MIN_MASK_AREA:
                self.logger.warning(f"Mask面积过小: {mask_area} < {ValidationConfig.MIN_MASK_AREA}")
            
            if mask_area / total_area > ValidationConfig.MAX_MASK_AREA_RATIO:
                self.logger.warning(f"Mask面积过大: {mask_area/total_area:.2%} > {ValidationConfig.MAX_MASK_AREA_RATIO:.2%}")
        
        return mask
    
    def analyze_style(self, image: Image.Image, mask: Image.Image) -> Dict[str, str]:
        """分析图片风格"""
        try:
            image_array = np.array(image)
            mask_array = np.array(mask)
            
            # 提取mask区域
            mask_binary = mask_array > 128
            if not np.any(mask_binary):
                return {"color": "neutral", "brightness": "medium", "material": "fabric"}
            
            masked_pixels = image_array[mask_binary]
            
            # 颜色分析
            avg_color = np.mean(masked_pixels, axis=0)
            brightness = np.mean(avg_color)
            
            # 亮度分类
            if brightness < 85:
                brightness_desc = "dark"
            elif brightness > 170:
                brightness_desc = "light"
            else:
                brightness_desc = "medium"
            
            # 色调分析
            r, g, b = avg_color
            color_diff_threshold = 25
            
            if r > g + color_diff_threshold and r > b + color_diff_threshold:
                color_desc = "warm red"
            elif g > r + color_diff_threshold and g > b + color_diff_threshold:
                color_desc = "cool green"
            elif b > r + color_diff_threshold and b > g + color_diff_threshold:
                color_desc = "cool blue"
            elif r > 150 and g > 150 and b > 150:
                color_desc = "light neutral"
            elif r < 80 and g < 80 and b < 80:
                color_desc = "dark neutral"
            else:
                color_desc = "neutral"
            
            # 纹理分析
            variance = np.var(masked_pixels)
            if variance < 300:
                material_desc = "smooth"
            elif variance > 1200:
                material_desc = "textured"
            else:
                material_desc = "moderate"
            
            return {
                "color": color_desc,
                "brightness": brightness_desc,
                "material": material_desc
            }
            
        except Exception as e:
            self.logger.warning(f"风格分析失败: {e}")
            return {"color": "neutral", "brightness": "medium", "material": "fabric"}
    
    def build_prompt(self, style_info: Dict[str, str], custom_prompt: Optional[str] = None) -> Tuple[str, str]:
        """构建提示词"""
        if custom_prompt:
            positive_prompt = custom_prompt
        else:
            # 使用模板构建
            style_desc = f"{style_info['brightness']} {style_info['color']} {style_info['material']} fabric"
            positive_prompt = PromptConfig.BASE_PROMPT_TEMPLATE.format(style_description=style_desc)
        
        # 选择负向提示词
        negative_prompt = PromptConfig.ENHANCED_NEGATIVE_PROMPT if style_info.get('difficult') else PromptConfig.NEGATIVE_PROMPT
        
        return positive_prompt, negative_prompt
    
    def generate_image(self, image: Image.Image, mask: Image.Image, 
                      positive_prompt: str, negative_prompt: str,
                      generation_config: Optional[Dict] = None) -> Image.Image:
        """生成图像"""
        # 合并生成配置
        gen_config = self.config["generation"].copy()
        if generation_config:
            gen_config.update(generation_config)
        
        # 设置随机种子
        generator = torch.manual_seed(gen_config["seed"]) if gen_config.get("seed") else None
        
        self.logger.info(f"开始生成 - 步数: {gen_config['num_inference_steps']}, 引导: {gen_config['guidance_scale']}")
        
        with torch.autocast(self.device):
            result = self.pipe(
                image=image,
                mask_image=mask,
                prompt=positive_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=gen_config["num_inference_steps"],
                guidance_scale=gen_config["guidance_scale"],
                strength=gen_config["strength"],
                generator=generator
            ).images[0]
        
        return result
    
    def save_result(self, image: Image.Image, output_path: str, 
                   backup_original: bool = None) -> str:
        """保存结果"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存配置
        save_kwargs = {
            "quality": self.config["image_processing"]["output_quality"],
            "optimize": ImageProcessingConfig.OPTIMIZE_OUTPUT
        }
        
        # 根据格式调整参数
        if output_path.suffix.lower() in ['.jpg', '.jpeg']:
            save_kwargs["format"] = "JPEG"
        elif output_path.suffix.lower() == '.png':
            save_kwargs["format"] = "PNG"
        
        image.save(output_path, **save_kwargs)
        
        return str(output_path)
    
    def process_single_image(self, image_path: str, mask_path: str,
                           output_path: Optional[str] = None,
                           custom_prompt: Optional[str] = None,
                           generation_config: Optional[Dict] = None) -> Tuple[bool, str, Optional[str]]:
        """
        处理单张图片
        
        Returns:
            (是否成功, 输出路径或错误信息, 处理时间)
        """
        start_time = time.time()
        
        try:
            # 验证输入
            if not self.validate_inputs(image_path, mask_path):
                return False, "输入验证失败", None
            
            self.logger.info(f"开始处理: {Path(image_path).name}")
            
            # 预处理
            image = self.preprocess_image(image_path)
            mask = self.preprocess_mask(mask_path, image.size)
            
            # 风格分析
            style_info = self.analyze_style(image, mask)
            self.logger.info(f"检测风格: {style_info}")
            
            # 构建提示词
            positive_prompt, negative_prompt = self.build_prompt(style_info, custom_prompt)
            self.logger.info(f"提示词: {positive_prompt}")
            
            # 生成图像
            result = self.generate_image(image, mask, positive_prompt, negative_prompt, generation_config)
            
            # 保存结果
            if output_path is None:
                input_name = Path(image_path).stem
                output_path = self.config["paths"]["output_dir"] / f"{input_name}_sweetheart.jpg"
            
            final_path = self.save_result(result, output_path)
            
            # 更新统计
            processing_time = time.time() - start_time
            self.stats["total_processed"] += 1
            self.stats["successful"] += 1
            self.stats["processing_time"] += processing_time
            
            self.logger.info(f"处理完成: {final_path} (耗时: {processing_time:.2f}s)")
            
            return True, final_path, f"{processing_time:.2f}s"
            
        except Exception as e:
            error_msg = f"处理失败: {str(e)}"
            self.logger.error(error_msg)
            
            self.stats["total_processed"] += 1
            self.stats["failed"] += 1
            
            return False, error_msg, None
    
    def batch_process(self, input_dir: Optional[str] = None,
                     mask_dir: Optional[str] = None,
                     output_dir: Optional[str] = None,
                     generation_config: Optional[Dict] = None) -> Dict:
        """批量处理"""
        # 使用配置中的路径
        input_path = Path(input_dir) if input_dir else self.config["paths"]["input_dir"]
        mask_path = Path(mask_dir) if mask_dir else self.config["paths"]["mask_dir"]
        output_path = Path(output_dir) if output_dir else self.config["paths"]["output_dir"]
        
        self.logger.info(f"开始批量处理: {input_path}")
        
        # 获取图片文件
        image_files = []
        for ext in ImageProcessingConfig.SUPPORTED_FORMATS:
            image_files.extend(input_path.glob(f"*{ext}"))
        
        if not image_files:
            self.logger.warning("未找到图片文件")
            return {"total": 0, "successful": 0, "failed": 0}
        
        self.logger.info(f"找到 {len(image_files)} 张图片")
        
        # 处理进度条
        results = []
        with tqdm(image_files, desc="处理进度") as pbar:
            for img_file in pbar:
                pbar.set_description(f"处理: {img_file.name}")
                
                # 查找对应的mask
                mask_candidates = [
                    mask_path / f"{img_file.stem}_mask{img_file.suffix}",
                    mask_path / f"{img_file.stem}_mask.png",
                    mask_path / f"{img_file.stem}.png"
                ]
                
                mask_file = None
                for candidate in mask_candidates:
                    if candidate.exists():
                        mask_file = candidate
                        break
                
                if mask_file is None:
                    self.logger.warning(f"未找到mask: {img_file.name}")
                    results.append((False, f"未找到mask: {img_file.name}"))
                    continue
                
                # 设置输出路径
                output_file = output_path / f"{img_file.stem}_sweetheart.jpg"
                
                # 处理图片
                success, message, time_str = self.process_single_image(
                    str(img_file), str(mask_file), str(output_file), 
                    generation_config=generation_config
                )
                
                results.append((success, message))
                
                # 清理缓存
                if self.stats["total_processed"] % PerformanceConfig.CLEAR_CACHE_INTERVAL == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        
        # 统计结果
        successful = sum(1 for success, _ in results if success)
        failed = len(results) - successful
        
        batch_stats = {
            "total": len(results),
            "successful": successful,
            "failed": failed,
            "success_rate": f"{successful/len(results)*100:.1f}%" if results else "0%",
            "avg_time": f"{self.stats['processing_time']/max(1, self.stats['total_processed']):.2f}s"
        }
        
        self.logger.info(f"批量处理完成: {batch_stats}")
        return batch_stats

def main():
    """主函数"""
    print(f"=== {AppConfig.APP_NAME} v{AppConfig.VERSION} ===")
    
    # 环境检查
    check_environment()
    
    try:
        # 初始化处理器
        print("正在初始化...")
        changer = NecklineChanger()
        
        # 交互式菜单
        while True:
            print("\n=== 操作菜单 ===")
            print("1. 处理单张图片")
            print("2. 批量处理")
            print("3. 使用高质量模式处理")
            print("4. 查看处理统计")
            print("5. 退出")
            
            choice = input("请选择操作 (1-5): ").strip()
            
            if choice == '1':
                # 单张图片处理
                img_path = input("输入图片路径: ").strip().strip('"')
                mask_path = input("输入mask路径: ").strip().strip('"')
                
                success, result, time_str = changer.process_single_image(img_path, mask_path)
                
                if success:
                    print(f"✓ 处理成功: {result}")
                    print(f"  耗时: {time_str}")
                else:
                    print(f"✗ 处理失败: {result}")
            
            elif choice == '2':
                # 批量处理
                stats = changer.batch_process()
                print(f"批量处理完成: {stats}")
            
            elif choice == '3':
                # 高质量模式
                img_path = input("输入图片路径: ").strip().strip('"')
                mask_path = input("输入mask路径: ").strip().strip('"')
                
                success, result, time_str = changer.process_single_image(
                    img_path, mask_path,
                    generation_config=GenerationConfig.HIGH_QUALITY_PARAMS
                )
                
                if success:
                    print(f"✓ 高质量处理成功: {result}")
                    print(f"  耗时: {time_str}")
                else:
                    print(f"✗ 处理失败: {result}")
            
            elif choice == '4':
                # 查看统计
                stats = changer.stats
                print(f"处理统计:")
                print(f"  总计: {stats['total_processed']}")
                print(f"  成功: {stats['successful']}")
                print(f"  失败: {stats['failed']}")
                if stats['total_processed'] > 0:
                    print(f"  成功率: {stats['successful']/stats['total_processed']*100:.1f}%")
                    print(f"  平均耗时: {stats['processing_time']/stats['total_processed']:.2f}s")
            
            elif choice == '5':
                print("感谢使用!")
                break
            
            else:
                print("无效选择，请重试")
    
    except KeyboardInterrupt:
        print("\n用户中断操作")
    except Exception as e:
        print(f"程序出错: {e}")
        logging.exception("程序异常")

if __name__ == "__main__":
    main()