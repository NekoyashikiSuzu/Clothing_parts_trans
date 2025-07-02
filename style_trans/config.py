# config.py
import os
from pathlib import Path

class PathConfig:
    """路径相关配置"""
    
    # 项目根目录
    PROJECT_ROOT = Path(__file__).parent
    
    # 数据目录
    INPUT_DIR = PROJECT_ROOT / "input_images"
    MASK_DIR = PROJECT_ROOT / "masks"
    OUTPUT_DIR = PROJECT_ROOT / "output_images"
    
    # 模型缓存目录
    MODEL_CACHE_DIR = PROJECT_ROOT / "models"
    
    # 日志目录
    LOG_DIR = PROJECT_ROOT / "logs"
    
    # 临时文件目录
    TEMP_DIR = PROJECT_ROOT / "temp"
    
    @classmethod
    def create_directories(cls):
        """创建所有必要的目录"""
        directories = [
            cls.INPUT_DIR, cls.MASK_DIR, cls.OUTPUT_DIR,
            cls.MODEL_CACHE_DIR, cls.LOG_DIR, cls.TEMP_DIR
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

class ModelConfig:
    """模型相关配置"""
    
    # FLUX模型配置 - 使用本地路径
    FLUX_MODEL_ID = str(PathConfig.PROJECT_ROOT / "models" / "flux-fill")
    
    # 如果本地模型不存在，使用在线模型作为备用
    FLUX_MODEL_ID_ONLINE = "black-forest-labs/FLUX.1-Fill-dev"
    
    # 备用模型（如果主模型不可用）
    BACKUP_MODELS = [
        "black-forest-labs/FLUX.1-dev",
        "stabilityai/stable-diffusion-xl-base-1.0"
    ]
    
    # 模型精度配置
    USE_FP16 = True 
    USE_ATTENTION_SLICING = True 
    USE_CPU_OFFLOAD = True
    
    # 设备配置
    DEVICE = "cuda"  # 可选: "cuda", "cpu", "mps"
    
    # 模型下载配置
    USE_AUTH_TOKEN = False  # 是否需要HuggingFace认证
    AUTH_TOKEN = None  # HuggingFace访问令牌
    
    # 模型变体
    MODEL_VARIANT = "fp16"  #  "fp16", "fp32", None

class GenerationConfig:
    """图像生成参数配置"""
    
    # 默认生成参数
    DEFAULT_PARAMS = {
        "num_inference_steps": 28,      # 推理步数 (20-50)
        "guidance_scale": 3.5,          # 引导强度 (1.0-10.0)
        "strength": 0.85,               # inpainting强度 (0.1-1.0)
        "seed": 42,                     # 随机种子
        "eta": 0.0,                     # DDIM eta参数
    }
    
    # 高质量配置
    HIGH_QUALITY_PARAMS = {
        "num_inference_steps": 40,
        "guidance_scale": 4.5,
        "strength": 0.9,
        "seed": 42,
        "eta": 0.0,
    }
    
    # 快速配置
    FAST_PARAMS = {
        "num_inference_steps": 20,
        "guidance_scale": 3.0,
        "strength": 0.8,
        "seed": 42,
        "eta": 0.0,
    }
    
    # 实验性配置
    EXPERIMENTAL_PARAMS = {
        "num_inference_steps": 35,
        "guidance_scale": 5.0,
        "strength": 0.95,
        "seed": 42,
        "eta": 0.1,
    }

class ImageProcessingConfig:
    """图像处理相关配置"""
    
    # 支持的图像格式
    SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    
    # 图像尺寸配置
    MAX_IMAGE_SIZE = (1024, 1024)  # 最大处理尺寸
    MIN_IMAGE_SIZE = (512, 512)    # 最小处理尺寸
    SIZE_MULTIPLE = 8              # 尺寸必须是此数的倍数
    
    # 图像质量配置
    OUTPUT_QUALITY = 95            # 输出图片质量 (1-100)
    OUTPUT_FORMAT = "JPEG"         # 输出格式
    OPTIMIZE_OUTPUT = True         # 是否优化输出
    
    # Mask处理配置
    MASK_BLUR_RADIUS = 5          # mask模糊半径
    MASK_EROSION_KERNEL = 3       # mask腐蚀核大小
    MASK_DILATION_KERNEL = 5      # mask膨胀核大小
    MASK_CONTRAST_ENHANCE = 1.2   # mask对比度增强
    
    # 图像增强配置
    AUTO_ADJUST_BRIGHTNESS = False # 是否自动调整亮度
    AUTO_ADJUST_CONTRAST = False   # 是否自动调整对比度
    AUTO_ADJUST_SATURATION = False # 是否自动调整饱和度

class PromptConfig:
    """提示词相关配置"""
    
    # 基础提示词模板
    BASE_PROMPT_TEMPLATE = "sweetheart neckline, elegant dress, {style_description}, high quality, professional photography"
    
    # 风格描述词库
    STYLE_DESCRIPTIONS = {
        "formal": "formal evening wear, luxurious fabric",
        "casual": "casual comfortable clothing, soft fabric",
        "elegant": "elegant sophisticated design, premium material",
        "vintage": "vintage classic style, traditional fabric",
        "modern": "modern contemporary design, sleek fabric"
    }
    
    # 材质描述词库
    MATERIAL_DESCRIPTIONS = {
        "silk": "silk fabric, smooth texture, lustrous finish",
        "cotton": "cotton fabric, soft texture, natural finish",
        "satin": "satin fabric, glossy texture, elegant drape",
        "lace": "lace fabric, delicate texture, intricate pattern",
        "chiffon": "chiffon fabric, flowing texture, ethereal drape"
    }
    
    # 颜色描述词库
    COLOR_DESCRIPTIONS = {
        "warm": "warm toned colors",
        "cool": "cool toned colors", 
        "neutral": "neutral balanced colors",
        "bright": "bright vibrant colors",
        "dark": "dark sophisticated colors",
        "light": "light soft colors"
    }
    
    # 负向提示词
    NEGATIVE_PROMPT = (
        "low quality, blurry, artifacts, deformed, extra limbs, "
        "different style, different fabric, different color, "
        "mismatched neckline, unrealistic, distorted proportions"
    )
    
    # 强化提示词（用于困难案例）
    ENHANCED_NEGATIVE_PROMPT = (
        "low quality, worst quality, blurry, out of focus, artifacts, "
        "deformed, disfigured, extra limbs, missing limbs, "
        "different style, different fabric, different color, different texture, "
        "mismatched neckline, unrealistic proportions, distorted anatomy, "
        "watermark, signature, text, logo, copyright"
    )

class LoggingConfig:
    """日志配置"""
    
    # 日志级别
    LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    
    # 日志格式
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    
    # 日志文件配置
    LOG_FILE = "neckline_changer.log"
    MAX_LOG_SIZE = 10 * 1024 * 1024  # 10MB
    BACKUP_COUNT = 5
    
    # 控制台日志
    CONSOLE_LOGGING = True
    
    # 详细日志（包含更多调试信息）
    VERBOSE_LOGGING = False

class PerformanceConfig:
    
    # 批处理配置
    BATCH_SIZE = 1                # 批处理大小
    MAX_CONCURRENT_JOBS = 2       # 最大并发任务数
    
    # 内存管理
    CLEAR_CACHE_INTERVAL = 10     # 每N张图片清理一次缓存
    MAX_MEMORY_USAGE = 0.8        # 最大内存使用率
    
    # GPU配置
    GPU_MEMORY_FRACTION = 0.9     # GPU内存使用比例
    ALLOW_GROWTH = True           # 是否允许显存动态增长
    
    # 多线程配置
    NUM_WORKERS = 4               # 数据加载工作线程数
    PIN_MEMORY = True             # 是否固定内存

class ValidationConfig:

    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    
    # 图像验证
    CHECK_IMAGE_INTEGRITY = True      # 是否检查图像完整性
    REQUIRE_MASK_MATCH = True         # 是否要求mask与图像匹配
    
    # Mask验证
    MIN_MASK_AREA = 100              # 最小mask面积（像素）
    MAX_MASK_AREA_RATIO = 0.5        # 最大mask面积比例
    
    # 提示词验证
    MAX_PROMPT_LENGTH = 500          # 最大提示词长度
    MIN_PROMPT_LENGTH = 10           # 最小提示词长度

class AppConfig:
    """应用程序配置"""
    
    # 应用信息
    APP_NAME = "FLUX茶歇领替换工具"
    VERSION = "1.0.0"
    AUTHOR = "用户"
    
    # 界面配置
    SHOW_PROGRESS_BAR = True         # 是否显示进度条
    SAVE_INTERMEDIATE_RESULTS = False # 是否保存中间结果
    
    # 安全配置
    SAFE_MODE = True                 # 安全模式（额外验证）
    AUTO_BACKUP = True               # 是否自动备份原图
    
    # 更新配置
    CHECK_UPDATES = False            # 是否检查更新
    AUTO_UPDATE = False              # 是否自动更新

class ConfigManager:
    """配置管理器"""
    
    @staticmethod
    def get_config_dict():
        """获取所有配置的字典形式"""
        return {
            "paths": {
                "input_dir": str(PathConfig.INPUT_DIR),
                "mask_dir": str(PathConfig.MASK_DIR),
                "output_dir": str(PathConfig.OUTPUT_DIR),
                "model_cache_dir": str(PathConfig.MODEL_CACHE_DIR),
            },
            "model": {
                "model_id": ModelConfig.FLUX_MODEL_ID,
                "use_fp16": ModelConfig.USE_FP16,
                "device": ModelConfig.DEVICE,
            },
            "generation": GenerationConfig.DEFAULT_PARAMS,
            "image_processing": {
                "max_size": ImageProcessingConfig.MAX_IMAGE_SIZE,
                "output_quality": ImageProcessingConfig.OUTPUT_QUALITY,
                "mask_blur": ImageProcessingConfig.MASK_BLUR_RADIUS,
            },
            "performance": {
                "batch_size": PerformanceConfig.BATCH_SIZE,
                "num_workers": PerformanceConfig.NUM_WORKERS,
            }
        }
    
    @staticmethod
    def load_user_config(config_file="user_config.json"):
        """加载用户自定义配置"""
        import json
        
        config_path = PathConfig.PROJECT_ROOT / config_file
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                print(f"已加载用户配置: {config_file}")
                return user_config
            except Exception as e:
                print(f"加载用户配置失败: {e}")
        return {}
    
    @staticmethod
    def save_user_config(config_dict, config_file="user_config.json"):
        """保存用户配置"""
        import json
        
        config_path = PathConfig.PROJECT_ROOT / config_file
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            print(f"用户配置已保存: {config_file}")
        except Exception as e:
            print(f"保存用户配置失败: {e}")

def check_environment():
    """检测运行环境"""
    import torch
    import sys
    
    print("=== 环境检测 ===")
    print(f"Python版本: {sys.version}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # 检查必要目录
    PathConfig.create_directories()
    print("目录结构检查完成")

if __name__ == "__main__":
    # 运行环境检测
    check_environment()
    
    # 创建示例用户配置
    sample_config = {
        "generation_params": {
            "num_inference_steps": 30,
            "guidance_scale": 4.0,
            "strength": 0.85
        },
        "custom_prompts": {
            "formal": "elegant sweetheart neckline evening gown, luxurious satin fabric",
            "casual": "comfortable sweetheart neckline top, soft cotton fabric"
        }
    }
    
    ConfigManager.save_user_config(sample_config, "sample_user_config.json")
    print("示例配置文件已创建: sample_user_config.json")