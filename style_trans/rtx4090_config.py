#!/usr/bin/env python3
# rtx4090_config.py - RTX 4090优化配置
import os
import torch
from pathlib import Path

# 设置环境变量优化RTX 4090
def setup_rtx4090_environment():
    """设置RTX 4090优化环境"""
    print("配置RTX 4090优化环境...")
    
    # CUDA优化设置
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,roundup_power2_divisions:16'
    os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'
    
    # RTX 4090特定优化
    os.environ['NVIDIA_TF32_OVERRIDE'] = '1'  # 启用TF32加速
    
    # 设置显存使用策略
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # 48GB显存，可以使用更大的内存分配
        torch.cuda.set_per_process_memory_fraction(0.95)
        
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

class RTX4090ModelConfig:
    """RTX 4090专用模型配置"""
    
    # 启用所有高性能选项
    USE_FP16 = True
    MODEL_VARIANT = "fp16"
    
    # 对于48GB显存，可以关闭CPU卸载获得更好性能
    USE_CPU_OFFLOAD = False
    USE_ATTENTION_SLICING = False  # 48GB显存足够，不需要切片
    
    # 启用高性能优化
    ENABLE_FLASH_ATTENTION = True
    ENABLE_MEMORY_EFFICIENT_ATTENTION = False  # Flash attention已足够
    
    DEVICE = "cuda"

class RTX4090GenerationConfig:
    """RTX 4090高质量生成配置"""
    
    # 高质量默认参数（利用大显存优势）
    DEFAULT_PARAMS = {
        "num_inference_steps": 35,  # 提高步数
        "guidance_scale": 4.0,      # 提高引导
        "strength": 0.9,            # 提高强度
        "seed": 42,
        "eta": 0.0,
    }
    
    # 超高质量配置
    ULTRA_QUALITY_PARAMS = {
        "num_inference_steps": 50,
        "guidance_scale": 5.0,
        "strength": 0.95,
        "seed": 42,
        "eta": 0.0,
    }
    
    # 快速高质量配置
    FAST_HQ_PARAMS = {
        "num_inference_steps": 28,
        "guidance_scale": 4.5,
        "strength": 0.85,
        "seed": 42,
        "eta": 0.0,
    }

class RTX4090ImageConfig:
    """RTX 4090图像处理配置"""
    
    # 利用大显存处理更大图像
    MAX_IMAGE_SIZE = (1536, 1536)  # 提高到1536
    MIN_IMAGE_SIZE = (512, 512)
    
    # 更高输出质量
    OUTPUT_QUALITY = 98
    
    # 批处理优化
    BATCH_SIZE = 2  # 可以处理更大批次
    
    # 更精细的mask处理
    MASK_BLUR_RADIUS = 3
    MASK_CONTRAST_ENHANCE = 1.3

def apply_rtx4090_optimizations(pipe):
    """应用RTX 4090特定优化"""
    print("🔧 应用RTX 4090优化...")
    
    try:
        # 禁用不必要的内存节省功能（我们有足够显存）
        if hasattr(pipe, 'disable_attention_slicing'):
            pipe.disable_attention_slicing()
            print("禁用注意力切片（大显存模式）")
        
        if hasattr(pipe, 'disable_vae_slicing'):
            pipe.disable_vae_slicing()
            print("禁用VAE切片（大显存模式）")
        
        # 启用高性能模式
        if hasattr(pipe, 'enable_model_cpu_offload'):
            # 对于48GB显存，不使用CPU卸载
            pass
        
        # 启用编译优化（如果支持）
        try:
            if hasattr(torch, 'compile'):
                pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead")
                print("启用PyTorch编译优化")
        except Exception as e:
            print(f"编译优化失败: {e}")
        
        # 设置为评估模式
        pipe.eval()
        
        # 预热GPU
        print("GPU预热中...")
        with torch.no_grad():
            dummy_input = torch.randn(1, 4, 64, 64).half().cuda()
            _ = dummy_input * 2
        torch.cuda.synchronize()
        print("GPU预热完成")
        
    except Exception as e:
        print(f" 优化应用部分失败: {e}")

def main():
    """RTX 4090优化配置主函数"""
    setup_rtx4090_environment()
    
    print("\n=== RTX 4090配置信息 ===")
    print(f"模型配置: 高性能模式")
    print(f"显存利用: 95%")
    print(f"最大图像: 1536x1536")
    print(f"FP16优化: 启用")
    print(f"TF32加速: 启用")
    print(f"编译优化: 启用")

if __name__ == "__main__":
    main()