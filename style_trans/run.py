# run.py - 简化启动脚本
import sys
import argparse
from pathlib import Path
from neckline_changer import NecklineChanger
from config import GenerationConfig, AppConfig, check_environment

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description=f"{AppConfig.APP_NAME} - 命令行工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python run.py --single image.jpg mask.png                    # 处理单张图片
  python run.py --batch                                        # 批量处理默认目录
  python run.py --batch --input ./photos --masks ./masks      # 批量处理指定目录
  python run.py --single image.jpg mask.png --quality high    # 高质量模式
  python run.py --single image.jpg mask.png --prompt "elegant sweetheart neckline dress"  # 自定义提示词
        """
    )
    
    # 操作模式
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--single', nargs=2, metavar=('IMAGE', 'MASK'),
                           help='处理单张图片: 图片路径 mask路径')
    mode_group.add_argument('--batch', action='store_true',
                           help='批量处理模式')
    mode_group.add_argument('--gui', action='store_true',
                           help='启动图形界面')
    
    # 路径参数
    parser.add_argument('--input', type=str, help='输入目录路径 (批量模式)')
    parser.add_argument('--masks', type=str, help='mask目录路径 (批量模式)')
    parser.add_argument('--output', type=str, help='输出路径')
    
    # 生成参数
    parser.add_argument('--quality', choices=['fast', 'default', 'high', 'experimental'],
                       default='default', help='生成质量 (默认: default)')
    parser.add_argument('--steps', type=int, help='推理步数')
    parser.add_argument('--guidance', type=float, help='引导强度')
    parser.add_argument('--strength', type=float, help='替换强度')
    parser.add_argument('--seed', type=int, help='随机种子')
    
    # 其他参数
    parser.add_argument('--prompt', type=str, help='自定义提示词')
    parser.add_argument('--model', type=str, help='模型路径')
    parser.add_argument('--verbose', action='store_true', help='详细输出')
    
    return parser.parse_args()

def get_generation_config(args):
    """根据参数获取生成配置"""
    # 选择质量预设
    quality_configs = {
        'fast': GenerationConfig.FAST_PARAMS,
        'default': GenerationConfig.DEFAULT_PARAMS,
        'high': GenerationConfig.HIGH_QUALITY_PARAMS,
        'experimental': GenerationConfig.EXPERIMENTAL_PARAMS
    }
    
    config = quality_configs[args.quality].copy()
    
    # 应用命令行覆盖
    if args.steps is not None:
        config['num_inference_steps'] = args.steps
    if args.guidance is not None:
        config['guidance_scale'] = args.guidance
    if args.strength is not None:
        config['strength'] = args.strength
    if args.seed is not None:
        config['seed'] = args.seed
    
    return config

def run_single_mode(args, changer):
    """单张图片处理模式"""
    image_path, mask_path = args.single
    output_path = args.output
    
    print(f"处理图片: {image_path}")
    print(f"使用mask: {mask_path}")
    print(f"质量模式: {args.quality}")
    
    # 获取生成配置
    gen_config = get_generation_config(args)
    
    # 处理图片
    success, result, time_str = changer.process_single_image(
        image_path=image_path,
        mask_path=mask_path,
        output_path=output_path,
        custom_prompt=args.prompt,
        generation_config=gen_config
    )
    
    if success:
        print(f"✓ 处理成功!")
        print(f"  输出: {result}")
        print(f"  耗时: {time_str}")
    else:
        print(f"✗ 处理失败: {result}")
        return False
    
    return True

def run_batch_mode(args, changer):
    """批量处理模式"""
    print("=== 批量处理模式 ===")
    print(f"质量模式: {args.quality}")
    
    # 获取生成配置
    gen_config = get_generation_config(args)
    
    # 执行批量处理
    stats = changer.batch_process(
        input_dir=args.input,
        mask_dir=args.masks,
        output_dir=args.output,
        generation_config=gen_config
    )
    
    print(f"\n=== 处理完成 ===")
    print(f"总计: {stats['total']}")
    print(f"成功: {stats['successful']}")
    print(f"失败: {stats['failed']}")
    print(f"成功率: {stats['success_rate']}")
    print(f"平均耗时: {stats['avg_time']}")
    
    return stats['successful'] > 0

def run_gui_mode():
    """图形界面模式"""
    try:
        import tkinter as tk
        from tkinter import ttk, filedialog, messagebox
    except ImportError:
        print("错误: 无法导入tkinter，请确保Python安装完整")
        return False
    
    class NecklineChangerGUI:
        def __init__(self, root):
            self.root = root
            self.root.title(f"{AppConfig.APP_NAME} v{AppConfig.VERSION}")
            self.root.geometry("600x500")
            
            self.changer = None
            self.setup_ui()
        
        def setup_ui(self):
            # 主框架
            main_frame = ttk.Frame(self.root, padding="10")
            main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
            
            # 标题
            title_label = ttk.Label(main_frame, text=AppConfig.APP_NAME, 
                                  font=("Arial", 16, "bold"))
            title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
            
            # 文件选择区域
            file_frame = ttk.LabelFrame(main_frame, text="文件选择", padding="10")
            file_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
            
            # 图片选择
            ttk.Label(file_frame, text="选择图片:").grid(row=0, column=0, sticky=tk.W, pady=2)
            self.image_var = tk.StringVar()
            ttk.Entry(file_frame, textvariable=self.image_var, width=50).grid(row=0, column=1, padx=(10, 5))
            ttk.Button(file_frame, text="浏览", command=self.select_image).grid(row=0, column=2)
            
            # Mask选择
            ttk.Label(file_frame, text="选择Mask:").grid(row=1, column=0, sticky=tk.W, pady=2)
            self.mask_var = tk.StringVar()
            ttk.Entry(file_frame, textvariable=self.mask_var, width=50).grid(row=1, column=1, padx=(10, 5))
            ttk.Button(file_frame, text="浏览", command=self.select_mask).grid(row=1, column=2)
            
            # 输出选择
            ttk.Label(file_frame, text="输出路径:").grid(row=2, column=0, sticky=tk.W, pady=2)
            self.output_var = tk.StringVar()
            ttk.Entry(file_frame, textvariable=self.output_var, width=50).grid(row=2, column=1, padx=(10, 5))
            ttk.Button(file_frame, text="浏览", command=self.select_output).grid(row=2, column=2)
            
            # 参数设置区域
            param_frame = ttk.LabelFrame(main_frame, text="参数设置", padding="10")
            param_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
            
            # 质量选择
            ttk.Label(param_frame, text="质量模式:").grid(row=0, column=0, sticky=tk.W, pady=2)
            self.quality_var = tk.StringVar(value="default")
            quality_combo = ttk.Combobox(param_frame, textvariable=self.quality_var, 
                                       values=["fast", "default", "high", "experimental"],
                                       state="readonly", width=15)
            quality_combo.grid(row=0, column=1, sticky=tk.W, padx=(10, 0))
            
            # 自定义提示词
            ttk.Label(param_frame, text="自定义提示词:").grid(row=1, column=0, sticky=tk.W, pady=2)
            self.prompt_var = tk.StringVar()
            ttk.Entry(param_frame, textvariable=self.prompt_var, width=60).grid(row=1, column=1, columnspan=2, 
                                                                              sticky=(tk.W, tk.E), padx=(10, 0))
            
            # 控制按钮区域
            button_frame = ttk.Frame(main_frame)
            button_frame.grid(row=3, column=0, columnspan=2, pady=(10, 0))
            
            # 初始化按钮
            self.init_button = ttk.Button(button_frame, text="初始化模型", command=self.initialize_model)
            self.init_button.pack(side=tk.LEFT, padx=(0, 10))
            
            # 处理按钮
            self.process_button = ttk.Button(button_frame, text="开始处理", command=self.process_image, state="disabled")
            self.process_button.pack(side=tk.LEFT, padx=(0, 10))
            
            # 批量处理按钮
            self.batch_button = ttk.Button(button_frame, text="批量处理", command=self.batch_process, state="disabled")
            self.batch_button.pack(side=tk.LEFT)
            
            # 进度条
            self.progress_var = tk.StringVar(value="准备就绪")
            ttk.Label(main_frame, textvariable=self.progress_var).grid(row=4, column=0, columnspan=2, pady=(20, 0))
            
            # 日志区域
            log_frame = ttk.LabelFrame(main_frame, text="处理日志", padding="10")
            log_frame.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
            
            self.log_text = tk.Text(log_frame, height=10, width=70)
            scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
            self.log_text.configure(yscrollcommand=scrollbar.set)
            
            self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
            scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
            
            # 配置网格权重
            self.root.columnconfigure(0, weight=1)
            self.root.rowconfigure(0, weight=1)
            main_frame.columnconfigure(1, weight=1)
            main_frame.rowconfigure(5, weight=1)
            log_frame.columnconfigure(0, weight=1)
            log_frame.rowconfigure(0, weight=1)
        
        def log(self, message):
            """添加日志信息"""
            self.log_text.insert(tk.END, f"{message}\n")
            self.log_text.see(tk.END)
            self.root.update_idletasks()
        
        def select_image(self):
            filename = filedialog.askopenfilename(
                title="选择图片",
                filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp")]
            )
            if filename:
                self.image_var.set(filename)
        
        def select_mask(self):
            filename = filedialog.askopenfilename(
                title="选择Mask",
                filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp")]
            )
            if filename:
                self.mask_var.set(filename)
        
        def select_output(self):
            filename = filedialog.asksaveasfilename(
                title="保存位置",
                defaultextension=".jpg",
                filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")]
            )
            if filename:
                self.output_var.set(filename)
        
        def initialize_model(self):
            """初始化模型"""
            try:
                self.progress_var.set("正在初始化模型...")
                self.init_button.config(state="disabled")
                self.log("开始初始化模型...")
                
                self.changer = NecklineChanger()
                
                self.process_button.config(state="normal")
                self.batch_button.config(state="normal")
                self.progress_var.set("模型初始化完成")
                self.log("✓ 模型初始化成功")
                
            except Exception as e:
                self.progress_var.set("初始化失败")
                self.log(f"✗ 初始化失败: {str(e)}")
                messagebox.showerror("错误", f"模型初始化失败:\n{str(e)}")
                self.init_button.config(state="normal")
        
        def process_image(self):
            """处理单张图片"""
            if not self.changer:
                messagebox.showwarning("警告", "请先初始化模型")
                return
            
            image_path = self.image_var.get().strip()
            mask_path = self.mask_var.get().strip()
            
            if not image_path or not mask_path:
                messagebox.showwarning("警告", "请选择图片和mask文件")
                return
            
            try:
                self.progress_var.set("正在处理...")
                self.process_button.config(state="disabled")
                
                # 获取生成配置
                quality_configs = {
                    'fast': GenerationConfig.FAST_PARAMS,
                    'default': GenerationConfig.DEFAULT_PARAMS,
                    'high': GenerationConfig.HIGH_QUALITY_PARAMS,
                    'experimental': GenerationConfig.EXPERIMENTAL_PARAMS
                }
                gen_config = quality_configs[self.quality_var.get()]
                
                custom_prompt = self.prompt_var.get().strip() or None
                output_path = self.output_var.get().strip() or None
                
                self.log(f"开始处理: {Path(image_path).name}")
                
                success, result, time_str = self.changer.process_single_image(
                    image_path=image_path,
                    mask_path=mask_path,
                    output_path=output_path,
                    custom_prompt=custom_prompt,
                    generation_config=gen_config
                )
                
                if success:
                    self.progress_var.set("处理完成")
                    self.log(f"✓ 处理成功: {result}")
                    self.log(f"  耗时: {time_str}")
                    messagebox.showinfo("成功", f"处理完成!\n输出: {result}")
                else:
                    self.progress_var.set("处理失败")
                    self.log(f"✗ 处理失败: {result}")
                    messagebox.showerror("失败", f"处理失败:\n{result}")
                
            except Exception as e:
                self.progress_var.set("处理出错")
                self.log(f"✗ 处理出错: {str(e)}")
                messagebox.showerror("错误", f"处理出错:\n{str(e)}")
            
            finally:
                self.process_button.config(state="normal")
        
        def batch_process(self):
            """批量处理"""
            if not self.changer:
                messagebox.showwarning("警告", "请先初始化模型")
                return
            
            # 选择输入目录
            input_dir = filedialog.askdirectory(title="选择输入图片目录")
            if not input_dir:
                return
            
            # 选择mask目录
            mask_dir = filedialog.askdirectory(title="选择mask目录")
            if not mask_dir:
                return
            
            # 选择输出目录
            output_dir = filedialog.askdirectory(title="选择输出目录")
            if not output_dir:
                return
            
            try:
                self.progress_var.set("批量处理中...")
                self.batch_button.config(state="disabled")
                
                # 获取生成配置
                quality_configs = {
                    'fast': GenerationConfig.FAST_PARAMS,
                    'default': GenerationConfig.DEFAULT_PARAMS,
                    'high': GenerationConfig.HIGH_QUALITY_PARAMS,
                    'experimental': GenerationConfig.EXPERIMENTAL_PARAMS
                }
                gen_config = quality_configs[self.quality_var.get()]
                
                self.log(f"开始批量处理:")
                self.log(f"  输入: {input_dir}")
                self.log(f"  Mask: {mask_dir}")
                self.log(f"  输出: {output_dir}")
                
                stats = self.changer.batch_process(
                    input_dir=input_dir,
                    mask_dir=mask_dir,
                    output_dir=output_dir,
                    generation_config=gen_config
                )
                
                self.progress_var.set("批量处理完成")
                self.log("=== 批量处理完成 ===")
                self.log(f"总计: {stats['total']}")
                self.log(f"成功: {stats['successful']}")
                self.log(f"失败: {stats['failed']}")
                self.log(f"成功率: {stats['success_rate']}")
                
                messagebox.showinfo("完成", f"批量处理完成!\n成功: {stats['successful']}/{stats['total']}")
                
            except Exception as e:
                self.progress_var.set("批量处理出错")
                self.log(f"✗ 批量处理出错: {str(e)}")
                messagebox.showerror("错误", f"批量处理出错:\n{str(e)}")
            
            finally:
                self.batch_button.config(state="normal")
    
    # 启动GUI
    root = tk.Tk()
    app = NecklineChangerGUI(root)
    root.mainloop()
    return True

def main():
    """主函数"""
    args = parse_arguments()
    
    # 环境检查
    if args.verbose:
        check_environment()
    
    # GUI模式
    if args.gui:
        print(f"启动 {AppConfig.APP_NAME} 图形界面...")
        return run_gui_mode()
    
    # 命令行模式
    try:
        print(f"=== {AppConfig.APP_NAME} v{AppConfig.VERSION} ===")
        
        # 初始化处理器
        print("正在初始化...")
        config_override = {}
        if args.verbose:
            # 可以在这里添加详细日志配置
            pass
        
        changer = NecklineChanger(model_path=args.model, config_override=config_override)
        
        # 执行对应模式
        if args.single:
            success = run_single_mode(args, changer)
        elif args.batch:
            success = run_batch_mode(args, changer)
        
        return success
        
    except KeyboardInterrupt:
        print("\n用户中断操作")
        return False
    except Exception as e:
        print(f"程序出错: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)