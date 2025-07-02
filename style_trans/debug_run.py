#!/usr/bin/env python3
# debug_run.py - 一键调试脚本
import sys
import shutil
from pathlib import Path
from datetime import datetime

def main():
    """一键调试主函数"""
    print("=== FLUX茶歇领替换工具 - 调试模式 ===")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*50)
    
    try:
        # 1. 检查文件是否存在
        print("\n检查输入文件...")
        
        required_files = [
            "input_images/051.png",
            "masks/051_mask.png",
            "models/flux-fill"
        ]
        
        all_files_exist = True
        for file_path in required_files:
            if Path(file_path).exists():
                if Path(file_path).is_dir():
                    print(f"目录存在: {file_path}")
                else:
                    size_mb = Path(file_path).stat().st_size / (1024*1024)
                    print(f"文件存在: {file_path} ({size_mb:.1f}MB)")
            else:
                print(f"缺失: {file_path}")
                all_files_exist = False
        
        if not all_files_exist:
            print("\n缺少必要文件，但仍会尝试运行以收集调试信息")
        
        # 2. 运行带日志的版本
        print("\n开始运行带详细日志的版本...")
        print("="*50)
        
        from run_with_logs import main as logged_main
        success = logged_main()
        
        print("="*50)
        
        # 3. 分析日志
        print("\n分析运行日志...")
        
        from analyze_logs import analyze_debug_logs, create_summary_report
        analyze_debug_logs()
        summary_file = create_summary_report()
        
        # 4. 创建调试包
        print("\n创建调试包...")
        debug_package = create_debug_package()
        
        # 5. 显示结果
        print(f"\n{'='*50}")
        if success:
            print("程序运行成功!")
        else:
            print("程序运行失败")
        
        print(f"\n调试信息已保存:")
        print(f"  调试包: {debug_package}")
        print(f"  摘要报告: {summary_file}")
        
        print(f"\n如果需要帮助:")
        print(f"  1. 将整个 {debug_package} 文件发送给我")
        print(f"  2. 或者发送 debug_logs/ 目录下的所有文件")
        
        return success
        
    except Exception as e:
        print(f"\n调试脚本本身出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_debug_package():
    """创建调试包"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    package_name = f"flux_debug_package_{timestamp}"
    package_dir = Path(package_name)
    
    try:
        # 创建调试包目录
        package_dir.mkdir(exist_ok=True)
        
        # 复制日志文件
        debug_logs_dir = Path("debug_logs")
        if debug_logs_dir.exists():
            shutil.copytree(debug_logs_dir, package_dir / "debug_logs", dirs_exist_ok=True)
        
        # 复制配置文件
        config_files = ["config.py", "rtx4090_config.py", "neckline_changer.py"]
        config_dir = package_dir / "config_files"
        config_dir.mkdir(exist_ok=True)
        
        for config_file in config_files:
            if Path(config_file).exists():
                shutil.copy2(config_file, config_dir / config_file)
        
        # 创建系统快照
        create_system_snapshot(package_dir / "system_snapshot.txt")
        
        # 创建打包说明
        create_package_readme(package_dir / "README.txt")
        
        # 压缩成zip文件
        shutil.make_archive(package_name, 'zip', package_dir)
        
        # 删除临时目录
        shutil.rmtree(package_dir)
        
        zip_file = f"{package_name}.zip"
        print(f"调试包已创建: {zip_file}")
        
        return zip_file
        
    except Exception as e:
        print(f"创建调试包失败: {e}")
        return str(package_dir)

def create_system_snapshot(output_file):
    """创建系统快照"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=== 系统快照 ===\n")
        f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Python信息
        f.write(f"Python版本: {sys.version}\n")
        f.write(f"Python路径: {sys.executable}\n\n")
        
        # 环境变量
        import os
        f.write("相关环境变量:\n")
        env_vars = ['CUDA_VISIBLE_DEVICES', 'PYTORCH_CUDA_ALLOC_CONF', 'TORCH_CUDNN_V8_API_ENABLED']
        for var in env_vars:
            value = os.environ.get(var, '未设置')
            f.write(f"  {var}: {value}\n")
        f.write("\n")
        
        # 文件结构
        f.write("项目文件结构:\n")
        try:
            for item in sorted(Path(".").iterdir()):
                if item.is_file():
                    size = item.stat().st_size
                    f.write(f"  {item.name} ({size} bytes)\n")
                elif item.is_dir() and not item.name.startswith('.'):
                    f.write(f"  {item.name}/\n")
                    try:
                        for subitem in sorted(item.iterdir())[:5]:  # 只显示前5个
                            f.write(f"    - {subitem.name}\n")
                        if len(list(item.iterdir())) > 5:
                            f.write(f"    ... 还有 {len(list(item.iterdir())) - 5} 个文件\n")
                    except:
                        f.write(f"    (无法读取目录内容)\n")
        except Exception as e:
            f.write(f"  错误: {e}\n")

def create_package_readme(output_file):
    """创建包说明文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("""=== FLUX调试包说明 ===

这个调试包包含以下内容:

debug_logs/
  - flux_debug_*.log: 详细运行日志
  - system_info_*.json: 系统信息
  - debug_report_*.txt: 调试报告
  - summary_report.txt: 摘要报告

config_files/
  - config.py: 配置文件
  - neckline_changer.py: 主程序
  - rtx4090_config.py: RTX 4090优化配置

system_snapshot.txt: 系统快照

如何使用这个调试包:
1. 将整个zip文件发送给开发者
2. 或者解压后发送其中的日志文件

重要文件优先级:
1. debug_logs/flux_debug_*.log (最重要)
2. debug_logs/system_info_*.json
3. debug_logs/summary_report.txt

联系开发者时请提供:
- 遇到的具体问题描述
- 期望的结果
- 这个调试包
""")

if __name__ == "__main__":
    success = main()
    input("\n按任意键退出...")
    sys.exit(0 if success else 1)