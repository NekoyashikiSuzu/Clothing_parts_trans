#!/usr/bin/env python3
# analyze_logs.py - 日志分析工具
import json
import re
from pathlib import Path
from datetime import datetime

def analyze_debug_logs(log_dir="debug_logs"):
    """分析调试日志"""
    log_dir = Path(log_dir)
    
    if not log_dir.exists():
        print("日志目录不存在")
        return
    
    print("=== 日志分析结果 ===")
    
    # 找到最新的日志文件
    log_files = list(log_dir.glob("flux_debug_*.log"))
    json_files = list(log_dir.glob("system_info_*.json"))
    
    if not log_files:
        print("未找到日志文件")
        return
    
    # 分析最新的日志
    latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
    print(f"分析日志文件: {latest_log.name}")
    
    with open(latest_log, 'r', encoding='utf-8') as f:
        log_content = f.read()
    
    # 分析系统信息
    if json_files:
        latest_json = max(json_files, key=lambda x: x.stat().st_mtime)
        with open(latest_json, 'r', encoding='utf-8') as f:
            system_info = json.load(f)
        
        print(f"\n系统信息:")
        print(f"  操作系统: {system_info['platform']['system']} {system_info['platform']['release']}")
        print(f"  Python: {system_info['platform']['python_version']}")
        print(f"  PyTorch: {system_info['pytorch']['version']}")
        print(f"  CUDA: {system_info['pytorch']['cuda_available']}")
        
        if system_info['pytorch']['cuda_available']:
            for gpu in system_info.get('gpu', []):
                print(f"  GPU: {gpu['name']} ({gpu['total_memory_gb']}GB)")
        
        print(f"  内存: {system_info['memory']['total_gb']}GB")
    
    # 分析日志内容
    print(f"\n日志分析:")
    
    # 错误统计
    errors = re.findall(r'ERROR.*', log_content)
    warnings = re.findall(r'WARNING.*', log_content)
    
    print(f"  错误数量: {len(errors)}")
    print(f"  警告数量: {len(warnings)}")
    
    # 找到关键步骤
    steps = re.findall(r'步骤: (.+)', log_content)
    if steps:
        print(f"\n执行步骤:")
        for i, step in enumerate(steps, 1):
            print(f"  {i}. {step}")
    
    # 查找GPU内存信息
    gpu_memory_logs = re.findall(r'GPU内存.*?: (.+)', log_content)
    if gpu_memory_logs:
        print(f"\nGPU内存使用:")
        for memory_log in gpu_memory_logs[-5:]:  # 显示最后5条
            print(f"  {memory_log}")
    
    # 查找异常
    if errors:
        print(f"\n发现的错误:")
        for error in errors[-3:]:  # 显示最后3个错误
            print(f"  {error}")
    
    # 查找性能信息
    timing_logs = re.findall(r'完成: (.+) \(耗时: (.+)\)', log_content)
    if timing_logs:
        print(f"\n性能统计:")
        for operation, time_taken in timing_logs:
            print(f"  {operation}: {time_taken}")
    
    # 查找最后的状态
    last_lines = log_content.split('\n')[-20:]
    
    if any('处理完成' in line for line in last_lines):
        print(f"\n状态: 处理成功完成")
    elif any('ERROR' in line for line in last_lines):
        print(f"\n状态: 处理过程中出现错误")
    elif any('初始化完成' in line for line in last_lines):
        print(f"\n状态: 初始化完成但未开始处理")
    else:
        print(f"\n状态: 未知")

def create_summary_report(log_dir="debug_logs"):
    """创建摘要报告"""
    log_dir = Path(log_dir)
    
    # 创建摘要文件
    summary_file = log_dir / "summary_report.txt"
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=== FLUX调试摘要报告 ===\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 重定向print输出到文件
        import sys
        old_stdout = sys.stdout
        sys.stdout = f
        
        try:
            analyze_debug_logs(log_dir)
        finally:
            sys.stdout = old_stdout
    
    print(f"摘要报告已生成: {summary_file}")
    return summary_file

if __name__ == "__main__":
    analyze_debug_logs()
    create_summary_report()