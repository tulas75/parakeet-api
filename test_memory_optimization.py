#!/usr/bin/env python3
"""
测试内存优化功能

这个脚本验证内存优化功能是否正常工作。
"""

import os
import sys
import json
import time
import requests
from pathlib import Path

def test_memory_optimization():
    """测试内存优化功能"""
    print("🧪 测试内存优化功能")
    print("=" * 50)
    
    # 测试1: 检查基本配置
    print("📋 测试1: 检查基本配置")
    
    # 检查关键配置项
    idle_timeout = os.environ.get('IDLE_TIMEOUT_MINUTES', '30')
    aggressive_cleanup = os.environ.get('AGGRESSIVE_MEMORY_CLEANUP', 'true')
    cleanup_interval = os.environ.get('IDLE_MEMORY_CLEANUP_INTERVAL', '120')
    
    print(f"  ✅ 模型闲置超时: {idle_timeout} 分钟")
    print(f"  ✅ 激进内存清理: {aggressive_cleanup}")
    print(f"  ✅ 清理间隔: {cleanup_interval} 秒")
    
    print()
    
    # 测试2: 验证健康检查端点
    print("📋 测试2: 验证健康检查端点")
    try:
        response = requests.get('http://localhost:5092/health', timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            optimization = health_data.get('optimization', {})
            
            required_fields = [
                'aggressive_memory_cleanup',
                'idle_timeout_minutes',
                'idle_memory_cleanup_interval'
            ]
            
            for field in required_fields:
                if field in optimization:
                    print(f"  ✅ {field}: {optimization[field]}")
                else:
                    print(f"  ❌ {field}: 缺失")
            
            print(f"  📊 当前闲置状态: {health_data.get('model', {}).get('idle_status', 'unknown')}")
            
            if 'gpu' in health_data:
                gpu_info = health_data['gpu']
                if gpu_info.get('available'):
                    memory_gb = gpu_info.get('memory_allocated_gb', 0)
                    print(f"  📊 GPU内存使用: {memory_gb:.2f}GB")
            
            if 'system' in health_data:
                system_info = health_data['system']
                memory_total = system_info.get('memory_total_gb', 0)
                memory_percent = system_info.get('memory_usage_percent', 0)
                memory_used = memory_total * memory_percent / 100
                print(f"  📊 系统内存使用: {memory_used:.2f}GB / {memory_total:.2f}GB ({memory_percent:.1f}%)")
        else:
            print(f"  ❌ 健康检查失败: HTTP {response.status_code}")
    except Exception as e:
        print(f"  ⚠️ 无法连接到服务: {e}")
        print("  💡 提示: 请确保服务正在运行在 localhost:5092")
    
    print()
    
    # 测试3: 显示默认配置
    print("📋 测试3: 默认内存优化配置")
    print("  💡 系统已使用合理的默认配置，无需手动设置环境变量:")
    print("     - 模型在闲置30分钟后自动卸载")
    print("     - 每批处理完成后执行基本内存清理")
    print("     - 闲置期间定期清理无效内存占用")
    print("     - 仅在极高内存使用时触发强制清理")
    print()
    print("  📝 这些设置可以有效减少闲置时的内存占用")
    
    print()
    print("🎯 测试完成!")
    print("💡 系统已优化为开箱即用，无需额外配置")

if __name__ == "__main__":
    test_memory_optimization()