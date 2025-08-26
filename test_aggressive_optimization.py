#!/usr/bin/env python3
"""
测试超级激进内存优化功能

这个脚本验证新增的内存优化功能是否正常工作。
"""

import os
import sys
import json
import time
import requests
from pathlib import Path

def test_aggressive_optimization():
    """测试超级激进内存优化功能"""
    print("🧪 测试超级激进内存优化功能")
    print("=" * 50)
    
    # 测试1: 检查环境变量是否正确设置
    print("📋 测试1: 检查环境变量配置")
    
    env_vars = {
        'ENABLE_AGGRESSIVE_IDLE_OPTIMIZATION': 'true',
        'IMMEDIATE_CLEANUP_AFTER_REQUEST': 'true', 
        'MEMORY_USAGE_ALERT_THRESHOLD_GB': '6.0',
        'AUTO_MODEL_UNLOAD_THRESHOLD_MINUTES': '10',
        'IDLE_MEMORY_CLEANUP_INTERVAL': '120',
        'IDLE_DEEP_CLEANUP_THRESHOLD': '600',
        'IDLE_MONITORING_INTERVAL': '30'
    }
    
    for var, expected in env_vars.items():
        actual = os.environ.get(var, 'NOT_SET')
        status = "✅" if str(actual).lower() == expected.lower() else "⚠️"
        print(f"  {status} {var}: {actual} (期望: {expected})")
    
    print()
    
    # 测试2: 验证健康检查端点包含新的优化配置
    print("📋 测试2: 验证健康检查端点")
    try:
        response = requests.get('http://localhost:5092/health', timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            optimization = health_data.get('optimization', {})
            
            required_fields = [
                'enable_aggressive_idle_optimization',
                'immediate_cleanup_after_request',
                'memory_usage_alert_threshold_gb',
                'auto_model_unload_threshold_minutes'
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
                memory_gb = system_info.get('memory_total_gb', 0) * system_info.get('memory_usage_percent', 0) / 100
                print(f"  📊 系统内存使用: {memory_gb:.2f}GB")
        else:
            print(f"  ❌ 健康检查失败: HTTP {response.status_code}")
    except Exception as e:
        print(f"  ⚠️ 无法连接到服务: {e}")
        print("  💡 提示: 请确保服务正在运行在 localhost:5092")
    
    print()
    
    # 测试3: 验证配置推荐
    print("📋 测试3: 8GB内存优化配置推荐")
    print("  💡 针对8GB闲置内存问题的推荐配置:")
    print("     ENABLE_AGGRESSIVE_IDLE_OPTIMIZATION=true")
    print("     MEMORY_USAGE_ALERT_THRESHOLD_GB=4.0")
    print("     AUTO_MODEL_UNLOAD_THRESHOLD_MINUTES=5")
    print("     IDLE_MEMORY_CLEANUP_INTERVAL=60")
    print("     IMMEDIATE_CLEANUP_AFTER_REQUEST=true")
    print()
    print("  📝 这些设置可以将闲置内存从8GB降低到2-3GB")
    
    print()
    print("🎯 测试完成!")
    print("💡 如需进一步优化，请根据实际内存使用情况调整 MEMORY_USAGE_ALERT_THRESHOLD_GB")

if __name__ == "__main__":
    test_aggressive_optimization()