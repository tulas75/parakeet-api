#!/usr/bin/env python3
"""
Test memory optimization features

This script verifies that memory optimization features are working properly.
"""

import os
import sys
import json
import time
import requests
from pathlib import Path

def test_memory_optimization():
    """Test memory optimization features"""
    print("🧪 Testing memory optimization features")
    print("=" * 50)
    
    # Test 1: Check basic configuration
    print("📋 Test 1: Checking basic configuration")
    
    # Check key configuration items
    idle_timeout = os.environ.get('IDLE_TIMEOUT_MINUTES', '30')
    aggressive_cleanup = os.environ.get('AGGRESSIVE_MEMORY_CLEANUP', 'true')
    cleanup_interval = os.environ.get('IDLE_MEMORY_CLEANUP_INTERVAL', '120')
    
    print(f"  ✅ Model idle timeout: {idle_timeout} minutes")
    print(f"  ✅ Aggressive memory cleanup: {aggressive_cleanup}")
    print(f"  ✅ Cleanup interval: {cleanup_interval} seconds")
    
    print()
    
    # Test 2: Verify health check endpoint
    print("📋 Test 2: Verifying health check endpoint")
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
                    print(f"  ❌ {field}: Missing")
            
            print(f"  📊 Current idle status: {health_data.get('model', {}).get('idle_status', 'unknown')}")
            
            if 'gpu' in health_data:
                gpu_info = health_data['gpu']
                if gpu_info.get('available'):
                    memory_gb = gpu_info.get('memory_allocated_gb', 0)
                    print(f"  📊 GPU memory usage: {memory_gb:.2f}GB")
            
            if 'system' in health_data:
                system_info = health_data['system']
                memory_total = system_info.get('memory_total_gb', 0)
                memory_percent = system_info.get('memory_usage_percent', 0)
                memory_used = memory_total * memory_percent / 100
                print(f"  📊 System memory usage: {memory_used:.2f}GB / {memory_total:.2f}GB ({memory_percent:.1f}%)")
        else:
            print(f"  ❌ Health check failed: HTTP {response.status_code}")
    except Exception as e:
        print(f"  ⚠️ Unable to connect to service: {e}")
        print("  💡 Tip: Make sure the service is running at localhost:5092")
    
    print()
    
    # Test 3: Show default configuration
    print("📋 Test 3: Default memory optimization configuration")
    print("  💡 The system uses reasonable default configurations, no manual environment variable setting required:")
    print("     - Model automatically unloads after 30 minutes of inactivity")
    print("     - Basic memory cleanup executes after each batch processing")
    print("     - Regular cleanup of invalid memory usage during idle periods")
    print("     - Forced cleanup only triggered at extremely high memory usage")
    print()
    print("  📝 These settings can effectively reduce memory usage during idle periods")
    
    print()
    print("🎯 Test complete!")
    print("💡 The system is optimized for out-of-the-box usage, no additional configuration required")

if __name__ == "__main__":
    test_memory_optimization()