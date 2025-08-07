#!/bin/bash

# 健康检查测试脚本
# 用于验证容器健康检查是否正常工作

echo "🔍 测试 Parakeet API 健康检查..."

# 等待服务启动
echo "等待服务启动..."
sleep 5

# 测试简单健康检查端点
echo "测试简单健康检查端点..."
if curl -f -s http://localhost:5092/health/simple; then
    echo "✅ 简单健康检查 - 通过"
else
    echo "❌ 简单健康检查 - 失败"
fi

# 测试详细健康检查端点
echo -e "\n测试详细健康检查端点..."
if curl -f -s http://localhost:5092/health | python3 -m json.tool; then
    echo "✅ 详细健康检查 - 通过"
else
    echo "❌ 详细健康检查 - 失败"
fi

# 检查Docker健康状态
echo -e "\n检查Docker容器健康状态..."
docker ps --filter "name=parakeet-api-docker" --format "table {{.Names}}\t{{.Status}}"
