#!/bin/bash

# Parakeet API 健康检查脚本
# 检查服务是否正常运行

# 设置超时时间
TIMEOUT=10
HOST="localhost"
PORT="5092"

# 检查端口是否监听
if ! nc -z $HOST $PORT 2>/dev/null; then
    echo "Port $PORT is not listening"
    exit 1
fi

# 检查简单健康端点
if curl -f -s --max-time $TIMEOUT http://$HOST:$PORT/health/simple >/dev/null 2>&1; then
    echo "Health check passed"
    exit 0
else
    echo "Health check failed"
    exit 1
fi
