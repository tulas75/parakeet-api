#!/bin/bash

# Parakeet API 健康检查脚本（纯 curl 版）
# 返回 0 表示健康，非 0 表示不健康

TIMEOUT=10
HOST="localhost"
PORT="5092"

# 优先检查简单健康端点，其次检查详细健康端点
if curl -f -s --max-time "$TIMEOUT" "http://$HOST:$PORT/health/simple" >/dev/null 2>&1; then
  echo "Health check passed (/health/simple)"
  exit 0
fi

if curl -f -s --max-time "$TIMEOUT" "http://$HOST:$PORT/health" >/dev/null 2>&1; then
  echo "Health check passed (/health)"
  exit 0
fi

echo "Health check failed"
exit 1