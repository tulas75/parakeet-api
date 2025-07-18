# 多阶段构建：第一阶段用于编译依赖
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS builder

# 设置环境变量，避免交互式安装
ENV DEBIAN_FRONTEND=noninteractive

# 更新并安装编译依赖，合并命令减少层数
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-dev \
    build-essential \
    pkg-config \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# 升级pip到最新版本
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# 复制requirements.txt并安装Python依赖
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir \
    numpy \
    typing_extensions \
    Cython \
    && pip install --no-cache-dir -r /tmp/requirements.txt

# 第二阶段：运行时镜像
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV NUMBA_CACHE_DIR=/tmp/numba_cache
ENV NUMBA_DISABLE_JIT=0

# 创建非root用户
RUN groupadd -r appuser && useradd -r -g appuser appuser

# 安装运行时依赖，合并命令并清理缓存
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    ffmpeg \
    sox \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && apt-get autoremove -y

# 从builder阶段复制已安装的Python包
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 设置工作目录
WORKDIR /app

# 复制应用代码
COPY --chown=appuser:appuser . .

# 创建必要的目录
RUN mkdir -p /app/models /app/temp_uploads /tmp/numba_cache \
    && chown -R appuser:appuser /app \
    && chmod 777 /tmp/numba_cache

# 设置环境变量
ENV HF_HOME=/app/models
ENV HF_HUB_DISABLE_SYMLINKS_WARNING=true
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV CUDA_LAUNCH_BLOCKING=1
ENV NUMBA_CACHE_DIR=/tmp/numba_cache
ENV NUMBA_DISABLE_JIT=0

# 创建启动脚本
RUN echo '#!/bin/bash\n\
# 使用环境变量中的UID/GID调整用户权限\n\
PUID=${PUID:-1000}\n\
PGID=${PGID:-1000}\n\
\n\
echo "配置用户权限: UID=$PUID, GID=$PGID"\n\
\n\
# 调整appuser的UID/GID\n\
if [ "$PUID" != "$(id -u appuser)" ]; then\n\
    echo "调整appuser UID从 $(id -u appuser) 到 $PUID"\n\
    usermod -u $PUID appuser 2>/dev/null || true\n\
fi\n\
if [ "$PGID" != "$(id -g appuser)" ]; then\n\
    echo "调整appuser GID从 $(id -g appuser) 到 $PGID"\n\
    groupmod -g $PGID appuser 2>/dev/null || true\n\
fi\n\
\n\
# 确保目录权限正确\n\
chown -R appuser:appuser /app/temp_uploads\n\
\n\
# 确保numba缓存目录权限正确\n\
mkdir -p /tmp/numba_cache\n\
chmod 777 /tmp/numba_cache\n\
\n\
# 如果models目录存在，尝试调整权限\n\
if [ -d "/app/models" ]; then\n\
    chown -R appuser:appuser /app/models 2>/dev/null || true\n\
fi\n\
\n\
# 切换到appuser并启动应用\n\
exec gosu appuser python3 app.py\n\
' > /usr/local/bin/start.sh && chmod +x /usr/local/bin/start.sh

# 安装gosu用于用户切换
RUN apt-get update && apt-get install -y gosu && rm -rf /var/lib/apt/lists/*

# 暴露端口
EXPOSE 5092

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:5092/', timeout=10)" || exit 1

# 启动命令
CMD ["/usr/local/bin/start.sh"]
