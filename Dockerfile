# 使用单阶段构建，确保包安装可靠性
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV NUMBA_CACHE_DIR=/tmp/numba_cache
ENV NUMBA_DISABLE_JIT=0

# 创建非root用户
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Update and install system dependencies (minimal: only runtime required)\nRUN apt-get update && apt-get install -y --no-install-recommends \\\n    python3.10 python3-pip \\\n    # Development toolchain needed for building (for compiling C++ extensions like texterrors)\\\n    build-essential g++ gcc python3-dev pkg-config \\\n    ffmpeg \\\n    curl \\\n    ca-certificates \\\n    gosu \\\n    libsndfile1 \\\n    libjemalloc2 \\\n    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 升级pip到最新版本
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# 设置CUDA环境变量（随CUDA镜像）
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 设置工作目录
WORKDIR /app

# 复制requirements.txt
COPY requirements.txt .

# Unified installation: get GPU wheels from PyTorch source for CUDA 13.0, and resolve dependencies like NeMo at the same time\n# Keep PyPI and NVIDIA sources as extra indexes to avoid missing packages during resolution\nRUN python3 -m pip install --no-cache-dir \\\n    --index-url https://download.pytorch.org/whl/cu130 \\\n    --extra-index-url https://pypi.org/simple \\\n    --extra-index-url https://pypi.nvidia.com \\\n    torch torchaudio -r requirements.txt \\\n    && python3 -m pip cache purge

# 验证关键包是否安装成功
RUN python3 -c "import flask, torch, torchaudio, numpy; print('OK:', flask.__version__, torch.__version__, torchaudio.__version__, numpy.__version__)"

# 复制应用代码和健康检查脚本
COPY --chown=appuser:appuser . .
COPY healthcheck.sh /usr/local/bin/healthcheck.sh
RUN chmod +x /usr/local/bin/healthcheck.sh

# 创建必要的目录
RUN mkdir -p /app/models /app/temp_uploads /tmp/numba_cache \
    && chown -R appuser:appuser /app \
    && chmod 777 /tmp/numba_cache

# 设置环境变量
ENV HF_HOME=/app/models
ENV HF_HUB_DISABLE_SYMLINKS_WARNING=true
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.85
ENV CUDA_LAUNCH_BLOCKING=0
# glibc 分配器调优：减少 arena 数，降低 RSS 膨胀
ENV MALLOC_ARENA_MAX=2
# 默认启用 jemalloc（用户无需配置），如不需要可在运行时设置 USE_JEMALLOC=false 禁用
ENV USE_JEMALLOC=true

# 创建启动脚本
RUN echo '#!/bin/bash\n\
# Adjust user permissions using UID/GID from environment variables\n\
PUID=${PUID:-1000}\n\
PGID=${PGID:-1000}\n\
\n\
echo \"Configuring user permissions: UID=$PUID, GID=$PGID\"\n\
\n\
# Adjust appuser UID/GID\n\
if [ \"$PUID\" != \"$(id -u appuser)\" ]; then\n\
    echo \"Adjusting appuser UID from $(id -u appuser) to $PUID\"\n\
    usermod -u $PUID appuser 2>/dev/null || true\n\
fi\n\
if [ \"$PGID\" != \"$(id -g appuser)\" ]; then\n\
    echo \"Adjusting appuser GID from $(id -g appuser) to $PGID\"\n\
    groupmod -g $PGID appuser 2>/dev/null || true\n\
    usermod -g $PGID appuser 2>/dev/null || true\n\
fi\n\
\n\
# Create and set user home directory permissions\n\
mkdir -p /home/appuser\n\
chown -R appuser:$PGID /home/appuser\n\
chmod 755 /home/appuser\n\
\n\
# Create config directories\n\
mkdir -p /home/appuser/.config/matplotlib\n\
mkdir -p /home/appuser/.lhotse/tools\n\
chown -R appuser:$PGID /home/appuser/.config\n\
chown -R appuser:$PGID /home/appuser/.lhotse\n\
chmod -R 755 /home/appuser/.config\n\
chmod -R 755 /home/appuser/.lhotse\n\
\n\
# Ensure application directory permissions are correct\n\
chown -R appuser:$PGID /app/temp_uploads\n\
\n\
# Ensure numba cache directory permissions are correct\n\
mkdir -p /tmp/numba_cache\n\
chmod 777 /tmp/numba_cache\n\
chown -R appuser:$PGID /tmp/numba_cache 2>/dev/null || true\n\
\n\
# If models directory exists, try adjusting permissions\n\
if [ -d \"/app/models\" ]; then\n\
    chown -R appuser:$PGID /app/models 2>/dev/null || true\n\
fi\n\
\n\
# Set environment variables to ensure correct config directory\n\
export MPLCONFIGDIR=/home/appuser/.config/matplotlib\n\
export HOME=/home/appuser\n\
\n\
# Optionally enable jemalloc to more actively return free memory\n\
if [ \"+${USE_JEMALLOC}+\" = \"+true+\" ]; then\n\
    if [ -f /usr/lib/x86_64-linux-gnu/libjemalloc.so.2 ]; then\n\
        export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2${LD_PRELOAD:+:$LD_PRELOAD}\n\
        # Background thread recycling, decay configuration (more aggressive release to OS)\n\
        export MALLOC_CONF=background_thread:true,dirty_decay_ms:1000,muzzy_decay_ms:1000\n\
        echo \"Using jemalloc: $LD_PRELOAD\"\n\
    else\n\
        echo \"jemalloc not found, skipping enable\"\n\
    fi\n\
fi\n\
\n\
# Verify Python packages are available\n\
echo \"Verifying Python packages...\"\n\
python3 -c \"import flask; print(\\\"Flask OK\\\")\" || exit 1\n\
python3 -c \"import torch; print(\\\"PyTorch OK\\\")\" || exit 1\n\
\n\
# Switch to appuser and start application\n\
exec gosu appuser python3 app.py\n\
' > /usr/local/bin/start.sh && chmod +x /usr/local/bin/start.sh

# 暴露端口
EXPOSE 5092

# 健康检查
HEALTHCHECK --interval=30s --timeout=15s --start-period=90s --retries=3 \
    CMD /usr/local/bin/healthcheck.sh

# 启动命令
CMD ["/usr/local/bin/start.sh"]
