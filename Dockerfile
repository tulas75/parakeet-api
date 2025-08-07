# 使用单阶段构建，确保包安装可靠性
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV NUMBA_CACHE_DIR=/tmp/numba_cache
ENV NUMBA_DISABLE_JIT=0

# 创建非root用户
RUN groupadd -r appuser && useradd -r -g appuser appuser

# 更新并安装系统依赖
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-dev \
    build-essential \
    pkg-config \
    cmake \
    git \
    ffmpeg \
    sox \
    curl \
    netcat-openbsd \
    gosu \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && apt-get autoremove -y

# 升级pip到最新版本
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# 设置CUDA环境变量
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 设置工作目录
WORKDIR /app

# 复制requirements.txt
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir \
    numpy \
    typing_extensions \
    Cython \
    && pip install --no-cache-dir -r requirements.txt

# 验证关键包是否安装成功
RUN python3 -c "import flask; print('Flask version:', flask.__version__)" && \
    python3 -c "import torch; print('PyTorch version:', torch.__version__)" && \
    python3 -c "import numpy; print('NumPy version:', numpy.__version__)"

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
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV CUDA_LAUNCH_BLOCKING=1

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
    usermod -g $PGID appuser 2>/dev/null || true\n\
fi\n\
\n\
# 创建并设置用户主目录权限\n\
mkdir -p /home/appuser\n\
chown -R appuser:$PGID /home/appuser\n\
chmod 755 /home/appuser\n\
\n\
# 创建配置目录\n\
mkdir -p /home/appuser/.config/matplotlib\n\
mkdir -p /home/appuser/.lhotse/tools\n\
chown -R appuser:$PGID /home/appuser/.config\n\
chown -R appuser:$PGID /home/appuser/.lhotse\n\
chmod -R 755 /home/appuser/.config\n\
chmod -R 755 /home/appuser/.lhotse\n\
\n\
# 确保应用目录权限正确\n\
chown -R appuser:$PGID /app/temp_uploads\n\
\n\
# 确保numba缓存目录权限正确\n\
mkdir -p /tmp/numba_cache\n\
chmod 777 /tmp/numba_cache\n\
chown -R appuser:$PGID /tmp/numba_cache 2>/dev/null || true\n\
\n\
# 如果models目录存在，尝试调整权限\n\
if [ -d "/app/models" ]; then\n\
    chown -R appuser:$PGID /app/models 2>/dev/null || true\n\
fi\n\
\n\
# 设置环境变量确保配置目录正确\n\
export MPLCONFIGDIR=/home/appuser/.config/matplotlib\n\
export HOME=/home/appuser\n\
\n\
# 验证Python包是否可用\n\
echo "验证Python包..."\n\
python3 -c "import flask; print(\"Flask OK\")" || exit 1\n\
python3 -c "import torch; print(\"PyTorch OK\")" || exit 1\n\
\n\
# 切换到appuser并启动应用\n\
exec gosu appuser python3 app.py\n\
' > /usr/local/bin/start.sh && chmod +x /usr/local/bin/start.sh

# 暴露端口
EXPOSE 5092

# 健康检查
HEALTHCHECK --interval=30s --timeout=15s --start-period=90s --retries=3 \
    CMD /usr/local/bin/healthcheck.sh

# 启动命令
CMD ["/usr/local/bin/start.sh"]
