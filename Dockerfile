# Use single-stage build to ensure reliable package installation
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV NUMBA_CACHE_DIR=/tmp/numba_cache
ENV NUMBA_DISABLE_JIT=0

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Update and install system dependencies (minimal: only runtime required)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip \
    # Development toolchain needed for building (for compiling C++ extensions like texterrors)
    build-essential g++ gcc python3-dev pkg-config \
    ffmpeg \
    curl \
    ca-certificates \
    gosu \
    libsndfile1 \
    libjemalloc2 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Upgrade pip to latest version
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Set CUDA environment variables (comes with CUDA image)
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Set working directory
WORKDIR /app

# Copy requirements.txt
COPY requirements.txt .

# Unified installation: get GPU wheels from PyTorch source for CUDA 13.0, and resolve dependencies like NeMo at the same time
# Keep PyPI and NVIDIA sources as extra indexes to avoid missing packages during resolution
RUN python3 -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/c128 \
    --extra-index-url https://pypi.org/simple \
    --extra-index-url https://pypi.nvidia.com \
    torch torchaudio -r requirements.txt \
    && python3 -m pip cache purge

# Verify that key packages are installed successfully
RUN python3 -c "import flask, torch, torchaudio, numpy; print('OK:', flask.__version__, torch.__version__, torchaudio.__version__, numpy.__version__)"

# Copy application code and health check script
COPY --chown=appuser:appuser . .
COPY healthcheck.sh /usr/local/bin/healthcheck.sh
RUN chmod +x /usr/local/bin/healthcheck.sh

# Create necessary directories
RUN mkdir -p /app/models /app/temp_uploads /tmp/numba_cache \
    && chown -R appuser:appuser /app \
    && chmod 777 /tmp/numba_cache

# Set environment variables
ENV HF_HOME=/app/models
ENV HF_HUB_DISABLE_SYMLINKS_WARNING=true
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.85
ENV CUDA_LAUNCH_BLOCKING=0
# glibc allocator tuning: reduce arena count, lower RSS inflation
ENV MALLOC_ARENA_MAX=2
# Enable jemalloc by default (no user configuration needed), can be disabled at runtime with USE_JEMALLOC=false
ENV USE_JEMALLOC=true

# Create startup script
RUN echo '#!/bin/bash\n\
# Adjust user permissions using UID/GID from environment variables\n\
PUID=${PUID:-1000}\n\
PGID=${PGID:-1000}\n\
\n\
echo "Configuring user permissions: UID=$PUID, GID=$PGID"\n\
\n\
# Adjust appuser UID/GID\n\
if [ "$PUID" != "$(id -u appuser)" ]; then\n\
    echo "Adjusting appuser UID from $(id -u appuser) to $PUID"\n\
    usermod -u $PUID appuser 2>/dev/null || true\n\
fi\n\
if [ "$PGID" != "$(id -g appuser)" ]; then\n\
    echo "Adjusting appuser GID from $(id -g appuser) to $PGID"\n\
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
if [ -d "/app/models" ]; then\n\
    chown -R appuser:$PGID /app/models 2>/dev/null || true\n\
fi\n\
\n\
# Set environment variables to ensure correct config directory\n\
export MPLCONFIGDIR=/home/appuser/.config/matplotlib\n\
export HOME=/home/appuser\n\
\n\
# Optionally enable jemalloc to more actively return free memory\n\
if [ "+${USE_JEMALLOC}+" = "+true+" ]; then\n\
    if [ -f /usr/lib/x86_64-linux-gnu/libjemalloc.so.2 ]; then\n\
        export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2${LD_PRELOAD:+:$LD_PRELOAD}\n\
        # Background thread recycling, decay configuration (more aggressive release to OS)\n\
        export MALLOC_CONF=background_thread:true,dirty_decay_ms:1000,muzzy_decay_ms:1000\n\
        echo "Using jemalloc: $LD_PRELOAD"\n\
    else\n\
        echo "jemalloc not found, skipping enable"\n\
    fi\n\
fi\n\
\n\
# Verify Python packages are available\n\
echo "Verifying Python packages..."\n\
python3 -c "import flask; print(\"Flask OK\")" || exit 1\n\
python3 -c "import torch; print(\"PyTorch OK\")" || exit 1\n\
\n\
# Switch to appuser and start application\n\
exec gosu appuser python3 app.py\n\
' > /usr/local/bin/start.sh && chmod +x /usr/local/bin/start.sh

# Expose port
EXPOSE 5092

# Health check
HEALTHCHECK --interval=30s --timeout=15s --start-period=90s --retries=3 \
    CMD /usr/local/bin/healthcheck.sh

# Startup command
CMD ["/usr/local/bin/start.sh"]