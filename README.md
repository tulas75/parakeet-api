# Parakeet-API-Docker

> 本项目 fork 自 [jianchang512/parakeet-api](https://github.com/jianchang512/parakeet-api)，感谢原作者的贡献！

基于 NVIDIA NeMo 的 Parakeet-tdt-0.6b-v2 语音识别 API，支持长音频分片转录，自动生成 SRT 字幕，兼容 OpenAI Whisper API，内置简洁前端上传页面。

## 主要特性
- 支持长音频/视频自动分片转录，显存占用低
- **FP16 半精度优化**：显存占用减少近一半，推理速度更快
- **智能显存管理**：懒加载模式 + 自动模型卸载，空闲时释放显存
- **可配置音频分块**：根据显存大小调整分块时长，支持更长上下文
- **API Key 认证**：支持 Whisper 兼容的 Bearer Token 认证
- 输出 SRT 字幕格式，带时间戳
- 兼容 OpenAI Whisper 语音识别 API 路由
- 前端页面支持拖拽上传、进度显示、结果下载
- 支持 Docker 一键部署，自动挂载模型和临时目录
- 支持 GPU 加速（需 NVIDIA 驱动和 CUDA 环境）

## 依赖环境
- Python 3.10+
- NVIDIA GPU + CUDA 12.1+（推荐 8GB+ 显存）
- ffmpeg、ffprobe
- 主要依赖包：
  - numpy
  - flask
  - waitress
  - torch
  - nemo_toolkit[asr]
  - huggingface_hub
  - cuda-python>=12.3

详见 `requirements.txt`

## 快速开始

### 1. 模型文件准备
- 下载 [parakeet-tdt-0.6b-v2.nemo](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) 并放入 `models/` 目录。
- 目录结构如下：
  ```
  models/
    parakeet-tdt-0.6b-v2.nemo
  ```

### 2. 本地运行
```bash
pip install -r requirements.txt
python app.py
```
- 默认监听 5092 端口。
- 访问 http://localhost:5092 查看前端页面。

### 3. Docker 部署
#### 构建镜像
```bash
docker build -t parakeet-api .
```
#### 运行容器
```bash
docker run --gpus all -p 5092:5092 \
  -v $PWD/models:/app/models \
  -v $PWD/temp_uploads:/app/temp_uploads \
  parakeet-api
```
#### 或使用 docker-compose
```bash
docker-compose up --build
```

## 环境变量配置

通过 `docker-compose.yml` 中的环境变量可以灵活配置服务行为：

```yaml
environment:
  - CHUNK_MINITE=10              # 音频分块时长（分钟），8GB显存建议10-15分钟
  - IDLE_TIMEOUT_MINUTES=30      # 模型自动卸载超时时间（分钟），0=禁用
  - ENABLE_LAZY_LOAD=true        # 懒加载开关，true=按需加载，false=启动时预加载
  - API_KEY=                     # API认证密钥，留空=不认证
```

### 配置说明

- **CHUNK_MINITE**：控制长音频的分块大小
  - 8GB 显存推荐：5-8 分钟
  - 更大的分块可以减少处理开销，但需要更多显存

- **IDLE_TIMEOUT_MINUTES**：模型自动卸载时间
  - 服务空闲超过此时间后，模型将从显存中卸载
  - 设置为 0 禁用自动卸载功能
  - 适合间歇性使用场景，节省显存资源

- **ENABLE_LAZY_LOAD**：懒加载模式
  - `true`：服务启动时不占用显存，首次请求时才加载模型
  - `false`：服务启动时立即加载模型到显存
  - 懒加载适合资源共享环境，预加载适合高并发场景

- **API_KEY**：API 认证
  - 设置后，所有请求必须在 `Authorization` 头中提供 `Bearer <key>`
  - 留空则不进行身份验证

## 权限配置

现在可以通过 `docker-compose.yml` 中的环境变量来配置用户权限：

```yaml
environment:
  - PUID=1000  # 用户ID
  - PGID=1000  # 组ID
```

### 如何设置正确的 UID/GID

#### Windows 用户
通常使用默认值即可：
```yaml
- PUID=1000
- PGID=1000
```

#### Linux 用户
1. 查看模型文件的所有者：
   ```bash
   ls -la models/parakeet-tdt-0.6b-v2.nemo
   ```
2. 输出示例：
   ```
   -rw-r--r-- 1 1001 1001 2.1G Jul 16 10:00 parakeet-tdt-0.6b-v2.nemo
   ```
3. 设置对应的 UID/GID：
   ```yaml
   - PUID=1001
   - PGID=1001
   ```

### 完整的 docker-compose.yml 示例

```yaml
version: '3.8'
services:
  parakeet-api-docker:
    container_name: parakeet-api-docker
    build: 
      context: .
      dockerfile: Dockerfile
    ports:
      - "5092:5092"
    volumes:
      - ./models:/app/models:ro
      - ./temp_uploads:/app/temp_uploads
    environment:
      - CHUNK_MINITE=10
      - IDLE_TIMEOUT_MINUTES=30
      - ENABLE_LAZY_LOAD=true
      - API_KEY=
      - PUID=1000    # 根据需要修改
      - PGID=1000    # 根据需要修改
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```


### 权限配置使用步骤

1. 将模型文件放置在 `models/parakeet-tdt-0.6b-v2.nemo`
2. 根据需要修改 `docker-compose.yml` 中的 `PUID` 和 `PGID`
3. 运行：`docker-compose up --build`

容器启动时会显示：
```
配置用户权限: UID=1000, GID=1000
调整appuser UID从 999 到 1000
调整appuser GID从 999 到 1000
```

## API 说明
### 语音转录接口
- 路由：`POST /v1/audio/transcriptions`
- 参数：
  - `file`：音频或视频文件（form-data，必需）
  - `model`：模型名称（可选，默认 `whisper-1`）
  - `response_format`：响应格式（可选，默认 `json`）
    - `json`：标准 JSON 格式，包含 `text` 字段
    - `text`：纯文本格式
    - `srt`：SRT 字幕格式
    - `vtt`：VTT 字幕格式
    - `verbose_json`：详细 JSON 格式，包含分段信息
  - `language`：音频语言（可选，如 `en`）
  - `prompt`：提示文本（可选）
  - `temperature`：采样温度（可选，默认 0）
- 返回：根据 `response_format` 返回相应格式
- 完全兼容 OpenAI Whisper API 调用方式

### API 调用示例

#### 基本调用（无认证）
```bash
# 基本调用（返回 JSON）
curl -X POST http://localhost:5092/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=whisper-1"

# 获取 SRT 字幕
curl -X POST http://localhost:5092/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=whisper-1" \
  -F "response_format=srt"

# 详细 JSON 格式（包含分段）
curl -X POST http://localhost:5092/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=whisper-1" \
  -F "response_format=verbose_json"
```

#### 带 API Key 认证
```bash
# 设置 API_KEY=your-secret-key 后的调用方式
curl -X POST http://localhost:5092/v1/audio/transcriptions \
  -H "Authorization: Bearer your-secret-key" \
  -F "file=@audio.wav" \
  -F "model=whisper-1"
```

### 前端页面
- 访问 `/`，可上传音视频文件，自动转录并下载 SRT 字幕
- 支持拖拽、进度显示、结果预览与下载

## 性能优化

### 显存优化
- **FP16 半精度**：自动启用，显存占用减少约 50%
- **分块处理**：长音频自动分片，避免显存峰值过高
- **即时清理**：每个分块处理后立即清理显存缓存
- **懒加载**：按需加载模型，空闲时自动卸载

### 推荐配置
- **8GB 显存**：`CHUNK_MINITE=15`, `IDLE_TIMEOUT_MINUTES=30`
- **4GB 显存**：`CHUNK_MINITE=8`, `IDLE_TIMEOUT_MINUTES=15`
- **高并发场景**：`ENABLE_LAZY_LOAD=false`
- **资源共享**：`ENABLE_LAZY_LOAD=true`, `IDLE_TIMEOUT_MINUTES=10`

## 目录结构
```
├── app.py                # 主程序，Flask + NeMo ASR
├── requirements.txt      # 依赖列表
├── Dockerfile            # Docker 镜像构建文件
├── docker-compose.yml    # Docker Compose 部署
├── models/               # 存放 .nemo 模型文件
└── LICENSE               # MIT 开源协议
```

## 开源协议

本项目基于 MIT License 开源，详见 LICENSE 文件。

---

如有问题欢迎提交 issue 或 PR。 