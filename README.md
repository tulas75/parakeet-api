# Parakeet-API-Docker

> 本项目 fork 自 [jianchang512/parakeet-api](https://github.com/jianchang512/parakeet-api)，感谢原作者的贡献！

基于 NVIDIA NeMo 的 Parakeet-tdt-0.6b-v2 语音识别 API，支持长音频分片转录，自动生成 SRT 字幕，兼容 OpenAI Whisper API，内置简洁前端上传页面。

## 主要特性
- 支持长音频/视频自动分片转录，显存占用低
- **FP16 半精度优化**：显存占用减少近一半，推理速度更快
- **智能显存管理**：懒加载模式 + 自动模型卸载，空闲时释放显存
- **激进显存优化**：动态调整chunk大小、实时清理缓存、梯度检查点
- **可配置音频分块**：根据显存大小调整分块时长，支持更长上下文
- **显存监控**：实时显示显存使用情况，自动进行内存管理
- **API Key 认证**：支持 Whisper 兼容的 Bearer Token 认证
- 输出 SRT 字幕格式，带时间戳
- 兼容 OpenAI Whisper 语音识别 API 路由
- 前端页面支持拖拽上传、进度显示、结果下载
- 支持 Docker 一键部署，自动挂载模型和临时目录
- 支持 GPU 加速（需 NVIDIA 驱动和 CUDA 环境）
 - **静音对齐切片**：目标切片边界自动对齐最近静音，避免在句中硬切
 - **可选去噪前处理**：一行开关 `ffmpeg` 温和降噪/均衡/动态范围，提升嘈杂环境识别率
 - **改进重叠去重**：更鲁棒的重叠段合并与去重，减少重复和缺字
 - **可选 Beam Search**：若模型支持，启用小规模 beam 提升解码稳定性
 - **显存利用优化**：仅按需请求时间戳、推理并发管控、可调 DataLoader/Batch、可配置 GPU 内存比例

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

通过 `docker-compose.yml` 中的环境变量可以灵活配置服务行为（简版，推荐）：

```yaml
environment:
  - ENABLE_LAZY_LOAD=true        # 懒加载开关，true=按需加载，false=启动时预加载
  - DECODING_STRATEGY=greedy     # 默认使用 greedy，避免 TDT Beam 在 timestamps 下的对齐限制
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

### 全量配置（进阶，可选）

以下为可选的完整环境变量列表（如不需要请忽略，保持简版即可）：

- **AGGRESSIVE_MEMORY_CLEANUP**：激进显存清理（默认：`true`）
  - `true`：启用激进的显存清理，每个chunk处理完后立即清理
  - `false`：使用标准清理策略，可能占用更多显存但性能稍好

- **ENABLE_GRADIENT_CHECKPOINTING**：梯度检查点（默认：`true`）
  - `true`：启用梯度检查点，显著降低显存占用
  - `false`：禁用梯度检查点，可能占用更多显存

- **FORCE_CLEANUP_THRESHOLD**：强制清理阈值（默认：`0.8`）
  - 显存使用超过此比例时强制清理
  - 范围：0.0-1.0，例如 0.8 表示 80%

- **MAX_CHUNK_MEMORY_MB**：单chunk最大显存占用（默认：`1500`MB）
  - 用于监控和调整处理策略
  - 根据实际GPU显存调整

配置示例（进阶）：
```yaml
environment:
  - AGGRESSIVE_MEMORY_CLEANUP=true
  - ENABLE_GRADIENT_CHECKPOINTING=true  
  - FORCE_CLEANUP_THRESHOLD=0.7
  - MAX_CHUNK_MEMORY_MB=1200
```

### Tensor Core 优化配置（可选）

新增专门的 Tensor Core 优化环境变量：

- **ENABLE_TENSOR_CORE**：启用Tensor Core（默认：`true`）
  - `true`：启用TF32和Tensor Core优化，大幅提升FP16推理速度
  - `false`：禁用Tensor Core，使用传统CUDA核心

- **ENABLE_CUDNN_BENCHMARK**：cuDNN基准测试（默认：`true`）
  - `true`：启用cuDNN自动调优，首次运行较慢但后续更快
  - `false`：禁用自动调优，确保结果完全一致

- **TENSOR_CORE_PRECISION**：Tensor Core精度模式（默认：`highest`）
  - `highest`：最高精度，适合对准确度要求极高的场景
  - `high`：高精度，平衡精度和性能  
  - `medium`：中等精度，最大化性能

配置示例（Docker）：
```yaml
environment:
  - ENABLE_TENSOR_CORE=true
  - ENABLE_CUDNN_BENCHMARK=true
  - TENSOR_CORE_PRECISION=high
```

### GPU 兼容性说明

**完全支持 Tensor Core:**
- RTX 20/30/40 系列 (Turing/Ampere/Ada)
- Tesla V100, A100, H100
- Quadro RTX 系列
- 计算能力 ≥ 7.0

**部分支持:**  
- GTX 16 系列 (有限的Tensor操作)
- Tesla P100 (计算能力 6.0)

**不支持:**
- GTX 10 系列及更早
- 计算能力 < 6.0

### 显存占用优化效果

通过这些优化，8分钟音频段的显存占用从原来的8GB降低到约2-3GB，同时：

**性能提升：**
- Tensor Core 加速：2-4x FP16推理速度提升
- cuDNN 优化：10-20% 额外性能提升  
- 内存优化：60-70% 显存占用减少

**保持质量：**
- FP16半精度推理精度
- 上下文连贯性
- 时间戳准确性
- API兼容性

### 句子完整性优化配置（可选）

解决分块处理中句子被截断的问题：

- **ENABLE_OVERLAP_CHUNKING**：重叠分割（默认：`true`）
  - `true`：启用重叠分割，确保句子完整性
  - `false`：使用传统硬分割，可能截断句子

- **CHUNK_OVERLAP_SECONDS**：重叠时长（默认：`30`秒）
  - 每个chunk之间的重叠时间
  - 更长重叠提供更好的上下文，但增加计算量

- **SENTENCE_BOUNDARY_THRESHOLD**：句子边界阈值（默认：`0.5`）
  - 用于检测最佳分割点的时间容忍度
  - 较小值提供更精确的句子边界检测

配置示例（Docker）：
```yaml
environment:
  - ENABLE_OVERLAP_CHUNKING=true
  - CHUNK_OVERLAP_SECONDS=30
  - SENTENCE_BOUNDARY_THRESHOLD=0.5
```

### 静音对齐与音频前处理（可选）

新增提升准确率的开关：

```yaml
environment:
  # 静音对齐切片（默认启用）
  - ENABLE_SILENCE_ALIGNED_CHUNKING=true
  - SILENCE_THRESHOLD_DB=-38dB        # 静音判定阈值（dB），绝对值越大越容易判静音
  - MIN_SILENCE_DURATION=0.35         # 静音最小时长（秒）
  - SILENCE_MAX_SHIFT_SECONDS=2.0     # 分割点向静音对齐的最大偏移（秒）

  # ffmpeg 预处理（默认关闭）
  - ENABLE_FFMPEG_DENOISE=false
  - DENOISE_FILTER=afftdn=nf=-25,highpass=f=50,lowpass=f=8000,dynaudnorm=m=7:s=5

  # 解码策略（若模型支持）
  - DECODING_STRATEGY=beam            # greedy | beam
  - RNNT_BEAM_SIZE=4
```

### 字幕后处理：最小时长与短字幕合并（可选）

为降低“闪字幕”（显示时间过短）现象，已内置后处理，默认开启。可通过以下变量微调：

- `MERGE_SHORT_SUBTITLES`：是否启用短字幕合并与延长（默认 `true`）
- `MIN_SUBTITLE_DURATION_SECONDS`：每条字幕的最短显示时长，默认 `1.5` 秒
- `SHORT_SUBTITLE_MERGE_MAX_GAP_SECONDS`：与下一条字幕的最大可合并间隙，默认 `0.3` 秒
- `SHORT_SUBTITLE_MIN_CHARS`：当文本字符数不超过此值时更倾向合并，默认 `6`
- `SUBTITLE_MIN_GAP_SECONDS`：段与段之间保留的最小安全间隔，默认 `0.06` 秒

推荐：若仍觉得跳动，可将 `MIN_SUBTITLE_DURATION_SECONDS` 调至 `1.8–2.2`；若误合并，略微减小 `SHORT_SUBTITLE_MERGE_MAX_GAP_SECONDS`。

示例（docker-compose 环境块中加入）：

```yaml
environment:
  - MERGE_SHORT_SUBTITLES=true
  - MIN_SUBTITLE_DURATION_SECONDS=1.8
  - SHORT_SUBTITLE_MERGE_MAX_GAP_SECONDS=0.25
  - SHORT_SUBTITLE_MIN_CHARS=6
  - SUBTITLE_MIN_GAP_SECONDS=0.06
```

### 快速变量维护指南（推荐）

面向大多数用户，只需维护 2–3 个变量即可：`PRESET`、`GPU_VRAM_GB`，必要时加 `CHUNK_MINITE`。其它保持默认。

#### 一分钟选型
- 4–6GB：`PRESET=speed`，`GPU_VRAM_GB=6`，`CHUNK_MINITE=3`
- 8GB（保守安全）：`PRESET=balanced`，`GPU_VRAM_GB=8`，`CHUNK_MINITE=5`，`DECODING_STRATEGY=greedy`
- ≥12GB：`PRESET=balanced`（或 `quality`），`GPU_VRAM_GB=12`，可不设 `CHUNK_MINITE`；若追求更稳可改 `DECODING_STRATEGY=beam` 并配合 `RNNT_BEAM_SIZE=4`

#### 最小环境块（8GB 稳妥组合，复制即用）
```yaml
environment:
  - PRESET=balanced        # speed | balanced | quality
  - GPU_VRAM_GB=8          # 显存大小（整数）
  - CHUNK_MINITE=5         # 8GB 建议 5 分钟，防止 OOM
  - DECODING_STRATEGY=greedy
  - MAX_CONCURRENT_INFERENCES=1
  - TRANSCRIBE_BATCH_SIZE=1
  - TRANSCRIBE_NUM_WORKERS=0
  - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:96,garbage_collection_threshold:0.85
```

#### 常用变量（只看这几个）
- **PRESET**: 一键风格。speed=更快更省显存；balanced=默认；quality=更稳但慢。
- **GPU_VRAM_GB**: 显存大小（整数），用于自动推导其它参数。
- **CHUNK_MINITE**: 单段转录时长，越大显存峰值越高。4–8GB 建议 3–6；8GB 推荐 5。
- **DECODING_STRATEGY**: `greedy` 显存更低；`beam` 更稳（配合 `RNNT_BEAM_SIZE=2–6`）。
- **MAX_CONCURRENT_INFERENCES**: 并发数。8GB 固定 1。
- **TRANSCRIBE_BATCH_SIZE / TRANSCRIBE_NUM_WORKERS**: 建议 1 / 0 最省显存。
- **PYTORCH_CUDA_ALLOC_CONF**: 建议保留以减少显存碎片（镜像已默认设置，显式声明可覆盖）。
- 可选质量/连贯：`ENABLE_OVERLAP_CHUNKING=true`、`CHUNK_OVERLAP_SECONDS=30`、`ENABLE_SILENCE_ALIGNED_CHUNKING=true`。

#### 调优口诀
- **OOM**：先把 `CHUNK_MINITE` 每次减 2 分钟；仍 OOM 再确认并发=1，并使用 `greedy` 或将 `RNNT_BEAM_SIZE` 降到 2。
- **太慢**：`PRESET=speed` 或适度增大 `CHUNK_MINITE`（注意显存）。
- **质量不稳**：改 `DECODING_STRATEGY=beam` 且 `RNNT_BEAM_SIZE=4`（显存允许时）。

#### 重载生效
更新 `docker-compose.yml` 后执行：

```bash
docker compose down
docker compose up -d --force-recreate
docker logs -f parakeet-api-docker
```

日志会打印最终推导值与 GPU 总显存，便于核对。遇到 OOM 时，通常只需调整 `CHUNK_MINITE` 即可。

### 一键预设（简化环境变量，按需）

最少只需要 1-2 个变量：

```yaml
environment:
  - PRESET=balanced          # speed | balanced | quality
  - GPU_VRAM_GB=8           # 可选，设置你显卡显存（GB），不填则自动探测
```

解释：
- **PRESET**
  - speed：优先速度，使用 greedy 解码，适度增大并发（高显存），分块稍大
  - balanced（默认）：准确与速度平衡，beam 小束宽，分块适中
  - quality：优先准确，使用 beam 解码，小并发，分块稍小
- **GPU_VRAM_GB**：设置后更精确；不设置则会尝试自动探测 GPU 显存

如需细调，仍可使用以下可选变量（有默认，不需要必填）：
```yaml
environment:
  - CHUNK_MINITE=10
  - MAX_CONCURRENT_INFERENCES=1
  - GPU_MEMORY_FRACTION=0.92
  - DECODING_STRATEGY=beam    # greedy | beam
  - RNNT_BEAM_SIZE=4
  - TRANSCRIBE_BATCH_SIZE=1
  - TRANSCRIBE_NUM_WORKERS=0
```

说明：
- **静音对齐切片**：在每个切片起点附近寻找最近静音边界，对齐后再裁切，尽量避免在句中间硬切导致的误识别与重复。
- **DENOISE_FILTER**：默认是一组较温和的降噪/滤波/动态范围归一参数，建议在嘈杂环境或强底噪场景开启。
- **Beam Search**：对支持 `change_decoding_strategy` 的 NeMo 模型生效，`beam_size` 建议 2-6 之间平衡延迟与质量。

### 句子完整性原理

**问题：** 传统硬分割可能在句子中间切断音频，导致：
- 句子前半部分在chunk1末尾被截断
- 句子后半部分在chunk2开头丢失上下文
- 影响转录准确性和连贯性

**解决方案：**
1. **重叠分割**：每个chunk包含前一个chunk的最后30秒
2. **句子边界检测**：在重叠区域寻找句子结束点
3. **智能合并**：去除重叠区域的重复内容，保持句子完整

**优势：**
- ✅ 保证句子完整性
- ✅ 维持上下文连贯性  
- ✅ 提高转录准确率
- ✅ 保持时间戳精确性
- ⚠️ 轻微增加计算开销（约10-15%）

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