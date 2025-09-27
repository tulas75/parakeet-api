# Parakeet API (Docker)

Chinese/multilingual speech recognition service based on NVIDIA NeMo, providing an OpenAI Whisper-compatible `/v1/audio/transcriptions` interface. Packaged as a GPU-enabled Docker image for one-click deployment.

- Pre-configured model: default `nvidia/parakeet-tdt-0.6b-v3`
- Supports 25 languages with automatic language detection
- Supports long audio chunking with overlapping stitching, provides SRT/VTT/verbose_json output formats
- Automatically detects CUDA compatibility: falls back to CPU mode when incompatible or no GPU is present (slower)
- OpenAI Whisper API compatible format, including error responses


## Table of Contents

- Quick Start (PowerShell)
- Prerequisites
- Running with Pre-built Images
- Building and Running from Source
- API Usage Examples
- Language Detection and Support
- Configuration and Environment Variables
- Ports, Volumes and File Structure
- Health Checks and Monitoring
- Frequently Asked Questions and Troubleshooting
- License and Acknowledgements


## Quick Start

1. Prepare directories and start the container (using pre-built image)

```bash
# Execute in the repository root directory
mkdir -p ./models ./temp_uploads

# Start (requires NVIDIA Container Toolkit installed)
docker compose up -d

# View logs (optional)
docker compose logs -f
```

1. Health checks

- Simple health: `http://localhost:5092/health/simple`
- Detailed health: `http://localhost:5092/health`

1. Test the API (example: JSON text output)

```bash
# Example using curl
curl -X POST "http://localhost:5092/v1/audio/transcriptions" \
  -F "file=@/path/to/audio.mp3" \
  -F "model=whisper-1" \
  -F "response_format=json"
```

> If API Key is enabled, add `-H "Authorization: Bearer YOUR_API_KEY"`.


## Prerequisites

- Operating System: Linux/macOS/Windows
- Docker: Docker Desktop or Docker Engine (Compose V2)
- GPU (optional but recommended):
  - NVIDIA GPU with drivers installed (recommended 535+), and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed
  - Image is based on `nvidia/cuda:13.0.0-runtime-ubuntu22.04`, requiring compatible drivers

Can also run without GPU (automatic CPU mode), but inference speed will be significantly slower.


## Running with Pre-built Images

The project provides `docker-compose.yml` which by default pulls the image `ghcr.io/fqscfqj/parakeet-api-docker:full`.

```bash
mkdir -p ./models ./temp_uploads
docker compose up -d
# To update the image
# docker compose pull; docker compose up -d
```

Compose main configuration:

- Port mapping: `5092:5092`
- Volumes:
  - `./models:/app/models` (models and cache)
  - `./temp_uploads:/app/temp_uploads` (temporary transcoding and chunking files)
- GPU: Requests all available GPUs via `deploy.resources.reservations.devices`


## Building and Running from Source

If you need to customize the Dockerfile or accelerate domestic builds, use `docker-compose-build.yml`:

```bash
mkdir -p ./models ./temp_uploads
docker compose -f docker-compose-build.yml up -d --build
```

Built image includes:

- Python 3.10 + Pip
- PyTorch/cu130 + torchaudio (from official CUDA 13.0 wheels)
- NeMo ASR and dependencies, FFmpeg, health check script


## API Usage Examples

- Endpoint: `POST /v1/audio/transcriptions`
- Fields (multipart/form-data):
  - `file`: Audio/video file
  - `model`: Compatibility field, default `whisper-1`
  - `response_format`: `json` | `text` | `srt` | `vtt` | `verbose_json`
  - `language`: Optional, default automatic
  - `prompt`, `temperature`: Optional

Example: Return SRT subtitles

```bash
curl -X POST "http://localhost:5092/v1/audio/transcriptions" \
  -F "file=@/path/to/audio.wav" \
  -F "model=whisper-1" \
  -F "response_format=srt"
```

With API Key enabled:

```bash
# After setting environment variable API_KEY in docker-compose.yml, include header when calling
curl -X POST "http://localhost:5092/v1/audio/transcriptions" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "file=@/path/to/audio.mp3" -F "response_format=json"
```

## Language Detection and Support

### Supported Languages (25)

This API supports transcription of the following 25 languages (based on parakeet-tdt-0.6b-v3 model):

| Language Code | Language Name | Language Code | Language Name | Language Code | Language Name |
|---------------|---------------|---------------|---------------|---------------|---------------|
| bg | Bulgarian | hr | Croatian | cs | Czech |
| da | Danish | nl | Dutch | en | English |
| et | Estonian | fi | Finnish | fr | French |
| de | German | el | Greek | hu | Hungarian |
| it | Italian | lv | Latvian | lt | Lithuanian |
| mt | Maltese | pl | Polish | pt | Portuguese |
| ro | Romanian | sk | Slovak | sl | Slovenian |
| es | Spanish | sv | Swedish | ru | Russian |
| uk | Ukrainian | | | | |

### Automatic Language Detection

When the `language` parameter is not specified in the request, the system automatically detects the audio language:

1. **Detection process**:
   - Extract the beginning of the audio (default 45 seconds) for quick transcription
   - Use the langdetect library to analyze the language of the transcribed text
   - If a supported language is detected, use that language for full transcription
   - If an unsupported language is detected, handle according to `ENABLE_AUTO_LANGUAGE_REJECTION` setting

2. **Processing rules**:
   - **Explicitly specified language**: Verify if the language is in the supported list; return OpenAI format error if not supported
   - **Automatically detected supported language**: Use the detected language for transcription
   - **Automatically detected unsupported language**:
     - If `ENABLE_AUTO_LANGUAGE_REJECTION=true`: return OpenAI format error
     - If `ENABLE_AUTO_LANGUAGE_REJECTION=false`: default to English for transcription

3. **Response format**:
   ```json
   {
     "text": "transcribed text content",
     "language": "auto-detected-lang-code"  // Only returned in verbose_json format
   }
   ```

4. **Configuration options**:
   - `ENABLE_AUTO_LANGUAGE_REJECTION`: Whether to reject unsupported languages (default `true`)
   - `LID_CLIP_SECONDS`: Audio clip length for language detection (default `45` seconds)

### Usage Examples

```bash
# Explicitly specify a supported language
curl -X POST "http://localhost:5092/v1/audio/transcriptions" \
  -F "file=@audio.wav" \
  -F "language=en" \
  -F "response_format=json"

# Automatically detect language
curl -X POST "http://localhost:5092/v1/audio/transcriptions" \
  -F "file=@audio.wav" \
  -F "response_format=verbose_json"  # Returns detected language

# Explicitly specify an unsupported language (returns error)
curl -X POST "http://localhost:5092/v1/audio/transcriptions" \
  -F "file=@audio.wav" \
  -F "language=zh"  # Returns OpenAI format error response
```


## Configuration and Environment Variables

Common environment variables (can be set in Compose's `environment:`):

- Model and Loading
  - `MODEL_ID`: Default `nvidia/parakeet-tdt-0.6b-v3`
  - `MODEL_LOCAL_PATH`: Priority path to load local `.nemo` file (after mounting to `./models`, can point to `/app/models/xxx.nemo`)
  - `ENABLE_LAZY_LOAD`: Whether to lazy load model (default `true`)
  - `IDLE_TIMEOUT_MINUTES`: Minutes to auto-unload model when idle, `0` to disable (default `30`)
  - `API_KEY`: If set, enables Bearer Token authentication
  - `HF_ENDPOINT`: Hugging Face mirror endpoint, default `https://hf-mirror.com`

- Performance and GPU Memory
  - `PRESET`: `speed` | `balanced` | `quality` | `simple` (=balanced). Used to derive parameters at startup
  - `GPU_VRAM_GB`: GPU memory capacity (integer, GB). If not set, will try to auto-detect
  - `CHUNK_MINUTE`: Chunk duration per segment (minutes, default `10`, can be lowered for less GPU memory)
  - `MAX_CONCURRENT_INFERENCES`: Maximum concurrent inferences (default `1`)
  - `GPU_MEMORY_FRACTION`: GPU memory fraction available to single process (default `0.90~0.95`)
  - `DECODING_STRATEGY`: `greedy` | `beam`, `RNNT_BEAM_SIZE`: Beam width
  - `AGGRESSIVE_MEMORY_CLEANUP`: Aggressive GPU memory cleanup (default `true`)
  - `ENABLE_TENSOR_CORE`, `ENABLE_CUDNN_BENCHMARK`, `TENSOR_CORE_PRECISION`: Tensor Core/Benchmark related

- Idle Resource Optimization
  - `IDLE_MEMORY_CLEANUP_INTERVAL`: Idle memory cleanup interval (seconds, default `120`)
  - `IDLE_DEEP_CLEANUP_THRESHOLD`: Deep cleanup threshold (seconds, default `600`)
  - `ENABLE_IDLE_CPU_OPTIMIZATION`: Enable CPU optimization when idle (default `true`)
  - `IDLE_MONITORING_INTERVAL`: Idle monitoring interval (seconds, default `30`)
  - `ENABLE_AGGRESSIVE_IDLE_OPTIMIZATION`: Enable aggressive memory optimization (default `true`)
  - `IMMEDIATE_CLEANUP_AFTER_REQUEST`: Immediate cleanup after request completion (default `true`)
  - `MEMORY_USAGE_ALERT_THRESHOLD_GB`: Force cleanup when memory usage exceeds this value (default `6.0`GB)
  - `AUTO_MODEL_UNLOAD_THRESHOLD_MINUTES`: Auto model unload threshold (default `10` minutes)

> 💡 **Resource Optimization Tip**: The new version significantly enhances idle optimization strategies, reducing 8GB idle memory to 2-3GB. With aggressive optimization enabled, the system performs multiple rounds of deep cleanup when the model is idle. Monitor `idle_status` and resource usage via the `/health` endpoint.

- Chunking and Sentence Integrity
  - `ENABLE_OVERLAP_CHUNKING`: Overlapping chunks (default `true`), `CHUNK_OVERLAP_SECONDS`: Overlap seconds (default `30`)
  - `ENABLE_SILENCE_ALIGNED_CHUNKING`: Silence-aligned splitting (default `true`)
  - `SILENCE_THRESHOLD_DB` (default `-38dB`), `MIN_SILENCE_DURATION` (default `0.35`), `SILENCE_MAX_SHIFT_SECONDS` (default `2.0`)

- Subtitle Post-processing and Line Breaks
  - `MERGE_SHORT_SUBTITLES` (default `true`), `MIN_SUBTITLE_DURATION_SECONDS` (default `1.5`)
  - `SHORT_SUBTITLE_MERGE_MAX_GAP_SECONDS`, `SHORT_SUBTITLE_MIN_CHARS`, `SUBTITLE_MIN_GAP_SECONDS`
  - `SPLIT_LONG_SUBTITLES` (default `true`), `MAX_SUBTITLE_DURATION_SECONDS` (default `6.0`)
  - `MAX_SUBTITLE_CHARS_PER_SEGMENT` (default `84`), `PREFERRED_LINE_LENGTH` (default `42`), `MAX_SUBTITLE_LINES` (default `2`)
  - `ENABLE_WORD_TIMESTAMPS_FOR_SPLIT` (default `false`)

- Other
  - `ENABLE_FFMPEG_DENOISE` (default `false`), `DENOISE_FILTER`: FFmpeg denoise/equalizer/dynamic range preprocessing
  - `NUMBA_CACHE_DIR` (default `/tmp/numba_cache`): Already handled and permissions assigned in image
  - `PUID` / `PGID`: Container will switch running user to specified UID/GID at startup, facilitating volume permission management

> Tip: If you just want "it to work", keep default values; if encountering GPU memory shortage, reduce `CHUNK_MINUTE`, set `PRESET=quality`, or set `DECODING_STRATEGY=greedy`.


## Ports, Volumes and File Structure

- Port: Container listens on `5092` internally, can be changed to other host ports in Compose
- Volumes:
  - `./models:/app/models`: Save/cache models (prioritizes loading `.nemo`)
  - `./temp_uploads:/app/temp_uploads`: Temporary transcoding and chunking data
- Key files:
  - `app.py`: Flask + Waitress service, providing API and chunking/post-processing logic
  - `Dockerfile`: CUDA 13.0 runtime + dependency installation + health check + startup script
  - `docker-compose.yml`: Using pre-built image
  - `docker-compose-build.yml`: Local building
  - `healthcheck.sh`: Container health check script


## Health Checks and Monitoring

- `/health/simple`: Returns 200 for alive status
- `/health`: Returns JSON with GPU/CPU, memory and model loading status, etc.
- Built-in `HEALTHCHECK` in container, Compose/Orchestration platforms can use this for restart policies


## Frequently Asked Questions and Troubleshooting (FAQ)

- Q: Log indicates "CUDA unavailable/compatibility error", service falls back to CPU?
  - A: Check that host NVIDIA drivers meet CUDA 13.x runtime requirements; confirm NVIDIA Container Toolkit is installed; verify device reservations in Compose are effective. CPU can be used when requirements cannot be met, but speed will be slower.

- Q: First startup model loading is slow or fails?
  - A: By default pulls from Hugging Face, set `MODEL_LOCAL_PATH` to point to local `.nemo`; or configure `HF_ENDPOINT` to use a mirror. Ensure `./models` volume is writable.

- Q: GPU memory shortage/frequent OOM?
  - A: Reduce `CHUNK_MINUTE` (e.g. 6~8); set `DECODING_STRATEGY=greedy`; `PRESET=quality` automatically lowers concurrency and GPU memory share; disable `ENABLE_OVERLAP_CHUNKING` if necessary.

- Q: Want to further optimize resource usage when idle?
  - A: The new version provides aggressive memory optimization strategies:
    - `ENABLE_AGGRESSIVE_IDLE_OPTIMIZATION=true`: Enable aggressive memory cleanup
    - `IMMEDIATE_CLEANUP_AFTER_REQUEST=true`: Immediate cleanup after request completion
    - `MEMORY_USAGE_ALERT_THRESHOLD_GB=6.0`: Auto force cleanup when memory exceeds 6GB
    - `AUTO_MODEL_UNLOAD_THRESHOLD_MINUTES=10`: Auto unload model after 10 minutes of idleness
    - `IDLE_MEMORY_CLEANUP_INTERVAL=120`: Perform memory cleanup every 2 minutes
    - `IDLE_DEEP_CLEANUP_THRESHOLD=600`: Perform deep cleanup after 10 minutes of idleness
    - `IDLE_MONITORING_INTERVAL=30`: Check idle status every 30 seconds

- Q: How to solve the 8GB idle memory usage issue?
  - A: Using the following environment variable configuration can significantly reduce idle memory:
    ```bash
    ENABLE_AGGRESSIVE_IDLE_OPTIMIZATION=true
    MEMORY_USAGE_ALERT_THRESHOLD_GB=4.0
    AUTO_MODEL_UNLOAD_THRESHOLD_MINUTES=5
    IDLE_MEMORY_CLEANUP_INTERVAL=60
    IMMEDIATE_CLEANUP_AFTER_REQUEST=true
    ```

- Q: Returned subtitles are too fragmented or flickering?
  - A: Adjust `MIN_SUBTITLE_DURATION_SECONDS`, `SHORT_SUBTITLE_MERGE_MAX_GAP_SECONDS`, `SHORT_SUBTITLE_MIN_CHARS`; or disable `MERGE_SHORT_SUBTITLES=false`.

- Q: Port conflict?
  - A: Modify the Compose `ports` mapping, for example `"18080:5092"`.

- Q: Permission issues (volume)?
  - A: Can set `PUID`/`PGID` or ensure Docker Desktop shared disk permissions are correct. When encountering permission restrictions, deleting and rebuilding the volume directory can also help.


## License and Acknowledgements

- This project: see `LICENSE`
- Models and dependencies: NVIDIA NeMo (ASR), PyTorch, FFmpeg, Hugging Face and other open-source ecosystems


---
Completion and Verification

- Build: Docker/Compose manifests ready
- Run: Provides GPU/CPU dual paths and health checks
- Usage: Provides cross-platform commands and curl examples
- Coverage: Added user-friendly README (this file)
