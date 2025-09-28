import os,sys,json,math

# Set environment variables to solve numba cache issues
os.environ['NUMBA_CACHE_DIR'] = '/tmp/numba_cache'
os.environ['NUMBA_DISABLE_JIT'] = '0'

# Set matplotlib config directory to avoid permission issues
# Prioritize using directory set by startup script, fallback to backup directory if not exists
if 'MPLCONFIGDIR' not in os.environ:
    os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib_config'
    os.makedirs('/tmp/matplotlib_config', exist_ok=True)
    os.chmod('/tmp/matplotlib_config', 0o777)
else:
    # Ensure set directory exists and has correct permissions
    mpl_dir = os.environ['MPLCONFIGDIR']
    try:
        os.makedirs(mpl_dir, exist_ok=True)
        os.chmod(mpl_dir, 0o755)
    except (PermissionError, OSError):
        # If unable to create or set permissions, fallback to tmp directory
        os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib_config'
        os.makedirs('/tmp/matplotlib_config', exist_ok=True)
        os.chmod('/tmp/matplotlib_config', 0o777)

host = '0.0.0.0'
port = 5092
# Default thread count is more memory-efficient; increase for concurrency if needed
threads = int(os.environ.get('APP_THREADS', '2'))
# By default, cut audio/video into segments every N minutes to reduce GPU memory usage. Can now be adjusted via CHUNK_MINITE environment variable.
# For 8GB GPU memory, it's recommended to set to 10-15 minutes for optimal performance.
CHUNK_MINITE = int(os.environ.get('CHUNK_MINITE', '10'))
# Automatically unload model after service is idle for N minutes to free GPU memory; set to 0 to disable (default 30 minutes)
IDLE_TIMEOUT_MINUTES = int(os.environ.get('IDLE_TIMEOUT_MINUTES', '30'))
# Lazy load toggle, defaults to true. Set to 'false' to preload model on startup.
ENABLE_LAZY_LOAD = os.environ.get('ENABLE_LAZY_LOAD', 'true').lower() not in ['false', '0', 'f']
# Whisper-compatible API Key. If left empty, no authentication is performed.
API_KEY = os.environ.get('API_KEY', None)
import shutil
from typing import Any, Dict
import uuid
import subprocess
import datetime
import threading
import time
from werkzeug.utils import secure_filename

from flask import Flask, request, jsonify, Response
from waitress import serve
from pathlib import Path
# ROOT_DIR is not needed in Docker environment
# Only set HF mirror when not explicitly configured (can be overridden via environment variables)
if 'HF_ENDPOINT' not in os.environ:
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# HF_HOME is set in the Dockerfile
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = 'true'
# PATH for ffmpeg is handled by the Docker image's system PATH

# Reduce PyTorch CUDA allocation fragmentation, lowering OOM chances (can be overridden via external environment variables)
if 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'

import nemo.collections.asr as nemo_asr  # type: ignore
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import gc
import psutil
import ctypes
import ctypes.util
try:
    # huggingface_hub may not be present in the editor environment; import defensively
    from huggingface_hub import HfApi, hf_hub_download  # type: ignore
except Exception:
    # Provide fallbacks so static checkers and runtime in minimal environments won't crash.
    HfApi = None  # type: ignore
    def hf_hub_download(*args, **kwargs):
        raise RuntimeError("huggingface_hub is not installed")

# --- Global Settings and Model State ---
asr_model = None
last_request_time = None
model_lock = threading.Lock()
cuda_available = False  # Global CUDA compatibility flag

# Supported languages (ISO 639-1, two-letter lowercase), based on parakeet-tdt-0.6b-v3 announcement
SUPPORTED_LANG_CODES = {
    'bg','hr','cs','da','nl','en','et','fi','fr','de','el','hu','it','lv','lt','mt','pl','pt','ro','sk','sl','es','sv','ru','uk'
}

# Automatic language rejection (when language is not explicitly passed, first perform language detection on short segments; return Whisper-style error if not supported)
ENABLE_AUTO_LANGUAGE_REJECTION = os.environ.get('ENABLE_AUTO_LANGUAGE_REJECTION', 'true').lower() in ['true', '1', 't']
LID_CLIP_SECONDS = int(os.environ.get('LID_CLIP_SECONDS', '45'))

# Inference concurrency control (avoid multiple requests simultaneously occupying GPU memory causing OOM)
MAX_CONCURRENT_INFERENCES = int(os.environ.get('MAX_CONCURRENT_INFERENCES', '1'))
inference_semaphore = threading.Semaphore(MAX_CONCURRENT_INFERENCES)

# GPU memory optimization configuration
AGGRESSIVE_MEMORY_CLEANUP = os.environ.get('AGGRESSIVE_MEMORY_CLEANUP', 'true').lower() in ['true', '1', 't']
ENABLE_GRADIENT_CHECKPOINTING = os.environ.get('ENABLE_GRADIENT_CHECKPOINTING', 'true').lower() in ['true', '1', 't']
MAX_CHUNK_MEMORY_MB = int(os.environ.get('MAX_CHUNK_MEMORY_MB', '1500'))
FORCE_CLEANUP_THRESHOLD = float(os.environ.get('FORCE_CLEANUP_THRESHOLD', '0.8'))
ENABLE_MALLOC_TRIM = os.environ.get('ENABLE_MALLOC_TRIM', 'true').lower() in ['true', '1', 't']

# Idle resource optimization configuration
IDLE_MEMORY_CLEANUP_INTERVAL = int(os.environ.get('IDLE_MEMORY_CLEANUP_INTERVAL', '120'))  # Memory cleanup interval during idle (seconds), default 2 minutes
IDLE_DEEP_CLEANUP_THRESHOLD = int(os.environ.get('IDLE_DEEP_CLEANUP_THRESHOLD', '600'))  # Deep cleanup threshold (seconds), default 10 minutes
ENABLE_IDLE_CPU_OPTIMIZATION = os.environ.get('ENABLE_IDLE_CPU_OPTIMIZATION', 'true').lower() in ['true', '1', 't']
IDLE_MONITORING_INTERVAL = int(os.environ.get('IDLE_MONITORING_INTERVAL', '30'))  # Idle monitoring interval (seconds), default 30 seconds
# Memory optimization configuration - simplified to reasonable default values, no user configuration needed
ENABLE_AGGRESSIVE_IDLE_OPTIMIZATION = os.environ.get('ENABLE_AGGRESSIVE_IDLE_OPTIMIZATION', 'true').lower() in ['true', '1', 't']
IMMEDIATE_CLEANUP_AFTER_REQUEST = os.environ.get('IMMEDIATE_CLEANUP_AFTER_REQUEST', 'true').lower() in ['true', '1', 't']
MEMORY_USAGE_ALERT_THRESHOLD_GB = float(os.environ.get('MEMORY_USAGE_ALERT_THRESHOLD_GB', '12.0'))  # Set higher threshold to avoid frequent cleanup
AUTO_MODEL_UNLOAD_THRESHOLD_MINUTES = int(os.environ.get('AUTO_MODEL_UNLOAD_THRESHOLD_MINUTES', '30'))  # Keep consistent with IDLE_TIMEOUT_MINUTES

# Tensor Core optimization configuration
ENABLE_TENSOR_CORE = os.environ.get('ENABLE_TENSOR_CORE', 'true').lower() in ['true', '1', 't']
ENABLE_CUDNN_BENCHMARK = os.environ.get('ENABLE_CUDNN_BENCHMARK', 'true').lower() in ['true', '1', 't']
TENSOR_CORE_PRECISION = os.environ.get('TENSOR_CORE_PRECISION', 'highest')  # highest, high, medium
GPU_MEMORY_FRACTION = float(os.environ.get('GPU_MEMORY_FRACTION', '0.95'))  # GPU memory fraction allowed for process

# Sentence integrity optimization configuration
ENABLE_OVERLAP_CHUNKING = os.environ.get('ENABLE_OVERLAP_CHUNKING', 'true').lower() in ['true', '1', 't']
CHUNK_OVERLAP_SECONDS = float(os.environ.get('CHUNK_OVERLAP_SECONDS', '30'))  # Overlap duration
SENTENCE_BOUNDARY_THRESHOLD = float(os.environ.get('SENTENCE_BOUNDARY_THRESHOLD', '0.5'))  # Sentence boundary detection threshold


# Silence-aligned slicing and preprocessing configuration
ENABLE_SILENCE_ALIGNED_CHUNKING = os.environ.get('ENABLE_SILENCE_ALIGNED_CHUNKING', 'true').lower() in ['true', '1', 't']
SILENCE_THRESHOLD_DB = os.environ.get('SILENCE_THRESHOLD_DB', '-38dB')  # ffmpeg silencedetect noise threshold
MIN_SILENCE_DURATION = float(os.environ.get('MIN_SILENCE_DURATION', '0.35'))  # Minimum duration considered as silence (seconds)
SILENCE_MAX_SHIFT_SECONDS = float(os.environ.get('SILENCE_MAX_SHIFT_SECONDS', '2.0'))  # Maximum offset allowed for alignment to silence near target split point (seconds)

ENABLE_FFMPEG_DENOISE = os.environ.get('ENABLE_FFMPEG_DENOISE', 'false').lower() in ['true', '1', 't']
# Reasonable default denoise/equalizer/dynamic range settings, as gentle as possible to avoid overfitting
DENOISE_FILTER = os.environ.get(
    'DENOISE_FILTER',
    'afftdn=nf=-25,highpass=f=50,lowpass=f=8000,dynaudnorm=m=7:s=5'
)

# Decoding strategy (if model supports)
DECODING_STRATEGY = os.environ.get('DECODING_STRATEGY', 'greedy')  # Options: greedy, beam
RNNT_BEAM_SIZE = int(os.environ.get('RNNT_BEAM_SIZE', '4'))

# Nemo transcription runtime configuration (batch and DataLoader)
TRANSCRIBE_BATCH_SIZE = int(os.environ.get('TRANSCRIBE_BATCH_SIZE', '1'))
TRANSCRIBE_NUM_WORKERS = int(os.environ.get('TRANSCRIBE_NUM_WORKERS', '0'))

# Subtitle post-processing configuration (to prevent subtitles from displaying too briefly)
MERGE_SHORT_SUBTITLES = os.environ.get('MERGE_SHORT_SUBTITLES', 'true').lower() in ['true', '1', 't']
MIN_SUBTITLE_DURATION_SECONDS = float(os.environ.get('MIN_SUBTITLE_DURATION_SECONDS', '1.5'))
SHORT_SUBTITLE_MERGE_MAX_GAP_SECONDS = float(os.environ.get('SHORT_SUBTITLE_MERGE_MAX_GAP_SECONDS', '0.3'))
SHORT_SUBTITLE_MIN_CHARS = int(os.environ.get('SHORT_SUBTITLE_MIN_CHARS', '6'))
SUBTITLE_MIN_GAP_SECONDS = float(os.environ.get('SUBTITLE_MIN_GAP_SECONDS', '0.06'))

# Long subtitle splitting and line breaks (optional)
# - Split overly long/long-duration subtitles into multiple entries; also wrap text within each subtitle for easier viewing
SPLIT_LONG_SUBTITLES = os.environ.get('SPLIT_LONG_SUBTITLES', 'true').lower() in ['true', '1', 't']
MAX_SUBTITLE_DURATION_SECONDS = float(os.environ.get('MAX_SUBTITLE_DURATION_SECONDS', '6.0'))
MAX_SUBTITLE_CHARS_PER_SEGMENT = int(os.environ.get('MAX_SUBTITLE_CHARS_PER_SEGMENT', '84'))  # Approximately two lines, ~42 per line
PREFERRED_LINE_LENGTH = int(os.environ.get('PREFERRED_LINE_LENGTH', '42'))
MAX_SUBTITLE_LINES = int(os.environ.get('MAX_SUBTITLE_LINES', '2'))
# If true, try to use word-level timestamps for more precise splitting (automatically fallback if model doesn't return words)
ENABLE_WORD_TIMESTAMPS_FOR_SPLIT = os.environ.get('ENABLE_WORD_TIMESTAMPS_FOR_SPLIT', 'false').lower() in ['true', '1', 't']
# Prioritize splitting by punctuation: comma/period/question mark/exclamation mark/semicolon, etc.
SUBTITLE_SPLIT_PUNCTUATION = os.environ.get('SUBTITLE_SPLIT_PUNCTUATION', '。！？!?.,;；，,')

# Simplified configuration: presets and GPU VRAM (GB)
PRESET = os.environ.get('PRESET', 'balanced').lower()  # speed | balanced | quality | simple(=balanced)
GPU_VRAM_GB_ENV = os.environ.get('GPU_VRAM_GB', '').strip()


# Ensure temporary upload directory exists
if not os.path.exists('./app/temp_uploads'):
    os.makedirs('./app/temp_uploads')

def setup_tensor_core_optimization():
    """Configure Tensor Core optimization settings"""
    global cuda_available
    if not cuda_available:
        print("CUDA unavailable, skipping Tensor Core optimization configuration")
        return
    
    print("Configuring Tensor Core optimization...")
    
    try:
        # Enable cuDNN benchmark mode
        if ENABLE_CUDNN_BENCHMARK:
            cudnn.benchmark = True
            cudnn.deterministic = False  # Allow nondeterministic for performance
            print("✅ cuDNN benchmark enabled")
        else:
            cudnn.benchmark = False
            cudnn.deterministic = True
            print("❌ cuDNN benchmark disabled (deterministic mode)")
        
        # Enable cuDNN to allow TensorCore
        if ENABLE_TENSOR_CORE:
            cudnn.allow_tf32 = True  # Allow TF32 (supported by A100, etc.)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("✅ Tensor Core (TF32) enabled")
        else:
            cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            print("❌ Tensor Core disabled")
        
        # Set Tensor Core precision strategy
        if TENSOR_CORE_PRECISION == 'highest':
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
            print("✅ Set to highest precision mode")
        elif TENSOR_CORE_PRECISION == 'high':
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
            print("✅ Set to high precision mode")
        else:  # medium
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
            print("✅ Set to medium precision mode")
        
        # Set memory allocation strategy to optimize Tensor Core usage
        try:
            torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION)
            print(f"✅ GPU memory allocation ratio: {GPU_MEMORY_FRACTION*100:.0f}%")
        except Exception as e:
            print(f"⚠️ Failed to set memory allocation ratio: {e}")
        print("✅ GPU memory allocation strategy optimized")
    except Exception as e:
        print(f"⚠️ Tensor Core optimization configuration failed: {e}")

def get_tensor_core_info():
    """Get Tensor Core support information"""
    global cuda_available
    if not cuda_available:
        return "N/A - CUDA unavailable"
    
    try:
        device = torch.cuda.get_device_properties(0)
        major, minor = device.major, device.minor
        
        # Detect Tensor Core support
        if major >= 7:  # V100, T4, RTX 20/30/40 series, etc.
            if major == 7:
                return f"✅ Tensor Core 1.0 (compute capability {major}.{minor})"
            elif major == 8:
                if minor >= 0:
                    return f"✅ Tensor Core 2.0 + TF32 (compute capability {major}.{minor})"
                else:
                    return f"✅ Tensor Core 2.0 (compute capability {major}.{minor})"
            elif major >= 9:
                return f"✅ Tensor Core 3.0+ (compute capability {major}.{minor})"
        elif major >= 6:  # P100, etc.
            return f"⚠️ Limited Tensor Core support (compute capability {major}.{minor})"
        else:
            return f"❌ Tensor Core not supported (compute capability {major}.{minor})"
        
        return f"Unknown (compute capability {major}.{minor})"
    except Exception as e:
        return f"❌ Failed to get GPU information: {e}"

def optimize_tensor_operations():
    """Optimize tensor operations to better utilize Tensor Core"""
    global cuda_available
    if not cuda_available:
        print("CUDA unavailable, skipping Tensor Core warmup")
        return
    
    try:
        # Set optimized CUDA streams
        torch.cuda.set_sync_debug_mode(0)  # Disable sync debugging to improve performance
        
        # Warm up GPU, ensuring Tensor Core is properly activated
        # Create some matrices aligned to 8/16 multiples for warmup
        device = torch.cuda.current_device()
        dummy_a = torch.randn(128, 128, device=device, dtype=torch.float16)
        dummy_b = torch.randn(128, 128, device=device, dtype=torch.float16)
        
        # Execute matrix multiplication to warm up Tensor Core
        with torch.cuda.amp.autocast():
            _ = torch.matmul(dummy_a, dummy_b)
        
        torch.cuda.synchronize()
        del dummy_a, dummy_b
        torch.cuda.empty_cache()
        print("✅ Tensor Core warmup completed")
    except Exception as e:
        print(f"⚠️ Tensor Core warmup failed: {e}")

def detect_sentence_boundaries(text: str) -> list:
    """Detect sentence boundaries, return list of sentence end positions"""
    import re
    
    # Chinese/English periods, question marks, exclamation marks, etc.
    sentence_endings = re.finditer(r'[.!?。！？]+[\s]*', text)
    boundaries = [match.end() for match in sentence_endings]
    return boundaries

def find_best_split_point(segments: list, target_time: float, tolerance: float = 2.0) -> int:
    """Find the best sentence split point near the target time"""
    if not segments:
        return 0
    
    best_index = 0
    min_distance = float('inf')
    
    # Find the sentence end point closest to target time
    for i, segment in enumerate(segments):
        segment_end = segment.get('end', 0)
        distance = abs(segment_end - target_time)
        
        # Check if it's a sentence end (contains punctuation)
        text = segment.get('segment', '').strip()
        if text and (text.endswith('.') or text.endswith('。') or 
                     text.endswith('!') or text.endswith('！') or
                     text.endswith('?') or text.endswith('？')):
            # Sentence end point, weight is higher
            distance *= 0.5
        
        if distance < min_distance and distance <= tolerance:
            min_distance = distance
            best_index = i + 1  # Return next paragraph's index
    
    return min(best_index, len(segments))

def merge_overlapping_segments(all_segments: list, chunk_boundaries: list, overlap_seconds: float) -> list:
    """Merge overlapping segments, remove duplicate content"""
    if not ENABLE_OVERLAP_CHUNKING or len(chunk_boundaries) <= 1:
        return all_segments
    
    # Simplified and more robust: sort by time, then deduplicate same-text segments based on overlap window
    if not all_segments:
        return []
    all_segments_sorted = sorted(all_segments, key=lambda s: (s.get('start', 0.0), s.get('end', 0.0)))
    merged = []
    for seg in all_segments_sorted:
        text = seg.get('segment', '').strip()
        if not text:
            continue
        if not merged:
            merged.append(seg)
            continue
        prev = merged[-1]
        # If highly overlapping in time and text is highly similar (or identical), keep the longer/higher-confidence one
        overlap = min(prev['end'], seg['end']) - max(prev['start'], seg['start'])
        window = overlap_seconds * 0.9 if overlap_seconds else 0.0
        def normalized(t: str) -> str:
            return ''.join(t.split()).lower()
        same_text = normalized(prev.get('segment', '')) == normalized(text)
        if overlap > 0 and overlap >= min(prev['end'] - prev['start'], seg['end'] - seg['start']) * 0.5:
            if same_text or overlap >= window:
                # Choose the segment with longer duration
                if (prev['end'] - prev['start']) >= (seg['end'] - seg['start']):
                    # Possibly extend the end
                    prev['end'] = max(prev['end'], seg['end'])
                else:
                    merged[-1] = seg
                continue
        # Otherwise directly append
        merged.append(seg)
    print(f"Merging completed, final {len(merged)} segments")
    return merged

def enforce_min_subtitle_duration(
    segments: list,
    min_duration: float,
    merge_max_gap: float,
    min_chars: int,
    min_gap: float,
) -> list:
    """Post-process the transcribed segments to avoid subtitles displaying too briefly:
    1) Try to merge adjacent segments that are too short or have too little text (gap between segments does not exceed merge_max_gap).
    2) If still shorter than min_duration, try to extend the current segment's end time to min_duration, but do not overlap with the next segment (reserve min_gap).

    segments: [{'start': float, 'end': float, 'segment': str}, ...]
    returns: processed segments (sorted by start time, and non-overlapping)
    """
    if not segments:
        return []

    # Sort by start time, deep copy to avoid modifying original object
    segments_sorted = sorted(
        [
            {
                'start': float(s.get('start', 0.0)),
                'end': float(s.get('end', 0.0)),
                'segment': str(s.get('segment', '')),
            }
            for s in segments
        ],
        key=lambda s: (s['start'], s['end'])
    )

    result: list = []
    i = 0
    n = len(segments_sorted)

    while i < n:
        current = segments_sorted[i]
        current_text = str(current.get('segment', '')).strip()

        # Try forward merging until minimum duration is satisfied or no more mergeable objects
        while MERGE_SHORT_SUBTITLES:
            duration = max(0.0, float(current.get('end', 0.0)) - float(current.get('start', 0.0)))
            too_short = duration < min_duration or len(current_text) <= min_chars
            if not too_short or i + 1 >= n:
                break
            next_seg = segments_sorted[i + 1]
            gap = max(0.0, float(next_seg.get('start', 0.0)) - float(current.get('end', 0.0)))
            if gap > merge_max_gap:
                break
            # Merge to current
            next_text = str(next_seg.get('segment', '')).strip()
            current['end'] = max(float(current.get('end', 0.0)), float(next_seg.get('end', 0.0)))
            current_text = (current_text + ' ' + next_text).strip()
            current['segment'] = current_text
            i += 1  # Swallow the next segment
        # After merging, if still short, try to extend, but must not overlap with next segment
        duration = max(0.0, float(current.get('end', 0.0)) - float(current.get('start', 0.0)))
        if duration < float(min_duration):
            desired_end = float(current.get('start', 0.0)) + float(min_duration)
            if i + 1 < n:
                next_start = float(segments_sorted[i + 1].get('start', 0.0))
                safe_end = max(float(current.get('end', 0.0)), min(desired_end, next_start - float(min_gap)))
                # Only update if it doesn't result in an invalid interval
                if safe_end > float(current.get('start', 0.0)):
                    current['end'] = safe_end
            else:
                # Already the last segment, extend directly
                current['end'] = desired_end

        result.append(current)
        i += 1

    # Finally ensure no overlaps and monotonic increase
    cleaned: list = []
    for seg in result:
        if not cleaned:
            cleaned.append(seg)
            continue
        prev = cleaned[-1]
        if seg['start'] < prev['end']:
            seg['start'] = prev['end'] + min_gap
            if seg['start'] > seg['end']:
                seg['start'] = seg['end']
        cleaned.append(seg)

    return cleaned

def process_chunk_segments(segments: list, overlap_start: float, overlap_seconds: float) -> list:
    """处理单个chunk的segments，处理重叠区域"""
    if not segments:
        return []
    
    processed = []
    overlap_end = overlap_start + overlap_seconds
    
    for segment in segments:
        segment_start = segment['start']
        segment_end = segment['end']
        
        # 如果segment完全在重叠区域之前，直接添加
        if segment_end <= overlap_start:
            processed.append(segment)
        # 如果segment跨越重叠区域开始，需要检查是否截断
        elif segment_start < overlap_start < segment_end:
            # 检查是否在句子中间截断
            text = segment.get('segment', '').strip()
            if text and not any(punct in text for punct in ['.', '。', '!', '！', '?', '？']):
                # 在句子中间，保留完整segment
                processed.append(segment)
            else:
                # 可以安全截断的句子结束
                processed.append(segment)
        
    return processed

def create_overlap_chunks(total_duration: float, chunk_duration: float, overlap_seconds: float) -> list:
    """Create overlapping chunk time periods"""
    chunks = []
    current_start = 0.0
    
    while current_start < total_duration:
        chunk_end = min(current_start + chunk_duration, total_duration)
        
        chunk_info = {
            'start': current_start,
            'end': chunk_end,
            'duration': chunk_end - current_start
        }
        chunks.append(chunk_info)
        
        # Next chunk's start time (considering overlap)
        if chunk_end >= total_duration:
            break
            
        current_start = chunk_end - overlap_seconds
        
    print(f"Created {len(chunks)} overlapping chunks:")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i}: {chunk['start']:.1f}s - {chunk['end']:.1f}s (duration: {chunk['duration']:.1f}s)")
    
    return chunks

def check_cuda_compatibility():
    """Check CUDA compatibility, disable CUDA if incompatible"""
    global cuda_available
    
    try:
        if not torch.cuda.is_available():
            print("CUDA unavailable, will use CPU mode")
            cuda_available = False
            return False
        
        # Try to get device count to test CUDA compatibility
        device_count = torch.cuda.device_count()
        if device_count == 0:
            print("No CUDA devices detected, will use CPU mode")
            cuda_available = False
            return False
            
        # Try to get device properties to further test compatibility
        device_props = torch.cuda.get_device_properties(0)
        print(f"✅ Compatible GPU detected: {device_props.name}")
        cuda_available = True
        return True
    except RuntimeError as e:
        if "forward compatibility was attempted on non supported HW" in str(e):
            print("⚠️ CUDA compatibility error: GPU hardware does not support current CUDA version")
            print("This is usually because the host GPU driver version is too old to support CUDA 13.x runtime in container")
            print("Will automatically switch to CPU mode")
        elif "CUDA" in str(e):
            print(f"⚠️ CUDA initialization failed: {e}")
            print("Will automatically switch to CPU mode")
        else:
            print(f"⚠️ Unknown CUDA error: {e}")
            print("Will automatically switch to CPU mode")
        
        cuda_available = False
        return False
    except Exception as e:
        print(f"⚠️ GPU compatibility check failed: {e}")
        print("Will automatically switch to CPU mode")
        cuda_available = False
        return False

def get_gpu_memory_usage():
    """Get GPU memory usage"""
    global cuda_available
    if not cuda_available:
        return 0, 0, 0
    
    try:
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        return allocated, reserved, total
    except Exception as e:
        print(f"⚠️ Failed to get GPU memory information: {e}")
        return 0, 0, 0

def aggressive_memory_cleanup():
    """Aggressive GPU memory cleanup function"""
    global cuda_available
    if cuda_available:
        try:
            # Clear CUDA cache
            torch.cuda.empty_cache()
            # Synchronize all CUDA operations
            torch.cuda.synchronize()
            # Reset peak memory statistics
            torch.cuda.reset_peak_memory_stats()
        except Exception as e:
            print(f"⚠️ CUDA cleanup operation failed: {e}")
    
    # Force Python garbage collection
    for _ in range(3):
        gc.collect()
    
    if cuda_available:
        try:
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"⚠️ CUDA cache cleanup failed: {e}")
    # Return glibc memory to OS
    try_malloc_trim()

def try_malloc_trim():
    """Try to return free memory to the OS through glibc's malloc_trim or available allocator.
    - For glibc: call malloc_trim(0)
    - If jemalloc is enabled and available, try to trigger background release (usually managed by MALLOC_CONF)
    """
    if not ENABLE_MALLOC_TRIM:
        return
    # glibc
    try:
        libc_path = ctypes.util.find_library('c') or 'libc.so.6'
        libc = ctypes.CDLL(libc_path)
        # malloc_trim(size_t) -> int
        try:
            libc.malloc_trim.argtypes = [ctypes.c_size_t]
            libc.malloc_trim.restype = ctypes.c_int
        except Exception:
            pass
        res = libc.malloc_trim(0)
        if res != 0:
            print("✅ Called malloc_trim to return free memory")
        else:
            # Returning 0 may also indicate no trimmable fragments
            print("ℹ️ malloc_trim called, no trimmable or already optimal")
    except Exception as e:
        # Optional jemalloc handling (if enabled via LD_PRELOAD, usually automatically reclaimed by background_thread)
        print(f"⚠️ malloc_trim call failed or unavailable: {e}")

def idle_deep_memory_cleanup():
    """Deep memory cleanup function during idle - simplified to basic cleanup"""
    global cuda_available
    print("🧹 Executing idle memory cleanup...")
    
    # Execute standard memory cleanup
    aggressive_memory_cleanup()
    
    # Additional cleanup measures
    if cuda_available:
        try:
            # Clear CUDA cache
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Reset memory statistics
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()
        except Exception as e:
            print(f"⚠️ CUDA cleanup failed: {e}")
    
    # Garbage collection
    for _ in range(2):
        gc.collect()
    try_malloc_trim()
    
    if cuda_available:
        allocated, reserved, total = get_gpu_memory_usage()
        print(f"✅ Cleanup completed, current GPU memory usage: {allocated:.2f}GB / {total:.2f}GB")
    else:
        memory = psutil.virtual_memory()
        print(f"✅ Cleanup completed, current memory usage: {memory.used/1024**3:.2f}GB / {memory.total/1024**3:.2f}GB")

def immediate_post_request_cleanup():
    """Basic memory cleanup executed after request completion"""
    print("🧽 Executing post-request cleanup...")
    global cuda_available
    
    if cuda_available:
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception:
            pass
    
    # Basic garbage collection
    gc.collect()
    try_malloc_trim()

def check_memory_usage_and_cleanup():
    """Check memory usage and trigger cleanup when necessary - only clean when usage is extremely high"""
    global cuda_available
    
    if cuda_available:
        allocated, _, total = get_gpu_memory_usage()
        # Only clean when GPU memory usage exceeds high threshold to avoid frequent interference
        if allocated > MEMORY_USAGE_ALERT_THRESHOLD_GB and allocated / total > 0.9:
            print(f"🚨 GPU memory usage too high ({allocated:.2f}GB), executing cleanup")
            aggressive_memory_cleanup()
            return True
    else:
        memory = psutil.virtual_memory()
        # Only clean when memory usage exceeds 90%
        if memory.percent > 90:
            print(f"🚨 Memory usage too high ({memory.percent:.1f}%), executing cleanup")
            aggressive_memory_cleanup()
            return True
    
    return False

def should_force_cleanup():
    """Check if GPU memory should be forcibly cleaned"""
    global cuda_available
    if not cuda_available:
        return False
    
    allocated, reserved, total = get_gpu_memory_usage()
    usage_ratio = allocated / total if total > 0 else 0
    return usage_ratio > FORCE_CLEANUP_THRESHOLD

def optimize_model_for_inference(model):
    """Optimize model to reduce GPU memory usage during inference"""
    if model is None:
        return model
    
    # Set to evaluation mode
    model.eval()
    
    # Enable gradient checkpointing (if supported)
    if ENABLE_GRADIENT_CHECKPOINTING and hasattr(model, 'encoder'):
        try:
            if hasattr(model.encoder, 'use_gradient_checkpointing'):
                model.encoder.use_gradient_checkpointing = True
            elif hasattr(model.encoder, 'gradient_checkpointing'):
                model.encoder.gradient_checkpointing = True
        except Exception as e:
            print(f"Cannot enable gradient checkpointing: {e}")
    
    # Disable automatic differentiation (gradients not needed for inference)
    for param in model.parameters():
        param.requires_grad = False
    
    return model

def create_streaming_config():
    """Create streaming processing configuration to reduce GPU memory occupancy"""
    return {
        'batch_size': 1,  # Single batch processing to reduce GPU memory usage
        'num_workers': 0,  # Avoid additional memory overhead from multiprocessing
        'pin_memory': False,  # Don't use page-locked memory to save system memory
        'drop_last': False,
        'persistent_workers': False  # Don't keep worker processes
    }

def load_model_if_needed():
    """Load model on demand, if model is not loaded, load it."""
    global asr_model, cuda_available
    # Use lock to ensure model is only loaded once in multi-threaded environment
    with model_lock:
        if asr_model is None:
            print("="*50)
            print("Model not currently loaded, initializing...")
            # New model default: v3; supports overriding via environment variable
            model_id = os.environ.get('MODEL_ID', 'nvidia/parakeet-tdt-0.6b-v3').strip()
            model_local_path_env = os.environ.get('MODEL_LOCAL_PATH', '').strip()
            print(f"Preferred model: {model_id}")
            try:
                # First check CUDA compatibility
                cuda_available = check_cuda_compatibility()
                
                # Ensure numba cache directory exists
                numba_cache_dir = os.environ.get('NUMBA_CACHE_DIR', '/tmp/numba_cache')
                if not os.path.exists(numba_cache_dir):
                    os.makedirs(numba_cache_dir, exist_ok=True)
                    os.chmod(numba_cache_dir, 0o777)
                
                # Local priority strategy: prioritize using MODEL_LOCAL_PATH; otherwise try common filenames; otherwise auto-download from HF
                candidate_local_paths = []
                if model_local_path_env:
                    candidate_local_paths.append(model_local_path_env)
                # New v3 default filename (if user manually downloaded .nemo)
                candidate_local_paths.append("./app/models/parakeet-tdt-0.6b-v3.nemo")
                # Compatible with old v2 filename (backward compatibility)
                candidate_local_paths.append("./app/models/parakeet-tdt-0.6b-v2.nemo")

                model_path = next((p for p in candidate_local_paths if os.path.exists(p)), None)

                if cuda_available:
                    print(f"✅ Compatible CUDA environment detected, will use GPU acceleration and enable half-precision (FP16) optimization.")
                    
                    # Set Tensor Core optimization
                    setup_tensor_core_optimization()
                    optimize_tensor_operations()
                    
                    # Show GPU and Tensor Core information
                    device_info = torch.cuda.get_device_properties(0)
                    print(f"GPU: {device_info.name}")
                    print(f"Tensor Core support: {get_tensor_core_info()}")
                    
                    # First load model on CPU, then transfer to GPU and enable FP16
                    if model_path:
                        # Local .nemo
                        # Check file permissions
                        if not os.access(model_path, os.R_OK):
                            raise PermissionError(f"Cannot read model file: {model_path}, please check file permissions.")
                        print(f"Restoring from local .nemo: {model_path}")
                        loaded_model = nemo_asr.models.ASRModel.restore_from(restore_path=model_path, map_location=torch.device('cpu'))
                    else:
                        # Auto-download from HF or try to directly fetch .nemo file to local cache directory
                        print(f"Attempting to get model file from Hugging Face: {model_id}")
                        os.makedirs('./app/models', exist_ok=True)
                        downloaded_path = None
                        try:
                            if HfApi is None:
                                raise RuntimeError("huggingface_hub not available")
                            api = HfApi()
                            repo_files = api.list_repo_files(model_id)
                            nemo_files = [f for f in repo_files if f.endswith('.nemo')]
                            if nemo_files:
                                target_fname = nemo_files[0]
                                print(f"Found remote .nemo file: {target_fname}, starting download...")
                                downloaded_path = hf_hub_download(repo_id=model_id, filename=target_fname, cache_dir='./app/models')
                                print(f"Model downloaded to: {downloaded_path}")
                            else:
                                print("No .nemo file found in remote repository, falling back to NeMo.from_pretrained() method to load")
                        except Exception as e:
                            print(f"Failed to get .nemo from Hugging Face: {e}")

                        if downloaded_path and os.path.exists(downloaded_path):
                            loaded_model = nemo_asr.models.ASRModel.restore_from(restore_path=downloaded_path, map_location=torch.device('cpu'))
                        else:
                            print(f"Using NeMo's from_pretrained to load model: {model_id}")
                            loaded_model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_id)
                    loaded_model = loaded_model.cuda()
                    loaded_model = loaded_model.half()
                    
                    # Apply inference optimization
                    loaded_model = optimize_model_for_inference(loaded_model)
                    
                    # Show GPU memory usage
                    allocated, reserved, total = get_gpu_memory_usage()
                    print(f"GPU memory usage after model loading: {allocated:.2f}GB / {total:.2f}GB ({allocated/total*100:.1f}%)")
                else:
                    print("🔄 Using CPU mode.")
                    print("Note: Inference speed will be slower in CPU mode, recommend using a compatible GPU.")
                    if model_path:
                        # Local .nemo
                        if not os.access(model_path, os.R_OK):
                            raise PermissionError(f"Cannot read model file: {model_path}, please check file permissions.")
                        print(f"Restoring from local .nemo: {model_path}")
                        loaded_model = nemo_asr.models.ASRModel.restore_from(restore_path=model_path)
                    else:
                        # Auto-download from HF or try to directly fetch .nemo file to local cache directory (CPU branch)
                        print(f"Attempting to get model file from Hugging Face: {model_id}")
                        os.makedirs('./app/models', exist_ok=True)
                        downloaded_path = None
                        try:
                            if HfApi is None:
                                raise RuntimeError("huggingface_hub not available")
                            api = HfApi()
                            repo_files = api.list_repo_files(model_id)
                            nemo_files = [f for f in repo_files if f.endswith('.nemo')]
                            if nemo_files:
                                target_fname = nemo_files[0]
                                print(f"Found remote .nemo file: {target_fname}, starting download...")
                                downloaded_path = hf_hub_download(repo_id=model_id, filename=target_fname, cache_dir='./app/models')
                                print(f"Model downloaded to: {downloaded_path}")
                            else:
                                print("No .nemo file found in remote repository, falling back to NeMo.from_pretrained() method to load")
                        except Exception as e:
                            print(f"Failed to get .nemo from Hugging Face: {e}")

                        if downloaded_path and os.path.exists(downloaded_path):
                            loaded_model = nemo_asr.models.ASRModel.restore_from(restore_path=downloaded_path)
                        else:
                            print(f"Using NeMo's from_pretrained to load model: {model_id}")
                            loaded_model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_id)
                    loaded_model = optimize_model_for_inference(loaded_model)
                
                # Configure decoding strategy (if model supports)
                try:
                    configure_decoding_strategy(loaded_model)
                except Exception as e:
                    print(f"⚠️ Failed to configure decoding strategy, will use default decoding: {e}")

                asr_model = loaded_model
                print("✅ NeMo ASR model loaded successfully!")
                print("="*50)
            except Exception as e:
                print(f"❌ Model loading failed: {e}")
                print("="*50)
                import traceback
                traceback.print_exc()
                # Re-raise exception so interface can catch and return error information
                raise e
    return asr_model

def predownload_model_artifacts():
    """Download model files to local cache directory in the background, but don't load into memory.
    This function is used to pre-download large files to `/app/models` when lazy loading is enabled to reduce subsequent first load delay.
    """
    try:
        model_id = os.environ.get('MODEL_ID', 'nvidia/parakeet-tdt-0.6b-v3').strip()
        model_local_path_env = os.environ.get('MODEL_LOCAL_PATH', '').strip()
        print(f"[predownload] Starting model pre-download check: {model_id}")

        # Local priority: if local file already exists, no need to download
        candidate_local_paths = []
        if model_local_path_env:
            candidate_local_paths.append(model_local_path_env)
        candidate_local_paths.append('./app/models/parakeet-tdt-0.6b-v3.nemo')
        candidate_local_paths.append('./app/models/parakeet-tdt-0.6b-v2.nemo')
        for p in candidate_local_paths:
            if p and os.path.exists(p):
                print(f"[predownload] Found local model file, no need to download: {p}")
                return

        # Create cache directory
        os.makedirs('./app/models', exist_ok=True)

        # Try to use huggingface_hub to download remote .nemo file (download only, don't restore/load)
        if HfApi is None:
            print("[predownload] huggingface_hub not available, skipping pre-download")
            return

        try:
            api = HfApi()
            repo_files = api.list_repo_files(model_id)
            nemo_files = [f for f in repo_files if f.endswith('.nemo')]
            if not nemo_files:
                print(f"[predownload] No .nemo files found in remote repository: {model_id}, skipping pre-download")
                return
            target_fname = nemo_files[0]
            print(f"[predownload] Found remote .nemo file: {target_fname}, starting download to ./app/models ...")
            try:
                downloaded_path = hf_hub_download(repo_id=model_id, filename=target_fname, cache_dir='./app/models')
                if downloaded_path and os.path.exists(downloaded_path):
                    print(f"[predownload] Model file downloaded: {downloaded_path}")
                else:
                    print(f"[predownload] Download returned path is invalid or doesn't exist: {downloaded_path}")
            except Exception as e:
                print(f"[predownload] hf_hub_download failed: {e}")
        except Exception as e:
            print(f"[predownload] Query remote repository file list failed: {e}")
    except Exception as e:
        print(f"[predownload] Pre-download thread exception: {e}")

def unload_model():
    """Unload model from memory/GPU memory."""
    global asr_model, last_request_time, cuda_available
    with model_lock:
        if asr_model is not None:
            print(f"Model idle for more than {IDLE_TIMEOUT_MINUTES} minutes, unloading from memory...")
            
            # Show GPU memory usage before unloading
            if cuda_available:
                allocated_before, _, total = get_gpu_memory_usage()
                print(f"GPU memory usage before unloading: {allocated_before:.2f}GB / {total:.2f}GB")
            
            asr_model = None
            
            # Execute deep cleanup immediately after unloading
            idle_deep_memory_cleanup()
            try_malloc_trim()
            
            # Show GPU memory usage after unloading
            if cuda_available:
                allocated_after, _, total = get_gpu_memory_usage()
                print(f"GPU memory usage after unloading: {allocated_after:.2f}GB / {total:.2f}GB")
                print(f"GPU memory freed: {allocated_before - allocated_after:.2f}GB")
            
            last_request_time = None # Reset timer to prevent duplicate unloading
            print("✅ Model successfully unloaded and deep cleanup completed.")

def model_cleanup_checker():
    """Background thread, periodically check if model has been idle too long and execute unloading."""
    last_cleanup_time = datetime.datetime.now()
    
    while True:
        # Adaptively adjust check interval based on system status
        current_time = datetime.datetime.now()
        
        # Base monitoring interval - use shorter intervals for more frequent checks
        sleep_interval = IDLE_MONITORING_INTERVAL
        
        # Periodically check memory usage and clean when extremely high
        if check_memory_usage_and_cleanup():
            last_cleanup_time = current_time
        
        if asr_model is not None and last_request_time is not None:
            idle_duration = (current_time - last_request_time).total_seconds()
            
            # Use configured model unload threshold
            model_unload_threshold = IDLE_TIMEOUT_MINUTES * 60
            
            # Check if model needs to be unloaded
            if idle_duration > model_unload_threshold:
                print(f"Model idle for {idle_duration/60:.1f} minutes, exceeding threshold of {model_unload_threshold/60:.1f} minutes")
                unload_model()
                # Execute deep cleanup immediately after model unload
                idle_deep_memory_cleanup()
                last_cleanup_time = current_time
            
            # Adjust check frequency based on idle time
            elif idle_duration > IDLE_DEEP_CLEANUP_THRESHOLD:
                # When idle for long periods, reduce check frequency but execute deep cleanup
                sleep_interval = max(60, IDLE_MONITORING_INTERVAL * 2)  # At least 1 minute interval
                if (current_time - last_cleanup_time).total_seconds() > IDLE_MEMORY_CLEANUP_INTERVAL:
                    print(f"Executing regular deep cleanup (idle for {idle_duration/60:.1f} minutes)")
                    idle_deep_memory_cleanup()
                    last_cleanup_time = current_time
            
            elif idle_duration > IDLE_MEMORY_CLEANUP_INTERVAL:
                # Medium idle time, execute light cleanup
                if (current_time - last_cleanup_time).total_seconds() > IDLE_MEMORY_CLEANUP_INTERVAL:
                    print(f"Executing regular memory cleanup (idle for {idle_duration/60:.1f} minutes)")
                    if AGGRESSIVE_MEMORY_CLEANUP and should_force_cleanup():
                        print("🧹 Executing idle memory cleanup...")
                        aggressive_memory_cleanup()
                    else:
                        # Even if forced cleanup not needed, perform basic cleanup
                        if cuda_available:
                            try:
                                torch.cuda.empty_cache()
                            except Exception:
                                pass
                        gc.collect()
                    last_cleanup_time = current_time
            
            # 即使在短期闲置时也进行最基本的清理
            elif idle_duration > 60:  # 闲置超过1分钟
                if (current_time - last_cleanup_time).total_seconds() > 120:  # 每2分钟清理一次
                    if cuda_available:
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                    gc.collect()
                    last_cleanup_time = current_time
        
        else:
            # 模型未加载或未有请求时，使用较长的检查间隔并定期清理
            sleep_interval = max(60, IDLE_MONITORING_INTERVAL * 2)  # 减少到最少1分钟间隔
            if (current_time - last_cleanup_time).total_seconds() > IDLE_MEMORY_CLEANUP_INTERVAL:
                print("执行无模型状态下的定期清理")
                aggressive_memory_cleanup()
                last_cleanup_time = current_time
        
        time.sleep(sleep_interval)


# --- Flask 应用初始化 ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './app/temp_uploads'
app.config['MAX_CONTENT_LENGTH'] = 2000 * 1024 * 1024  

# --- 辅助函数 ---
def get_audio_duration(file_path: str) -> float:
    """使用 ffprobe 获取音频文件的时长（秒）"""
    command = [
        'ffprobe',
        '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        file_path
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return float(result.stdout)
    except (subprocess.CalledProcessError, ValueError) as e:
        print(f"无法获取文件 '{file_path}' 的时长: {e}")
        return 0.0

def format_srt_time(seconds: float) -> str:
    """将秒数格式化为 SRT 时间戳格式 HH:MM:SS,ms"""
    delta = datetime.timedelta(seconds=seconds)
    # 格式化为 0:00:05.123000
    s = str(delta)
    # 分割秒和微秒
    if '.' in s:
        parts = s.split('.')
        integer_part = parts[0]
        fractional_part = parts[1][:3] # 取前三位毫秒
    else:
        integer_part = s
        fractional_part = "000"

    # 填充小时位
    if len(integer_part.split(':')) == 2:
        integer_part = "0:" + integer_part
    
    return f"{integer_part},{fractional_part}"


def segments_to_srt(segments: list) -> str:
    """将 NeMo 的分段时间戳转换为 SRT 格式字符串"""
    srt_content = []
    for i, segment in enumerate(segments):
        start_time = format_srt_time(segment['start'])
        end_time = format_srt_time(segment['end'])
        text = segment['segment'].strip()
        if text and PREFERRED_LINE_LENGTH > 0:
            text = wrap_text_for_display(
                text,
                preferred_line_length=PREFERRED_LINE_LENGTH,
                max_lines=MAX_SUBTITLE_LINES,
            )
        
        if text: # 仅添加有内容的字幕
            srt_content.append(str(i + 1))
            srt_content.append(f"{start_time} --> {end_time}")
            srt_content.append(text)
            srt_content.append("") # 空行分隔
            
    return "\n".join(srt_content)


def parse_ffmpeg_silence_log(ffmpeg_stderr: str) -> list:
    """解析 ffmpeg silencedetect 输出，返回静音区间 [(start, end), ...]。"""
    import re
    silence_starts = []
    silence_intervals = []
    # silencedetect 输出示例:
    # [silencedetect @ 0x...] silence_start: 12.345
    # [silencedetect @ 0x...] silence_end: 13.789 | silence_duration: 1.444
    start_re = re.compile(r"silence_start:\s*([0-9.]+)")
    end_re = re.compile(r"silence_end:\s*([0-9.]+)")
    for line in ffmpeg_stderr.splitlines():
        m = start_re.search(line)
        if m:
            silence_starts.append(float(m.group(1)))
            continue
        m = end_re.search(line)
        if m and silence_starts:
            start = silence_starts.pop(0)
            end = float(m.group(1))
            silence_intervals.append((start, end))
    return silence_intervals


def find_nearest_silence(target_time: float, silence_intervals: list, max_shift: float) -> float:
    """在 target_time 附近查找最近的静音边界，返回建议的切片开始时间。若未找到合适静音点，则返回 target_time。"""
    if not silence_intervals:
        return target_time
    best_time = target_time
    best_dist = max_shift + 1.0
    for start, end in silence_intervals:
        for edge in (start, end):
            dist = abs(edge - target_time)
            if dist < best_dist and dist <= max_shift:
                best_dist = dist
                best_time = edge
    return best_time


def detect_silences_with_ffmpeg(source_wav: str) -> list:
    """使用 ffmpeg silencedetect 检测静音区间。"""
    command = [
        'ffmpeg', '-hide_banner', '-nostats', '-i', source_wav,
        '-af', f'silencedetect=noise={SILENCE_THRESHOLD_DB}:d={MIN_SILENCE_DURATION}',
        '-f', 'null', '-' 
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    # 无论返回码如何，stderr 都包含 silencedetect 输出
    return parse_ffmpeg_silence_log(result.stderr)

# --- Flask 路由 ---

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint - used for Docker health checks and service monitoring"""
    try:
        current_time = datetime.datetime.now()
        # Basic status
        health_status: Dict[str, Any] = {
            "status": "healthy",
            "timestamp": current_time.isoformat(),
            "service": "parakeet-api",
            "version": "1.0.0",
        }

        # GPU information
        global cuda_available
        if cuda_available:
            try:
                allocated, reserved, total = get_gpu_memory_usage()
                health_status["gpu"] = {
                    "available": True,
                    "memory_allocated_gb": round(allocated, 2),
                    "memory_reserved_gb": round(reserved, 2),
                    "memory_total_gb": round(total, 2),
                    "memory_usage_percent": round((allocated / total) * 100, 1) if total > 0 else 0,
                    "memory_reserved_percent": round((reserved / total) * 100, 1) if total > 0 else 0,
                }
            except Exception as e:
                health_status["gpu"] = {"available": True, "error": str(e)}
        else:
            health_status["gpu"] = {"available": False, "mode": "cpu"}

        # Model and idle information
        model_info: Dict[str, Any] = {"loaded": asr_model is not None, "lazy_load": ENABLE_LAZY_LOAD}
        if last_request_time is not None:
            idle_seconds = (current_time - last_request_time).total_seconds()
            model_info["last_request_time"] = last_request_time.isoformat()
            model_info["idle_duration_seconds"] = round(idle_seconds, 1)
            model_info["idle_duration_minutes"] = round(idle_seconds / 60, 1)
            if idle_seconds > IDLE_TIMEOUT_MINUTES * 60:
                model_info["idle_status"] = "ready_for_unload"
            elif idle_seconds > IDLE_DEEP_CLEANUP_THRESHOLD:
                model_info["idle_status"] = "deep_idle"
            elif idle_seconds > IDLE_MEMORY_CLEANUP_INTERVAL:
                model_info["idle_status"] = "idle"
            else:
                model_info["idle_status"] = "active"
        else:
            model_info["idle_status"] = "no_requests" if asr_model is not None else "unloaded"
        health_status["model"] = model_info

        # System resources
        memory = psutil.virtual_memory()
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
        except Exception:
            cpu_percent = 0.0
        health_status["system"] = {
            "memory_usage_percent": memory.percent,
            "memory_available_gb": round(memory.available / 1024**3, 2),
            "memory_total_gb": round(memory.total / 1024**3, 2),
            "cpu_usage_percent": round(cpu_percent, 1),
        }

        # Optimization configuration summary
        health_status["optimization"] = {
            "aggressive_memory_cleanup": AGGRESSIVE_MEMORY_CLEANUP,
            "idle_timeout_minutes": IDLE_TIMEOUT_MINUTES,
            "idle_memory_cleanup_interval": IDLE_MEMORY_CLEANUP_INTERVAL,
        }

        return jsonify(health_status), 200
    except Exception as e:
        error_status = {
            "status": "unhealthy",
            "timestamp": datetime.datetime.now().isoformat(),
            "error": str(e),
        }
        return jsonify(error_status), 500

@app.route('/health/simple', methods=['GET'])
def simple_health_check():
    """
    Simple health check endpoint - returns only HTTP 200 status
    """
    return "OK", 200

@app.route('/v1/audio/transcriptions', methods=['POST'])
def transcribe_audio():
    """
    OpenAI-compatible speech recognition interface, supports long audio chunking processing.
    """
    # --- -1. API Key Authentication ---
    if API_KEY:
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({"error": "Authorization header is missing or invalid. It must be in 'Bearer <key>' format."}), 401
        
        provided_key = auth_header.split(' ')[1]
        if provided_key != API_KEY:
            return jsonify({"error": "Invalid API key."}), 401

    # --- 0. Ensure model is loaded and update timestamp ---
    try:
        # If lazy loading is enabled, load on demand; otherwise, directly use the already loaded global model
        local_asr_model = load_model_if_needed() if ENABLE_LAZY_LOAD else asr_model
        if not local_asr_model:
            # This case covers both lazy loading failure and pre-loading failure scenarios
            return jsonify({"error": "Model loading failed or not loaded, cannot process request"}), 500
    except Exception as e:
        return jsonify({"error": f"Critical error occurred during model loading: {e}"}), 500
    
    # If lazy loading is enabled, update the last request time
    if ENABLE_LAZY_LOAD:
        global last_request_time
        last_request_time = datetime.datetime.now()

    # --- 1. Basic validation ---
    if 'file' not in request.files:
        return jsonify({"error": "File part not found in request"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    if not shutil.which('ffmpeg'):
        return jsonify({"error": "FFmpeg not installed or not in system PATH"}), 500
    if not shutil.which('ffprobe'):
        return jsonify({"error": "ffprobe not installed or not in system PATH"}), 500

    # Get request parameters
    model_name = request.form.get('model', 'whisper-1')
    response_format = request.form.get('response_format', 'json')  # Supports json, text, srt, verbose_json, vtt
    language = request.form.get('language', None)
    prompt = request.form.get('prompt', None)
    temperature = float(request.form.get('temperature', 0))
    
    print(f"Request received, model: '{model_name}', response format: '{response_format}', language: '{language}'")

    # --- 0.5 Language whitelist validation (Whisper-compatible behavior) ---
    # If client explicitly passes language, we only accept the supported 25 languages, otherwise directly reject
    detected_language = None  # for storing auto-detected language
    if language:
        lang_norm = str(language).strip().lower().replace('_', '-')
        # Compatible with locale codes like "en-US": only take the main language part
        primary = lang_norm.split('-')[0]
        if primary not in SUPPORTED_LANG_CODES:
            # Consistent with Whisper style: return 400 and indicate unsupported in message
            return jsonify({
                "error": {
                    "message": f"Unsupported language: {language}",
                    "type": "invalid_request_error",
                    "param": "language",
                    "code": "unsupported_language"
                }
            }), 400

    original_filename = secure_filename(str(file.filename or 'uploaded_file'))
    unique_id = str(uuid.uuid4())
    temp_original_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}_{original_filename}")
    target_wav_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}.wav")
    
    # List for cleaning all temporary files
    temp_files_to_clean = []

    try:
        # --- 2. Save and uniformly convert to 16k mono WAV ---
        file.save(temp_original_path)
        temp_files_to_clean.append(temp_original_path)
        
        print(f"[{unique_id}] Converting '{original_filename}' to standard WAV format...")
        # Optional preprocessing filter
        ffmpeg_filters = []
        if ENABLE_FFMPEG_DENOISE:
            ffmpeg_filters.append(DENOISE_FILTER)
        ffmpeg_command = [
            'ffmpeg', '-y', '-vn', '-sn', '-dn', '-i', temp_original_path,
            '-ac', '1', '-ar', '16000'
        ]
        if ffmpeg_filters:
            ffmpeg_command += ['-af', ','.join(ffmpeg_filters)]
        ffmpeg_command += [target_wav_path]
        result = subprocess.run(ffmpeg_command, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"FFmpeg error: {result.stderr}")
            return jsonify({"error": "File conversion failed", "details": result.stderr}), 500
        temp_files_to_clean.append(target_wav_path)

        # --- 2.5 Automatic language detection and validation (when language is not explicitly passed) ---
        if not language:
            try:
                lid_clip_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}_lid.wav")
                temp_files_to_clean.append(lid_clip_path)
                # Take short segment for fast transcription
                clip_seconds = max(5, int(LID_CLIP_SECONDS))
                probe_dur = get_audio_duration(target_wav_path)
                if probe_dur > 0:
                    clip_seconds = min(clip_seconds, int(math.ceil(probe_dur)))
                clip_cmd = [
                    'ffmpeg', '-y', '-i', target_wav_path,
                    '-t', str(clip_seconds),
                    '-ac', '1', '-ar', '16000',
                    lid_clip_path
                ]
                _res = subprocess.run(clip_cmd, capture_output=True, text=True)
                if _res.returncode == 0 and os.path.exists(lid_clip_path):
                    # Text-only inference (no timestamps, reduce overhead)
                    with inference_semaphore:
                        lid_out = safe_transcribe(
                            local_asr_model,
                            lid_clip_path,
                            need_timestamps=False,
                            batch_size=1,
                            num_workers=0,
                        )
                    # Extract text
                    lid_text = ""
                    if isinstance(lid_out, list) and lid_out:
                        first = lid_out[0]
                        try:
                            if hasattr(first, 'text') and first.text:
                                lid_text = str(first.text)
                            elif hasattr(first, 'segment') and first.segment:
                                lid_text = str(first.segment)
                            else:
                                lid_text = str(first)
                        except Exception:
                            lid_text = str(first)

                    # Use lightweight text language detection for language detection
                    if lid_text and lid_text.strip():
                        try:
                            try:
                                from langdetect import detect  # type: ignore
                            except Exception:
                                detect = None  # type: ignore
                            detected = None
                            if detect is not None:
                                detected = detect(lid_text)
                            # If language can be detected
                            if detected:
                                det_primary = str(detected).strip().lower().split('-')[0]
                                if det_primary:
                                    if det_primary in SUPPORTED_LANG_CODES:
                                        # Detected supported language, store for later use
                                        detected_language = det_primary
                                        print(f"[{unique_id}] Auto-detected language: {detected_language}")
                                    elif ENABLE_AUTO_LANGUAGE_REJECTION:
                                        # Detected unsupported language and auto-rejection is enabled
                                        return jsonify({
                                            "error": {
                                                "message": f"Unsupported language: {detected}",
                                                "type": "invalid_request_error",
                                                "param": "language",
                                                "code": "unsupported_language"
                                            }
                                        }), 400
                                    else:
                                        # Detected unsupported language but auto-rejection not enabled, default to English
                                        detected_language = "en"
                                        print(f"[{unique_id}] Detected unsupported language {detected}, defaulting to English")
                        except Exception as _e:
                            # Detection failure doesn't affect main flow, default to English
                            print(f"[{unique_id}] Language auto-detection failed, defaulting to English: {_e}")
                            detected_language = "en"
                    else:
                        # Cannot extract text, default to English
                        print(f"[{unique_id}] Cannot extract text for language detection, defaulting to English")
                        detected_language = "en"
            except Exception as _e:
                # Language detection failure, default to English
                print(f"[{unique_id}] Language detection process failed, defaulting to English: {_e}")
                detected_language = "en"
        else:
            # Language was explicitly provided, use the primary part
            detected_language = language.strip().lower().replace('_', '-').split('-')[0]

        # --- 3. Audio chunking ---
        # Dynamically adjust chunk size based on GPU memory usage
        heavy_ts_request = response_format in ['srt', 'vtt', 'verbose_json']
        if cuda_available:
            allocated, _, total = get_gpu_memory_usage()
            memory_usage_ratio = allocated / total if total > 0 else 0
            
            if memory_usage_ratio > 0.6:  # If GPU memory usage exceeds 60%
                # Reduce chunk size to decrease GPU memory pressure
                adjusted_chunk_minutes = max(3, CHUNK_MINITE - 2)
                print(f"[{unique_id}] High GPU memory usage ({memory_usage_ratio*100:.1f}%), adjusting chunk size from {CHUNK_MINITE} minutes to {adjusted_chunk_minutes} minutes")
                CHUNK_DURATION_SECONDS = adjusted_chunk_minutes * 60
            else:
                CHUNK_DURATION_SECONDS = CHUNK_MINITE * 60
            # Set more conservative upper limit for ≤8~12GB GPU memory devices or requests requiring timestamps to avoid attention matrix OOM
            try:
                vram_gb = total
                cap_env = os.environ.get('CHUNK_SECONDS_CAP', '').strip()
                if cap_env:
                    cap_sec = int(float(cap_env))
                else:
                    if vram_gb <= 8.5:
                        cap_sec = 180 if heavy_ts_request else 240
                    elif vram_gb <= 12.0:
                        cap_sec = 300 if heavy_ts_request else 480
                    else:
                        cap_sec = 600
                if CHUNK_DURATION_SECONDS > cap_sec:
                    print(f"[{unique_id}] Based on GPU VRAM ({vram_gb:.1f}GB){' and timestamps needed' if heavy_ts_request else ''}, limiting chunk duration to {cap_sec}s")
                    CHUNK_DURATION_SECONDS = cap_sec
            except Exception:
                pass
        else:
            # Use smaller chunks in CPU mode to avoid memory shortage
            cpu_chunk_minutes = max(3, CHUNK_MINITE // 2)  # Halve chunk size in CPU mode
            print(f"[{unique_id}] CPU mode, adjusting chunk size to {cpu_chunk_minutes} minutes")
            CHUNK_DURATION_SECONDS = cpu_chunk_minutes * 60
            # CPU mode also sets upper limit, especially when timestamps are needed
            try:
                cap_env = os.environ.get('CHUNK_SECONDS_CAP', '').strip()
                cap_sec = int(float(cap_env)) if cap_env else (180 if heavy_ts_request else 240)
                if CHUNK_DURATION_SECONDS > cap_sec:
                    print(f"[{unique_id}] CPU mode limiting chunk duration to {cap_sec}s")
                    CHUNK_DURATION_SECONDS = cap_sec
            except Exception:
                pass
                
        total_duration = get_audio_duration(target_wav_path)
        if total_duration == 0:
            return jsonify({"error": "Cannot process audio with 0 duration"}), 400

        # Check if chunking is needed, if audio duration is less than chunking threshold, process directly
        if total_duration <= CHUNK_DURATION_SECONDS:
            print(f"[{unique_id}] Total file duration: {total_duration:.2f}s. Less than chunking threshold ({CHUNK_DURATION_SECONDS}s), no chunking needed.")
            chunk_paths = [target_wav_path]
            chunk_info_list = [{'start': 0, 'end': total_duration, 'duration': total_duration}]
            num_chunks = 1
        else:
            # Use overlapping chunking strategy
            if ENABLE_OVERLAP_CHUNKING:
                print(f"[{unique_id}] Enabling overlapping chunking mode, overlap duration: {CHUNK_OVERLAP_SECONDS}s")
                chunk_info_list = create_overlap_chunks(total_duration, CHUNK_DURATION_SECONDS, CHUNK_OVERLAP_SECONDS)
            else:
                # Traditional hard chunking
                num_chunks = math.ceil(total_duration / CHUNK_DURATION_SECONDS)
                chunk_info_list = []
                for i in range(num_chunks):
                    start_time = i * CHUNK_DURATION_SECONDS
                    end_time = min(start_time + CHUNK_DURATION_SECONDS, total_duration)
                    chunk_info_list.append({
                        'start': start_time,
                        'end': end_time, 
                        'duration': end_time - start_time
                    })
            
            chunk_paths = []
            num_chunks = len(chunk_info_list)
            print(f"[{unique_id}] Total file duration: {total_duration:.2f}s. Will be chunked into {num_chunks} pieces.")
            
            # If silence alignment is enabled, pre-detect silence intervals
            silence_intervals = []
            if ENABLE_SILENCE_ALIGNED_CHUNKING and total_duration > CHUNK_DURATION_SECONDS:
                print(f"[{unique_id}] Detecting silence intervals for chunk alignment: noise={SILENCE_THRESHOLD_DB}, min_dur={MIN_SILENCE_DURATION}s")
                silence_intervals = detect_silences_with_ffmpeg(target_wav_path)
                print(f"[{unique_id}] Detected {len(silence_intervals)} silence intervals")

            for i, chunk_info in enumerate(chunk_info_list):
                chunk_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}_chunk_{i}.wav")
                chunk_paths.append(chunk_path)
                temp_files_to_clean.append(chunk_path)
                
                start_time = chunk_info['start']
                # Align chunk start to nearest silence boundary (not exceeding max offset)
                if ENABLE_SILENCE_ALIGNED_CHUNKING and silence_intervals:
                    aligned_start = find_nearest_silence(start_time, silence_intervals, SILENCE_MAX_SHIFT_SECONDS)
                    if aligned_start != start_time:
                        print(f"[{unique_id}] Chunk {i+1} start time {start_time:.2f}s aligned to silence {aligned_start:.2f}s")
                        # Also adjust the end of this chunk, keeping duration unchanged
                        shift = aligned_start - start_time
                        start_time = max(0.0, aligned_start)
                        chunk_info['start'] = start_time
                        chunk_info['end'] = min(total_duration, chunk_info['end'] + shift)
                        chunk_info['duration'] = chunk_info['end'] - chunk_info['start']
                duration = chunk_info['duration']
                
                print(f"[{unique_id}] Creating chunk {i+1}/{num_chunks} ({start_time:.1f}s - {chunk_info['end']:.1f}s)...")
                chunk_command = [
                    'ffmpeg', '-y', '-i', target_wav_path,
                    '-ss', str(start_time),
                    '-t', str(duration),
                    '-c', 'copy',
                    chunk_path
                ]
                result = subprocess.run(chunk_command, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"[{unique_id}] ⚠️ Warning creating chunk {i+1}: {result.stderr}")
                    # Continue processing, don't interrupt
        
        # --- 4. Loop through transcriptions and merge results ---
        all_segments = []
        all_words = []
        chunk_boundaries = []
        # Only request timestamps when SRT/VTT/verbose_json is needed, reducing GPU memory and computation
        need_timestamps = response_format in ['srt', 'vtt', 'verbose_json']
        # When long subtitle splitting is needed and word-level timestamp-based splitting is enabled, also try to collect word-level timestamps
        collect_word_timestamps = (response_format == 'verbose_json') or (SPLIT_LONG_SUBTITLES and ENABLE_WORD_TIMESTAMPS_FOR_SPLIT)
        full_text_parts = []  # When timestamps not needed, directly collect text

        for i, (chunk_path, chunk_info) in enumerate(zip(chunk_paths, chunk_info_list)):
            print(f"[{unique_id}] Transcribing chunk {i+1}/{num_chunks}...")
            
            # Check GPU memory usage, if too high then force cleanup
            if should_force_cleanup():
                print(f"[{unique_id}] High GPU memory usage, executing forced cleanup...")
                aggressive_memory_cleanup()
            
            # Show current GPU/CPU memory usage
            if cuda_available:
                allocated, _, total = get_gpu_memory_usage()
                print(f"[{unique_id}] GPU memory usage before processing chunk {i+1}: {allocated:.2f}GB / {total:.2f}GB")
            else:
                # Show CPU memory usage
                memory = psutil.virtual_memory()
                print(f"[{unique_id}] CPU memory usage before processing chunk {i+1}: {memory.used/1024**3:.2f}GB / {memory.total/1024**3:.2f}GB ({memory.percent:.1f}%)")
            
            # Transcribe current chunk
            # Use with torch.cuda.amp.autocast() to run inference in half-precision
            # Inference mode further reduces memory/overhead, concurrency control to avoid OOM
            with inference_semaphore:
                output = safe_transcribe(
                    local_asr_model,
                    chunk_path,
                    need_timestamps=need_timestamps,
                    batch_size=TRANSCRIBE_BATCH_SIZE,
                    num_workers=TRANSCRIBE_NUM_WORKERS,
                )

            # Immediate memory cleanup
            if AGGRESSIVE_MEMORY_CLEANUP:
                aggressive_memory_cleanup()
            else:
                if cuda_available:
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                gc.collect()
            
            # Record chunk boundaries for later merging
            chunk_start_offset = chunk_info['start']
            chunk_boundaries.append(chunk_start_offset)
            
            if need_timestamps:
                if output and getattr(output[0], 'timestamp', None):
                    # Correct and collect segment timestamps
                    if 'segment' in output[0].timestamp:
                        for seg in output[0].timestamp['segment']:
                            seg['start'] += chunk_start_offset
                            seg['end'] += chunk_start_offset
                            all_segments.append(seg)
                    # Correct and collect word timestamps (only needed for verbose_json)
                    if collect_word_timestamps and 'word' in output[0].timestamp:
                        for word in output[0].timestamp['word']:
                            word['start'] += chunk_start_offset
                            word['end'] += chunk_start_offset
                            all_words.append(word)
                else:
                    # Some models/configurations may not return timestamps, try direct text fallback
                    if isinstance(output, list) and output:
                        full_text_parts.append(str(output[0]))
            else:
                # No timestamps needed, directly get text
                if isinstance(output, list) and output:
                    # NeMo returned elements might be Hypothesis objects, prioritize extracting .text or .segment fields
                    first = output[0]
                    try:
                        # Prioritize common attributes
                        if hasattr(first, 'text') and first.text:
                            full_text_parts.append(str(first.text))
                        elif hasattr(first, 'segment') and first.segment:
                            full_text_parts.append(str(first.segment))
                        else:
                            full_text_parts.append(str(first))
                    except Exception:
                        full_text_parts.append(str(first))
            
            # Release temporary output reference
            try:
                del output
            except Exception:
                pass
            # Immediately delete processed chunk files to save disk space and memory
            if num_chunks > 1 and os.path.exists(chunk_path):
                try:
                    os.remove(chunk_path)
                    temp_files_to_clean.remove(chunk_path)
                    print(f"[{unique_id}] Deleted processed chunk file: chunk_{i}")
                except Exception as e:
                    print(f"[{unique_id}] Error deleting chunk file: {e}")

        print(f"[{unique_id}] All chunks transcribed, merging results.")
        
        # --- 4.5. Process overlapping regions and merge segments ---
        if ENABLE_OVERLAP_CHUNKING and len(chunk_boundaries) > 1:
            print(f"[{unique_id}] Processing overlapping regions, removing duplicate content...")
            all_segments = merge_overlapping_segments(all_segments, chunk_boundaries, CHUNK_OVERLAP_SECONDS)
            print(f"[{unique_id}] Overlap processing completed, final segment count: {len(all_segments)}")

        # --- 4.6. Subtitle post-processing: merge/extend short subtitles to avoid flickering ---
        if MERGE_SHORT_SUBTITLES and all_segments:
            before_cnt = len(all_segments)
            all_segments = enforce_min_subtitle_duration(
                all_segments,
                min_duration=MIN_SUBTITLE_DURATION_SECONDS,
                merge_max_gap=SHORT_SUBTITLE_MERGE_MAX_GAP_SECONDS,
                min_chars=SHORT_SUBTITLE_MIN_CHARS,
                min_gap=SUBTITLE_MIN_GAP_SECONDS,
            )
            print(f"[{unique_id}] Subtitle post-processing completed: {before_cnt} -> {len(all_segments)} segments (minimum duration {MIN_SUBTITLE_DURATION_SECONDS}s)")

        # --- 4.7. Long subtitle splitting (by duration/character count limits) ---
        if SPLIT_LONG_SUBTITLES and all_segments:
            before_cnt = len(all_segments)
            all_segments = split_and_wrap_long_subtitles(
                segments=all_segments,
                words=all_words if collect_word_timestamps else None,
                max_duration=MAX_SUBTITLE_DURATION_SECONDS,
                max_chars=MAX_SUBTITLE_CHARS_PER_SEGMENT,
                preferred_line_length=PREFERRED_LINE_LENGTH,
                max_lines=MAX_SUBTITLE_LINES,
                punctuation=SUBTITLE_SPLIT_PUNCTUATION,
            )
            print(f"[{unique_id}] Long subtitle splitting completed: {before_cnt} -> {len(all_segments)} segments (max duration {MAX_SUBTITLE_DURATION_SECONDS}s, max chars {MAX_SUBTITLE_CHARS_PER_SEGMENT})")

        # --- 5. Format final output ---
        # If neither timestamp segments nor direct text, consider as failure;
        # otherwise even if no segments (e.g., model returns pure text only), should return text result.
        if not all_segments and not full_text_parts:
            return jsonify({"error": "Transcription failed, model did not return any valid content"}), 500

        # Build complete transcription text
        full_text = " ".join([seg['segment'].strip() for seg in all_segments if seg['segment'].strip()])
        
        # Return different formats based on response_format
        if response_format == 'text':
            if not full_text:
                # When timestamps not enabled and directly collecting text
                full_text = " ".join(full_text_parts) if full_text_parts else ""
            return Response(full_text, mimetype='text/plain')
        elif response_format == 'srt':
            srt_result = segments_to_srt(all_segments)
            return Response(srt_result, mimetype='text/plain')
        elif response_format == 'vtt':
            vtt_result = segments_to_vtt(all_segments)
            return Response(vtt_result, mimetype='text/plain')
        elif response_format == 'verbose_json':
            # Detailed JSON format, containing more information
            response_data = {
                "task": "transcribe",
                "language": language or detected_language or "en",
                "duration": total_duration,
                "text": full_text,
                "segments": [
                    {
                        "id": i,
                        "seek": int(seg['start'] * 100),  # Convert to centiseconds
                        "start": seg['start'],
                        "end": seg['end'],
                        "text": seg['segment'].strip(),
                        "tokens": [],  # NeMo doesn't provide tokens, leave empty
                        "temperature": temperature,
                        "avg_logprob": -0.5,  # Simulation value
                        "compression_ratio": 1.0,  # Simulation value
                        "no_speech_prob": 0.0,  # Simulation value
                        "words": [
                            {
                                "word": word['word'],
                                "start": word['start'],
                                "end": word['end'],
                                "probability": 0.9  # Simulation value
                            }
                            for word in all_words 
                            if word['start'] >= seg['start'] and word['end'] <= seg['end']
                        ] if all_words else []
                    }
                    for i, seg in enumerate(all_segments) if seg['segment'].strip()
                ]
            }
            return jsonify(response_data)
        else:
            # Default JSON format (response_format == 'json')
            if not all_segments:
                # When timestamps not enabled, text comes from direct output
                if not full_text:
                    full_text = " ".join(full_text_parts) if full_text_parts else ""
            response_data = {"text": full_text}
            return jsonify(response_data)

    except Exception as e:
        print(f"Serious error occurred during processing: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Internal server error", "details": str(e)}), 500
    finally:
        # --- 6. Clean up all temporary files ---
        print(f"[{unique_id}] Cleaning temporary files...")
        for f_path in temp_files_to_clean:
            if os.path.exists(f_path):
                os.remove(f_path)
        print(f"[{unique_id}] Temporary files cleaned.")
        
        # --- 7. Execute immediate post-request cleanup ---
        immediate_post_request_cleanup()
        
        # --- 8. Force memory cleanup to avoid accumulation ---
        print(f"[{unique_id}] Executing final memory cleanup...")
        if cuda_available:
            allocated_before, _, total = get_gpu_memory_usage()
            print(f"[{unique_id}] GPU memory usage before cleanup: {allocated_before:.2f}GB / {total:.2f}GB")
        else:
            memory_before = psutil.virtual_memory()
            print(f"[{unique_id}] Memory usage before cleanup: {memory_before.used/1024**3:.2f}GB / {memory_before.total/1024**3:.2f}GB")
        
        # Execute standard memory cleanup
        aggressive_memory_cleanup()
        try_malloc_trim()
        
        if cuda_available:
            allocated_after, _, total = get_gpu_memory_usage()
            print(f"[{unique_id}] GPU memory usage after cleanup: {allocated_after:.2f}GB / {total:.2f}GB")
            if allocated_before > 0:
                print(f"[{unique_id}] Released GPU memory: {allocated_before - allocated_after:.2f}GB")
        else:
            memory_after = psutil.virtual_memory()
            print(f"[{unique_id}] Memory usage after cleanup: {memory_after.used/1024**3:.2f}GB / {memory_after.total/1024**3:.2f}GB")
        print(f"[{unique_id}] Memory cleanup completed.")

def format_vtt_time(seconds: float) -> str:
    """将秒数格式化为 VTT 时间戳格式 HH:MM:SS.mmm"""
    delta = datetime.timedelta(seconds=seconds)
    # 格式化为 0:00:05.123000
    s = str(delta)
    # 分割秒和微秒
    if '.' in s:
        parts = s.split('.')
        integer_part = parts[0]
        fractional_part = parts[1][:3]  # 取前三位毫秒
    else:
        integer_part = s
        fractional_part = "000"

    # 填充小时位
    if len(integer_part.split(':')) == 2:
        integer_part = "0:" + integer_part
    
    return f"{integer_part}.{fractional_part}"


def wrap_text_for_display(text: str, preferred_line_length: int, max_lines: int) -> str:
    """将单行文本按字数软换行为最多 max_lines 行，尽量在空格或标点处断行。
    若文本超过行数限制，后续仍保留但不强制增加行数（SRT/VTT可多行）。
    """
    if preferred_line_length <= 0 or max_lines <= 0:
        return text
    import re
    words = re.findall(r"\S+|\s+", text)
    lines = []
    current = ""
    for token in words:
        tentative = current + token
        if len(tentative.strip()) <= preferred_line_length or not current:
            current = tentative
        else:
            lines.append(current.strip())
            current = token.lstrip()
            if len(lines) >= max_lines - 1:
                break
    if current.strip():
        lines.append(current.strip())
    return "\n".join(lines)


def split_and_wrap_long_subtitles(
    segments: list,
    words: list | None,
    max_duration: float,
    max_chars: int,
    preferred_line_length: int,
    max_lines: int,
    punctuation: str,
) -> list:
    """按时长与字符数将过长字幕拆分为多条，并对每条文本进行换行。
    - 若提供 words（词级时间戳），优先在词边界拆分；否则退化为按标点/字符切分。
    """
    if not segments:
        return []

    # 建立每段内的 words 索引（若提供）
    words_by_range: list[list] = []
    if words:
        for seg in segments:
            start, end = seg.get('start', 0.0), seg.get('end', 0.0)
            seg_words = [w for w in words if w.get('start', 0.0) >= start and w.get('end', 0.0) <= end]
            words_by_range.append(seg_words)
    else:
        words_by_range = [[] for _ in segments]

    import re
    punct_set = set(punctuation)

    def split_points_by_chars(text: str) -> list[int]:
        points: list[int] = []
        last = 0
        while last + max_chars < len(text):
            cut = last + max_chars
            # 尽量向左回退到最近的空格或标点
            back = cut
            while back > last and text[back - 1] not in punct_set and not text[back - 1].isspace():
                back -= 1
            if back == last:
                back = cut
            points.append(back)
            last = back
        return points

    new_segments: list = []
    for seg, seg_words in zip(segments, words_by_range):
        start = float(seg.get('start', 0.0))
        end = float(seg.get('end', 0.0))
        text = str(seg.get('segment', '')).strip()
        if not text:
            continue

        duration = max(0.0, end - start)
        too_long_by_time = duration > max_duration
        too_long_by_chars = len(text) > max_chars

        if not too_long_by_time and not too_long_by_chars:
            seg_copy = dict(seg)
            seg_copy['segment'] = wrap_text_for_display(text, preferred_line_length, max_lines)
            new_segments.append(seg_copy)
            continue

        # 计算应拆分的片段数（时间/字符双约束）
        parts_by_time = max(1, int(math.ceil(duration / max_duration))) if max_duration > 0 else 1
        parts_by_chars = max(1, int(math.ceil(len(text) / max_chars))) if max_chars > 0 else 1
        parts = max(parts_by_time, parts_by_chars)

        # 基于词时间戳拆分
        if seg_words:
            total_dur = duration if duration > 0 else 1e-6
            target_bounds = [start + i * (total_dur / parts) for i in range(1, parts)]
            cut_times: list[float] = []
            for tb in target_bounds:
                # 找到离 tb 最近的词边界
                best_t = None
                best_d = 1e9
                for w in seg_words:
                    for edge in (w.get('start', 0.0), w.get('end', 0.0)):
                        d = abs(edge - tb)
                        if d < best_d:
                            best_d = d
                            best_t = edge
                if best_t is not None:
                    cut_times.append(best_t)
            cut_times = sorted(t for t in cut_times if start < t < end)

            # 按 cut_times 切片
            times = [start] + cut_times + [end]
            # 将词按时间段分桶，并组装文本
            for i in range(len(times) - 1):
                s_i, e_i = times[i], times[i + 1]
                sub_words = [w for w in seg_words if w.get('start', 0.0) >= s_i and w.get('end', 0.0) <= e_i]
                sub_text = " ".join(w.get('word', '').strip() for w in sub_words if w.get('word'))
                if not sub_text:
                    # 回退到原文本的切片估计
                    ratio_a = (s_i - start) / total_dur
                    ratio_b = (e_i - start) / total_dur
                    a = int(ratio_a * len(text))
                    b = int(ratio_b * len(text))
                    sub_text = text[a:b].strip()
                sub_text = wrap_text_for_display(sub_text, preferred_line_length, max_lines)
                new_segments.append({'start': s_i, 'end': e_i, 'segment': sub_text})
            continue

        # 无词级时间戳时：按字符与标点近似拆分
        # 先按字符上限计算断点
        points = split_points_by_chars(text) if too_long_by_chars else []
        # 加入标点断点（句末优先）
        for m in re.finditer(r"[。！？!?.,;；，]", text):
            idx = m.end()
            # 只在过长时考虑
            if len(text) > max_chars or duration > max_duration:
                points.append(idx)
        points = sorted(set(p for p in points if 0 < p < len(text)))

        # 根据 points 切文本，时间均分
        stops = [0] + points + [len(text)]
        times = [start + i * ((end - start) / (len(stops) - 1)) for i in range(len(stops))]
        for i in range(len(stops) - 1):
            a, b = stops[i], stops[i + 1]
            s_i, e_i = times[i], times[i + 1]
            sub_text = text[a:b].strip()
            if not sub_text:
                continue
            sub_text = wrap_text_for_display(sub_text, preferred_line_length, max_lines)
            new_segments.append({'start': s_i, 'end': e_i, 'segment': sub_text})

    # 最终保证按开始时间排序
    new_segments.sort(key=lambda s: (s.get('start', 0.0), s.get('end', 0.0)))
    return new_segments


@app.route('/admin/unload', methods=['POST'])
def admin_unload_model():
    """手动卸载模型并执行深度清理。
    - 若设置了 API_KEY，则需要 Bearer 认证。
    - 返回当前内存/显存使用信息。
    """
    if API_KEY:
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({"error": "Authorization header is missing or invalid."}), 401
        provided_key = auth_header.split(' ')[1]
        if provided_key != API_KEY:
            return jsonify({"error": "Invalid API key."}), 401

    prev_gpu = None
    if cuda_available:
        try:
            prev_gpu = get_gpu_memory_usage()
        except Exception:
            prev_gpu = None

    unload_model()
    try_malloc_trim()

    resp: Dict[str, Any] = {"status": "unloaded"}
    try:
        mem = psutil.virtual_memory()
        resp["system_memory_gb"] = {
            "used": round(mem.used/1024**3, 2),
            "total": round(mem.total/1024**3, 2),
            "percent": mem.percent,
        }
    except Exception:
        pass
    if cuda_available:
        try:
            alloc, _, total = get_gpu_memory_usage()
            resp["gpu_memory_gb"] = {
                "allocated": round(alloc, 2),
                "total": round(total, 2),
            }
            if prev_gpu:
                resp["gpu_freed_gb"] = round(max(prev_gpu[0]-alloc, 0.0), 2)
        except Exception:
            pass
    return jsonify(resp), 200

def configure_decoding_strategy(model):
    """配置 NeMo 模型的解码策略（若支持）。
    - 对 RNNT/Conformer-Transducer 等模型，尝试开启 beam search。
    - 若模型不支持相应属性，静默跳过。
    """
    try:
        if hasattr(model, 'change_decoding_strategy'):
            if DECODING_STRATEGY == 'beam':
                model.change_decoding_strategy(decoding_cfg={
                    'strategy': 'beam',
                    'beam_size': RNNT_BEAM_SIZE,
                })
                print(f"✅ 启用 Beam Search，beam_size={RNNT_BEAM_SIZE}")
            else:
                # 在低显存环境下，禁用 CUDA graphs 降低一次性显存峰值
                model.change_decoding_strategy(decoding_cfg={
                    'strategy': 'greedy',
                    'allow_cuda_graphs': False,
                    'greedy': {
                        'use_cuda_graph_decoder': False,
                        'max_symbols_per_step': 10,
                        'loop_labels': True,
                    }
                })
                print("✅ 启用 Greedy 解码")
        elif hasattr(model, 'decoder') and hasattr(model.decoder, 'cfg'):
            # 兼容部分模型的 decoder 配置
            decoder_cfg = getattr(model.decoder, 'cfg')
            if DECODING_STRATEGY == 'beam' and hasattr(decoder_cfg, 'beam_size'):
                decoder_cfg.beam_size = RNNT_BEAM_SIZE
                print(f"✅ 配置 decoder.beam_size={RNNT_BEAM_SIZE}")
            # 其余情况按默认
    except Exception as e:
        print(f"⚠️ 设置解码策略时出错: {e}")


def safe_transcribe(model, audio_path: str, need_timestamps: bool, batch_size: int, num_workers: int):
    """执行一次安全的转写：
    - 使用 autocast + inference_mode 降低显存
    - 如遇 CUDA OOM，自动降级为 greedy 解码并重试一次
    """
    global DECODING_STRATEGY
    try:
        if cuda_available:
            with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.float16):
                return model.transcribe(
                    [audio_path],
                    timestamps=need_timestamps,
                    batch_size=batch_size,
                    num_workers=num_workers,
                )
        else:
            with torch.inference_mode():
                return model.transcribe(
                    [audio_path],
                    timestamps=need_timestamps,
                    batch_size=batch_size,
                    num_workers=num_workers,
                )
    except RuntimeError as e:
        if 'CUDA out of memory' in str(e) or 'CUDA error' in str(e):
            print("⚠️ 检测到 CUDA 内存不足，尝试降级为 greedy 解码并重试一次…")
            aggressive_memory_cleanup()
            # 记录原策略并降级
            original_strategy = DECODING_STRATEGY
            try:
                # 强制切换为 greedy
                os.environ['DECODING_STRATEGY'] = 'greedy'
                DECODING_STRATEGY = 'greedy'
                configure_decoding_strategy(model)
                # 重试
                if cuda_available:
                    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.float16):
                        return model.transcribe(
                            [audio_path],
                            timestamps=need_timestamps,
                            batch_size=1,  # 进一步收缩批量
                            num_workers=0,
                        )
                else:
                    with torch.inference_mode():
                        return model.transcribe(
                            [audio_path],
                            timestamps=need_timestamps,
                            batch_size=1,
                            num_workers=0,
                        )
            finally:
                # 尝试恢复原策略（若需要）
                os.environ['DECODING_STRATEGY'] = original_strategy
                DECODING_STRATEGY = original_strategy
                try:
                    configure_decoding_strategy(model)
                except Exception:
                    pass
        # 非OOM错误原样抛出
        raise
    except ValueError as e:
        # 处理 NeMo TDT Beam 在开启 timestamps 时不支持 alignment preservation 的情况
        if 'Alignment preservation has not been implemented' in str(e):
            print("⚠️ 检测到 TDT Beam 解码不支持对齐保留，自动切换到 greedy 解码并重试一次…")
            aggressive_memory_cleanup()
            original_strategy = DECODING_STRATEGY
            try:
                # 强制切换为 greedy
                os.environ['DECODING_STRATEGY'] = 'greedy'
                globals()['DECODING_STRATEGY'] = 'greedy'
                configure_decoding_strategy(model)
                # 重试（进一步降低批量与并发）
                if cuda_available:
                    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.float16):
                        return model.transcribe(
                            [audio_path],
                            timestamps=need_timestamps,
                            batch_size=1,
                            num_workers=0,
                        )
                else:
                    with torch.inference_mode():
                        return model.transcribe(
                            [audio_path],
                            timestamps=need_timestamps,
                            batch_size=1,
                            num_workers=0,
                        )
            finally:
                # 尝试恢复原策略
                os.environ['DECODING_STRATEGY'] = original_strategy
                globals()['DECODING_STRATEGY'] = original_strategy
                try:
                    configure_decoding_strategy(model)
                except Exception:
                    pass
        # 其他 ValueError 继续抛出
        raise

# --- Waitress 服务器启动 ---
if __name__ == '__main__':
    
    # 根据是否启用懒加载来决定是预加载模型还是启动后台监控线程
    if ENABLE_LAZY_LOAD:
        print("懒加载模式已启用。模型将在第一次请求时加载。")
        # 启动后台线程来监控和卸载闲置的模型
        if IDLE_TIMEOUT_MINUTES > 0:
            print(f"将启用模型自动卸载功能，闲置超时: {IDLE_TIMEOUT_MINUTES} 分钟。")
            cleanup_thread = threading.Thread(target=model_cleanup_checker, daemon=True)
            cleanup_thread.start()
        else:
            print("模型自动卸载功能已禁用 (IDLE_TIMEOUT_MINUTES=0)。")
        # 启动后台线程在容器启动时预下载模型文件（仅下载到磁盘，不加载到内存）
        print("在后台启动模型预下载线程（仅下载文件，不加载到内存）...")
        try:
            predownload_thread = threading.Thread(target=predownload_model_artifacts, daemon=True)
            predownload_thread.start()
        except Exception as e:
            print(f"启动模型预下载线程失败: {e}")
    else:
        # 懒加载被禁用，在启动时直接加载模型
        print("懒加载模式已禁用，正在启动时预加载模型...")
        try:
            load_model_if_needed()
        except Exception as e:
            print(f"❌ 启动时预加载模型失败: {e}")
            # 预加载失败时退出，以避免运行一个损坏的服务
            exit(1)

    if API_KEY:
        print(f"API Key 认证已启用。请在请求头中提供 'Authorization: Bearer YOUR_API_KEY'")
    else:
        print("API Key 认证已禁用，任何请求都将被接受。")


    # === 简化配置预设推导 ===
    # 计算可用 GPU 显存（或读取用户提供的 GPU_VRAM_GB），结合 PRESET 设置其它参数
    try:
        detected_vram_gb = None
        if check_cuda_compatibility():
            _, _, total_gb = get_gpu_memory_usage()
            detected_vram_gb = round(total_gb)
    except Exception:
        detected_vram_gb = None

    gpu_vram_gb = None
    try:
        gpu_vram_gb = int(GPU_VRAM_GB_ENV) if GPU_VRAM_GB_ENV else detected_vram_gb
    except Exception:
        gpu_vram_gb = detected_vram_gb

    preset = PRESET if PRESET in ['speed', 'balanced', 'quality', 'simple'] else 'balanced'
    if preset == 'simple':
        preset = 'balanced'

    # Derive parameters based on preset and VRAM (only effective when user doesn't explicitly override)
    def set_if_default(name: str, current, value):
        # Only replace default when environment variable is not explicitly set
        if os.environ.get(name) is None:
            return value
        return current

    # CHUNK_MINITE
    if gpu_vram_gb is not None:
        if preset == 'speed':
            CHUNK_MINITE = set_if_default('CHUNK_MINITE', CHUNK_MINITE,  min(20, 10 if gpu_vram_gb < 12 else 15))
        elif preset == 'quality':
            CHUNK_MINITE = set_if_default('CHUNK_MINITE', CHUNK_MINITE,  max(6, 8 if gpu_vram_gb >= 8 else 6))
        else:  # balanced
            CHUNK_MINITE = set_if_default('CHUNK_MINITE', CHUNK_MINITE,  8 if gpu_vram_gb < 8 else 10)

    # Concurrency and GPU memory fraction
    if preset == 'speed':
        MAX_CONCURRENT_INFERENCES = set_if_default('MAX_CONCURRENT_INFERENCES', MAX_CONCURRENT_INFERENCES, 2 if (gpu_vram_gb and gpu_vram_gb >= 16) else 1)
        GPU_MEMORY_FRACTION = set_if_default('GPU_MEMORY_FRACTION', GPU_MEMORY_FRACTION, 0.95)
        DECODING_STRATEGY = set_if_default('DECODING_STRATEGY', DECODING_STRATEGY, 'greedy')
    elif preset == 'quality':
        MAX_CONCURRENT_INFERENCES = set_if_default('MAX_CONCURRENT_INFERENCES', MAX_CONCURRENT_INFERENCES, 1)
        GPU_MEMORY_FRACTION = set_if_default('GPU_MEMORY_FRACTION', GPU_MEMORY_FRACTION, 0.90)
        DECODING_STRATEGY = set_if_default('DECODING_STRATEGY', DECODING_STRATEGY, 'beam')
        RNNT_BEAM_SIZE = set_if_default('RNNT_BEAM_SIZE', RNNT_BEAM_SIZE, 4 if (gpu_vram_gb and gpu_vram_gb >= 8) else 2)
    else:  # balanced
        MAX_CONCURRENT_INFERENCES = set_if_default('MAX_CONCURRENT_INFERENCES', MAX_CONCURRENT_INFERENCES, 1)
        GPU_MEMORY_FRACTION = set_if_default('GPU_MEMORY_FRACTION', GPU_MEMORY_FRACTION, 0.92 if (gpu_vram_gb and gpu_vram_gb >= 12) else 0.90)
        DECODING_STRATEGY = set_if_default('DECODING_STRATEGY', DECODING_STRATEGY, 'greedy')

    # Record final preset
    print(f"Preset: {preset}  | GPU_VRAM_GB: {gpu_vram_gb if gpu_vram_gb is not None else 'unknown'}")
    print(f"Derived: CHUNK_MINITE={CHUNK_MINITE}, MAX_CONCURRENT_INFERENCES={MAX_CONCURRENT_INFERENCES}, GPU_MEMORY_FRACTION={GPU_MEMORY_FRACTION}, DECODING_STRATEGY={DECODING_STRATEGY}")

    # Update concurrency semaphore to match derived value
    try:
        new_max_conc = int(MAX_CONCURRENT_INFERENCES) if isinstance(MAX_CONCURRENT_INFERENCES, (int, float, str)) else 1
        if new_max_conc < 1:
            new_max_conc = 1
        globals()['inference_semaphore'] = threading.Semaphore(new_max_conc)
    except Exception as e:
        print(f"⚠️ Failed to initialize concurrency semaphore, using default 1: {e}")
        globals()['inference_semaphore'] = threading.Semaphore(1)

    print(f"🚀 Starting server...")
    print(f"API endpoint: POST http://{host}:{port}/v1/audio/transcriptions")
    print(f"Service will run with {threads} threads.")
    print("")
    print("=== GPU Memory Optimization Configuration ===")
    print(f"Aggressive GPU memory cleanup: {'Enabled' if AGGRESSIVE_MEMORY_CLEANUP else 'Disabled'}")
    print(f"Gradient checkpointing: {'Enabled' if ENABLE_GRADIENT_CHECKPOINTING else 'Disabled'}")
    print(f"Force cleanup threshold: {FORCE_CLEANUP_THRESHOLD*100:.0f}%")
    print(f"Max chunk memory: {MAX_CHUNK_MEMORY_MB}MB")
    print(f"Default chunk duration: {CHUNK_MINITE} minutes")
    print("=" * 25)
    print("")
    print("=== Idle Resource Optimization Configuration ===")
    print(f"Model idle timeout: {IDLE_TIMEOUT_MINUTES} minutes")
    print(f"Idle memory cleanup interval: {IDLE_MEMORY_CLEANUP_INTERVAL} seconds")
    print(f"Monitoring interval: {IDLE_MONITORING_INTERVAL} seconds")
    # Initialize CUDA compatibility check
    print("Checking CUDA compatibility...")
    cuda_available = check_cuda_compatibility()
    
    if cuda_available:
        _, _, total_memory = get_gpu_memory_usage()
        print(f"GPU total VRAM: {total_memory:.1f}GB")
    else:
        memory = psutil.virtual_memory()
        print(f"System memory: {memory.total/1024**3:.1f}GB")
    print("=" * 25)
    print("")
    print("=== Tensor Core Configuration ===")
    print(f"Tensor Core: {'Enabled' if ENABLE_TENSOR_CORE else 'Disabled'}")
    print(f"cuDNN Benchmark: {'Enabled' if ENABLE_CUDNN_BENCHMARK else 'Disabled'}")
    print(f"Precision mode: {TENSOR_CORE_PRECISION}")
    if cuda_available:
        print(f"GPU support: {get_tensor_core_info()}")
    else:
        print("GPU support: N/A - CUDA unavailable or incompatible")
    print("=" * 25)
    print("")
    print("=== Sentence Integrity Optimization ===")
    print(f"Overlapping chunking: {'Enabled' if ENABLE_OVERLAP_CHUNKING else 'Disabled'}")
    if ENABLE_OVERLAP_CHUNKING:
        print(f"Overlap duration: {CHUNK_OVERLAP_SECONDS}s")
        print(f"Boundary threshold: {SENTENCE_BOUNDARY_THRESHOLD}")
    print("=" * 25)
    serve(app, host=host, port=port, threads=threads)