import os,sys,json,math

# 设置环境变量来解决numba缓存问题
os.environ['NUMBA_CACHE_DIR'] = '/tmp/numba_cache'
os.environ['NUMBA_DISABLE_JIT'] = '0'

host = '0.0.0.0'
port = 5092
threads = 4
# 默认按照N分钟将音视频裁切为多段，减少显存占用。现在可以通过环境变量 CHUNK_MINITE 来调整。
# 8G显存建议设置为 10-15 分钟以获得最佳性能。
CHUNK_MINITE = int(os.environ.get('CHUNK_MINITE', '10'))
# 服务闲置N分钟后自动卸载模型以释放显存，设置为0则禁用
IDLE_TIMEOUT_MINUTES = int(os.environ.get('IDLE_TIMEOUT_MINUTES', '30'))
# 懒加载开关，默认为 true。设置为 'false' 可在启动时预加载模型。
ENABLE_LAZY_LOAD = os.environ.get('ENABLE_LAZY_LOAD', 'true').lower() not in ['false', '0', 'f']
# Whisper 兼容的 API Key。如果留空，则不进行身份验证。
API_KEY = os.environ.get('API_KEY', None)
import shutil
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
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
# HF_HOME is set in the Dockerfile
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = 'true'
# PATH for ffmpeg is handled by the Docker image's system PATH

import nemo.collections.asr as nemo_asr  # type: ignore
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import gc
import psutil

# --- 全局设置与模型状态 ---
asr_model = None
last_request_time = None
model_lock = threading.Lock()
cuda_available = False  # 全局CUDA兼容性标志

# 显存优化配置
AGGRESSIVE_MEMORY_CLEANUP = os.environ.get('AGGRESSIVE_MEMORY_CLEANUP', 'true').lower() in ['true', '1', 't']
ENABLE_GRADIENT_CHECKPOINTING = os.environ.get('ENABLE_GRADIENT_CHECKPOINTING', 'true').lower() in ['true', '1', 't']
MAX_CHUNK_MEMORY_MB = int(os.environ.get('MAX_CHUNK_MEMORY_MB', '1500'))
FORCE_CLEANUP_THRESHOLD = float(os.environ.get('FORCE_CLEANUP_THRESHOLD', '0.8'))

# Tensor Core 优化配置
ENABLE_TENSOR_CORE = os.environ.get('ENABLE_TENSOR_CORE', 'true').lower() in ['true', '1', 't']
ENABLE_CUDNN_BENCHMARK = os.environ.get('ENABLE_CUDNN_BENCHMARK', 'true').lower() in ['true', '1', 't']
TENSOR_CORE_PRECISION = os.environ.get('TENSOR_CORE_PRECISION', 'highest')  # highest, high, medium

# 句子完整性优化配置
ENABLE_OVERLAP_CHUNKING = os.environ.get('ENABLE_OVERLAP_CHUNKING', 'true').lower() in ['true', '1', 't']
CHUNK_OVERLAP_SECONDS = float(os.environ.get('CHUNK_OVERLAP_SECONDS', '30'))  # 重叠时长
SENTENCE_BOUNDARY_THRESHOLD = float(os.environ.get('SENTENCE_BOUNDARY_THRESHOLD', '0.5'))  # 句子边界检测阈值


# 确保临时上传目录存在
if not os.path.exists('/app/temp_uploads'):
    os.makedirs('/app/temp_uploads')

def setup_tensor_core_optimization():
    """配置Tensor Core优化设置"""
    global cuda_available
    if not cuda_available:
        print("CUDA不可用，跳过Tensor Core优化配置")
        return
    
    print("正在配置 Tensor Core 优化...")
    
    try:
        # 启用 cuDNN benchmark 模式
        if ENABLE_CUDNN_BENCHMARK:
            cudnn.benchmark = True
            cudnn.deterministic = False  # 为了性能，允许非确定性
            print("✅ cuDNN benchmark 已启用")
        else:
            cudnn.benchmark = False
            cudnn.deterministic = True
            print("❌ cuDNN benchmark 已禁用（确定性模式）")
        
        # 启用 cuDNN 允许 TensorCore
        if ENABLE_TENSOR_CORE:
            cudnn.allow_tf32 = True  # 允许TF32（A100等支持）
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("✅ Tensor Core (TF32) 已启用")
        else:
            cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            print("❌ Tensor Core 已禁用")
        
        # 设置 Tensor Core 精度策略
        if TENSOR_CORE_PRECISION == 'highest':
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
            print("✅ 设置为最高精度模式")
        elif TENSOR_CORE_PRECISION == 'high':
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
            print("✅ 设置为高精度模式")
        else:  # medium
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
            print("✅ 设置为中等精度模式")
        
        # 设置内存分配策略以优化 Tensor Core 使用
        torch.cuda.set_per_process_memory_fraction(0.95)  # 使用95%的显存
        print("✅ GPU 内存分配策略已优化")
    except Exception as e:
        print(f"⚠️ Tensor Core优化配置失败: {e}")

def get_tensor_core_info():
    """获取 Tensor Core 支持信息"""
    global cuda_available
    if not cuda_available:
        return "N/A - CUDA不可用"
    
    try:
        device = torch.cuda.get_device_properties(0)
        major, minor = device.major, device.minor
        
        # 检测 Tensor Core 支持
        if major >= 7:  # V100, T4, RTX 20/30/40系列等
            if major == 7:
                return f"✅ Tensor Core 1.0 (计算能力 {major}.{minor})"
            elif major == 8:
                if minor >= 0:
                    return f"✅ Tensor Core 2.0 + TF32 (计算能力 {major}.{minor})"
                else:
                    return f"✅ Tensor Core 2.0 (计算能力 {major}.{minor})"
            elif major >= 9:
                return f"✅ Tensor Core 3.0+ (计算能力 {major}.{minor})"
        elif major >= 6:  # P100等
            return f"⚠️ 有限Tensor Core支持 (计算能力 {major}.{minor})"
        else:
            return f"❌ 不支持Tensor Core (计算能力 {major}.{minor})"
        
        return f"未知 (计算能力 {major}.{minor})"
    except Exception as e:
        return f"❌ 获取GPU信息失败: {e}"

def optimize_tensor_operations():
    """优化张量操作以更好地利用 Tensor Core"""
    global cuda_available
    if not cuda_available:
        print("CUDA不可用，跳过Tensor Core预热")
        return
    
    try:
        # 设置优化的 CUDA 流
        torch.cuda.set_sync_debug_mode(0)  # 禁用同步调试以提升性能
        
        # 预热GPU，确保Tensor Core正确激活
        # 创建一些对齐到8/16倍数的矩阵进行预热
        device = torch.cuda.current_device()
        dummy_a = torch.randn(128, 128, device=device, dtype=torch.float16)
        dummy_b = torch.randn(128, 128, device=device, dtype=torch.float16)
        
        # 执行矩阵乘法预热Tensor Core
        with torch.cuda.amp.autocast():
            _ = torch.matmul(dummy_a, dummy_b)
        
        torch.cuda.synchronize()
        del dummy_a, dummy_b
        torch.cuda.empty_cache()
        print("✅ Tensor Core 预热完成")
    except Exception as e:
        print(f"⚠️ Tensor Core 预热失败: {e}")

def detect_sentence_boundaries(text: str) -> list:
    """检测句子边界，返回句子结束位置列表"""
    import re
    
    # 中英文句号、问号、感叹号等
    sentence_endings = re.finditer(r'[.!?。！？]+[\s]*', text)
    boundaries = [match.end() for match in sentence_endings]
    return boundaries

def find_best_split_point(segments: list, target_time: float, tolerance: float = 2.0) -> int:
    """在目标时间附近找到最佳的句子分割点"""
    if not segments:
        return 0
    
    best_index = 0
    min_distance = float('inf')
    
    # 寻找最接近目标时间的句子结束点
    for i, segment in enumerate(segments):
        segment_end = segment.get('end', 0)
        distance = abs(segment_end - target_time)
        
        # 检查是否是句子结束（包含标点符号）
        text = segment.get('segment', '').strip()
        if text and (text.endswith('.') or text.endswith('。') or 
                     text.endswith('!') or text.endswith('！') or
                     text.endswith('?') or text.endswith('？')):
            # 句子结束点，权重更高
            distance *= 0.5
        
        if distance < min_distance and distance <= tolerance:
            min_distance = distance
            best_index = i + 1  # 返回下一个段落的索引
    
    return min(best_index, len(segments))

def merge_overlapping_segments(all_segments: list, chunk_boundaries: list, overlap_seconds: float) -> list:
    """合并重叠区域的segments，去除重复内容"""
    if not ENABLE_OVERLAP_CHUNKING or len(chunk_boundaries) <= 1:
        return all_segments
    
    merged_segments = []
    current_chunk_segments = []
    current_chunk_index = 0
    
    print(f"开始合并 {len(all_segments)} 个segments，chunk边界: {chunk_boundaries}")
    
    for segment in all_segments:
        segment_start = segment['start']
        segment_end = segment['end']
        
        # 确定当前segment属于哪个chunk
        while (current_chunk_index < len(chunk_boundaries) - 1 and 
               segment_start >= chunk_boundaries[current_chunk_index + 1] - overlap_seconds):
            # 处理前一个chunk的segments
            if current_chunk_segments:
                # 处理重叠区域
                overlap_start = chunk_boundaries[current_chunk_index + 1] - overlap_seconds
                processed_segments = process_chunk_segments(
                    current_chunk_segments, overlap_start, overlap_seconds
                )
                merged_segments.extend(processed_segments)
                current_chunk_segments = []
            
            current_chunk_index += 1
        
        current_chunk_segments.append(segment)
    
    # 处理最后一个chunk
    if current_chunk_segments:
        merged_segments.extend(current_chunk_segments)
    
    print(f"合并完成，最终 {len(merged_segments)} 个segments")
    return merged_segments

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
    """创建带重叠的chunk时间段"""
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
        
        # 下一个chunk的开始时间（考虑重叠）
        if chunk_end >= total_duration:
            break
            
        current_start = chunk_end - overlap_seconds
        
    print(f"创建了 {len(chunks)} 个重叠chunks:")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i}: {chunk['start']:.1f}s - {chunk['end']:.1f}s (时长: {chunk['duration']:.1f}s)")
    
    return chunks

def check_cuda_compatibility():
    """检查CUDA兼容性，如果不兼容则禁用CUDA"""
    global cuda_available
    
    try:
        if not torch.cuda.is_available():
            print("CUDA不可用，将使用CPU模式")
            cuda_available = False
            return False
        
        # 尝试获取设备数量来测试CUDA兼容性
        device_count = torch.cuda.device_count()
        if device_count == 0:
            print("未检测到CUDA设备，将使用CPU模式")
            cuda_available = False
            return False
            
        # 尝试获取设备属性来进一步测试兼容性
        device_props = torch.cuda.get_device_properties(0)
        print(f"✅ 检测到兼容的GPU: {device_props.name}")
        cuda_available = True
        return True
    except RuntimeError as e:
        if "forward compatibility was attempted on non supported HW" in str(e):
            print("⚠️ CUDA兼容性错误: GPU硬件不支持当前CUDA版本")
            print("这通常是因为主机的GPU驱动版本过旧，不支持容器中的CUDA 12.3版本")
            print("将自动切换到CPU模式运行")
        elif "CUDA" in str(e):
            print(f"⚠️ CUDA初始化失败: {e}")
            print("将自动切换到CPU模式运行")
        else:
            print(f"⚠️ 未知CUDA错误: {e}")
            print("将自动切换到CPU模式运行")
        
        cuda_available = False
        return False
    except Exception as e:
        print(f"⚠️ GPU兼容性检查失败: {e}")
        print("将自动切换到CPU模式运行")
        cuda_available = False
        return False

def get_gpu_memory_usage():
    """获取GPU显存使用情况"""
    global cuda_available
    if not cuda_available:
        return 0, 0, 0
    
    try:
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        return allocated, reserved, total
    except Exception as e:
        print(f"⚠️ 获取GPU内存信息失败: {e}")
        return 0, 0, 0

def aggressive_memory_cleanup():
    """激进的显存清理函数"""
    global cuda_available
    if cuda_available:
        try:
            # 清空CUDA缓存
            torch.cuda.empty_cache()
            # 同步所有CUDA操作
            torch.cuda.synchronize()
            # 重置峰值内存统计
            torch.cuda.reset_peak_memory_stats()
        except Exception as e:
            print(f"⚠️ CUDA清理操作失败: {e}")
    
    # 强制Python垃圾回收
    for _ in range(3):
        gc.collect()
    
    if cuda_available:
        try:
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"⚠️ CUDA缓存清理失败: {e}")

def should_force_cleanup():
    """检查是否需要强制清理显存"""
    global cuda_available
    if not cuda_available:
        return False
    
    allocated, reserved, total = get_gpu_memory_usage()
    usage_ratio = allocated / total if total > 0 else 0
    return usage_ratio > FORCE_CLEANUP_THRESHOLD

def optimize_model_for_inference(model):
    """优化模型以减少推理时的显存占用"""
    if model is None:
        return model
    
    # 设置为评估模式
    model.eval()
    
    # 启用梯度检查点（如果支持）
    if ENABLE_GRADIENT_CHECKPOINTING and hasattr(model, 'encoder'):
        try:
            if hasattr(model.encoder, 'use_gradient_checkpointing'):
                model.encoder.use_gradient_checkpointing = True
            elif hasattr(model.encoder, 'gradient_checkpointing'):
                model.encoder.gradient_checkpointing = True
        except Exception as e:
            print(f"无法启用梯度检查点: {e}")
    
    # 禁用自动求导（推理时不需要梯度）
    for param in model.parameters():
        param.requires_grad = False
    
    return model

def create_streaming_config():
    """创建流式处理配置以减少显存占用"""
    return {
        'batch_size': 1,  # 单批处理减少显存占用
        'num_workers': 0,  # 避免多进程带来的额外内存开销
        'pin_memory': False,  # 不使用锁页内存以节省系统内存
        'drop_last': False,
        'persistent_workers': False  # 不保持worker进程
    }

def load_model_if_needed():
    """按需加载模型，如果模型未加载，则进行加载。"""
    global asr_model, cuda_available
    # 使用锁确保多线程环境下模型只被加载一次
    with model_lock:
        if asr_model is None:
            print("="*50)
            print("模型当前未加载，正在从磁盘加载...")
            print("模型名称: nvidia/parakeet-tdt-0.6b-v2")
            try:
                # 首先检查CUDA兼容性
                cuda_available = check_cuda_compatibility()
                
                # 确保numba缓存目录存在
                numba_cache_dir = os.environ.get('NUMBA_CACHE_DIR', '/tmp/numba_cache')
                if not os.path.exists(numba_cache_dir):
                    os.makedirs(numba_cache_dir, exist_ok=True)
                    os.chmod(numba_cache_dir, 0o777)
                
                model_path = "/app/models/parakeet-tdt-0.6b-v2.nemo"
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"模型文件未找到: {model_path}，请确认 models 文件夹已正确挂载。")

                # 检查文件权限
                if not os.access(model_path, os.R_OK):
                    raise PermissionError(f"无法读取模型文件: {model_path}，请检查文件权限。")

                if cuda_available:
                    print(f"✅ 检测到兼容的CUDA环境，将使用 GPU 加速并开启半精度(FP16)优化。")
                    
                    # 设置 Tensor Core 优化
                    setup_tensor_core_optimization()
                    optimize_tensor_operations()
                    
                    # 显示 GPU 和 Tensor Core 信息
                    device_info = torch.cuda.get_device_properties(0)
                    print(f"GPU: {device_info.name}")
                    print(f"Tensor Core 支持: {get_tensor_core_info()}")
                    
                    # 先在CPU上加载模型，然后转移到GPU并启用FP16
                    loaded_model = nemo_asr.models.ASRModel.restore_from(restore_path=model_path, map_location=torch.device('cpu'))
                    loaded_model = loaded_model.cuda()
                    loaded_model = loaded_model.half()
                    
                    # 应用推理优化
                    loaded_model = optimize_model_for_inference(loaded_model)
                    
                    # 显示显存使用情况
                    allocated, reserved, total = get_gpu_memory_usage()
                    print(f"模型加载后显存使用: {allocated:.2f}GB / {total:.2f}GB ({allocated/total*100:.1f}%)")
                else:
                    print("🔄 使用 CPU 模式运行。")
                    print("注意: CPU模式下推理速度会较慢，建议使用兼容的GPU。")
                    loaded_model = nemo_asr.models.ASRModel.restore_from(restore_path=model_path)
                    loaded_model = optimize_model_for_inference(loaded_model)
                
                asr_model = loaded_model
                print("✅ NeMo ASR 模型加载成功！")
                print("="*50)
            except Exception as e:
                print(f"❌ 模型加载失败: {e}")
                print("="*50)
                import traceback
                traceback.print_exc()
                # 向上抛出异常，以便接口可以捕获并返回错误信息
                raise e
    return asr_model

def unload_model():
    """从内存/显存中卸载模型。"""
    global asr_model, last_request_time, cuda_available
    with model_lock:
        if asr_model is not None:
            print(f"模型闲置超过 {IDLE_TIMEOUT_MINUTES} 分钟，正在从内存中卸载...")
            
            # 显示卸载前的显存使用
            if cuda_available:
                allocated_before, _, total = get_gpu_memory_usage()
                print(f"卸载前显存使用: {allocated_before:.2f}GB / {total:.2f}GB")
            
            asr_model = None
            aggressive_memory_cleanup()
            
            # 显示卸载后的显存使用
            if cuda_available:
                allocated_after, _, total = get_gpu_memory_usage()
                print(f"卸载后显存使用: {allocated_after:.2f}GB / {total:.2f}GB")
                print(f"释放显存: {allocated_before - allocated_after:.2f}GB")
            
            last_request_time = None # 重置计时器，防止重复卸载
            print("✅ 模型已成功卸载。")

def model_cleanup_checker():
    """后台线程，周期性检查模型是否闲置过久并执行卸载。"""
    while True:
        # 每 60 秒检查一次
        time.sleep(60)
        if asr_model is not None and last_request_time is not None:
            idle_duration = (datetime.datetime.now() - last_request_time).total_seconds()
            if idle_duration > IDLE_TIMEOUT_MINUTES * 60:
                unload_model()


# --- Flask 应用初始化 ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/app/temp_uploads'
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
        
        if text: # 仅添加有内容的字幕
            srt_content.append(str(i + 1))
            srt_content.append(f"{start_time} --> {end_time}")
            srt_content.append(text)
            srt_content.append("") # 空行分隔
            
    return "\n".join(srt_content)

# --- Flask 路由 ---

@app.route('/v1/audio/transcriptions', methods=['POST'])
def transcribe_audio():
    """
    兼容 OpenAI 的语音识别接口，支持长音频分片处理。
    """
    # --- -1. API Key 认证 ---
    if API_KEY:
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({"error": "Authorization header is missing or invalid. It must be in 'Bearer <key>' format."}), 401
        
        provided_key = auth_header.split(' ')[1]
        if provided_key != API_KEY:
            return jsonify({"error": "Invalid API key."}), 401

    # --- 0. 确保模型加载并更新时间戳 ---
    try:
        # 如果懒加载启用，则按需加载；否则，直接使用已加载的全局模型
        local_asr_model = load_model_if_needed() if ENABLE_LAZY_LOAD else asr_model
        if not local_asr_model:
             # 此情况涵盖了懒加载失败和预加载失败两种场景
             return jsonify({"error": "模型加载失败或尚未加载，无法处理请求"}), 500
    except Exception as e:
        return jsonify({"error": f"模型加载时发生严重错误: {e}"}), 500
    
    # 如果启用了懒加载，则更新最后请求时间
    if ENABLE_LAZY_LOAD:
        global last_request_time
        last_request_time = datetime.datetime.now()


    # --- 1. 基本校验 ---
    if 'file' not in request.files:
        return jsonify({"error": "请求中未找到文件部分"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "未选择文件"}), 400
    if not shutil.which('ffmpeg'):
        return jsonify({"error": "FFmpeg 未安装或未在系统 PATH 中"}), 500
    if not shutil.which('ffprobe'):
        return jsonify({"error": "ffprobe 未安装或未在系统 PATH 中"}), 500

    # 获取请求参数
    model_name = request.form.get('model', 'whisper-1')
    response_format = request.form.get('response_format', 'json')  # 支持 json, text, srt, verbose_json, vtt
    language = request.form.get('language', None)
    prompt = request.form.get('prompt', None)
    temperature = float(request.form.get('temperature', 0))
    
    print(f"接收到请求，模型: '{model_name}', 响应格式: '{response_format}'")

    original_filename = secure_filename(file.filename)
    unique_id = str(uuid.uuid4())
    temp_original_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}_{original_filename}")
    target_wav_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}.wav")
    
    # 用于清理所有临时文件的列表
    temp_files_to_clean = []

    try:
        # --- 2. 保存并统一转换为 16k 单声道 WAV ---
        file.save(temp_original_path)
        temp_files_to_clean.append(temp_original_path)
        
        print(f"[{unique_id}] 正在将 '{original_filename}' 转换为标准 WAV 格式...")
        ffmpeg_command = [
            'ffmpeg', '-y', '-i', temp_original_path,
            '-ac', '1', '-ar', '16000', target_wav_path
        ]
        result = subprocess.run(ffmpeg_command, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"FFmpeg 错误: {result.stderr}")
            return jsonify({"error": "文件转换失败", "details": result.stderr}), 500
        temp_files_to_clean.append(target_wav_path)

        # --- 3. 音频切片 (Chunking) ---
        # 动态调整chunk大小基于显存使用情况
        if cuda_available:
            allocated, _, total = get_gpu_memory_usage()
            memory_usage_ratio = allocated / total if total > 0 else 0
            
            if memory_usage_ratio > 0.6:  # 如果显存使用超过60%
                # 减少chunk大小以降低显存压力
                adjusted_chunk_minutes = max(3, CHUNK_MINITE - 2)
                print(f"[{unique_id}] 显存使用较高({memory_usage_ratio*100:.1f}%)，调整chunk大小从 {CHUNK_MINITE} 分钟到 {adjusted_chunk_minutes} 分钟")
                CHUNK_DURATION_SECONDS = adjusted_chunk_minutes * 60
            else:
                CHUNK_DURATION_SECONDS = CHUNK_MINITE * 60
        else:
            # CPU模式下使用较小的chunk以避免内存不足
            cpu_chunk_minutes = max(3, CHUNK_MINITE // 2)  # CPU模式减半chunk大小
            print(f"[{unique_id}] CPU模式，调整chunk大小到 {cpu_chunk_minutes} 分钟")
            CHUNK_DURATION_SECONDS = cpu_chunk_minutes * 60
            
        total_duration = get_audio_duration(target_wav_path)
        if total_duration == 0:
            return jsonify({"error": "无法处理时长为0的音频"}), 400

        # 检查是否需要切片，如果音频时长小于切片阈值，则直接处理
        if total_duration <= CHUNK_DURATION_SECONDS:
            print(f"[{unique_id}] 文件总时长: {total_duration:.2f}s. 小于切片阈值({CHUNK_DURATION_SECONDS}s)，无需切片。")
            chunk_paths = [target_wav_path]
            chunk_info_list = [{'start': 0, 'end': total_duration, 'duration': total_duration}]
            num_chunks = 1
        else:
            # 使用重叠分割策略
            if ENABLE_OVERLAP_CHUNKING:
                print(f"[{unique_id}] 启用重叠分割模式，重叠时长: {CHUNK_OVERLAP_SECONDS}s")
                chunk_info_list = create_overlap_chunks(total_duration, CHUNK_DURATION_SECONDS, CHUNK_OVERLAP_SECONDS)
            else:
                # 传统硬分割
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
            print(f"[{unique_id}] 文件总时长: {total_duration:.2f}s. 将切分为 {num_chunks} 个片段。")
            
            for i, chunk_info in enumerate(chunk_info_list):
                chunk_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}_chunk_{i}.wav")
                chunk_paths.append(chunk_path)
                temp_files_to_clean.append(chunk_path)
                
                start_time = chunk_info['start']
                duration = chunk_info['duration']
                
                print(f"[{unique_id}] 正在创建切片 {i+1}/{num_chunks} ({start_time:.1f}s - {chunk_info['end']:.1f}s)...")
                chunk_command = [
                    'ffmpeg', '-y', '-i', target_wav_path,
                    '-ss', str(start_time),
                    '-t', str(duration),
                    '-c', 'copy',
                    chunk_path
                ]
                result = subprocess.run(chunk_command, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"[{unique_id}] ⚠️ 创建切片 {i+1} 时出现警告: {result.stderr}")
                    # 继续处理，不中断
            
        # --- 4. 循环转录并合并结果 ---
        all_segments = []
        all_words = []
        chunk_boundaries = []

        for i, (chunk_path, chunk_info) in enumerate(zip(chunk_paths, chunk_info_list)):
            print(f"[{unique_id}] 正在转录切片 {i+1}/{num_chunks}...")
            
            # 检查显存使用情况，如果过高则强制清理
            if should_force_cleanup():
                print(f"[{unique_id}] 显存使用过高，执行强制清理...")
                aggressive_memory_cleanup()
            
            # 显示当前显存/内存使用
            if cuda_available:
                allocated, _, total = get_gpu_memory_usage()
                print(f"[{unique_id}] 处理切片 {i+1} 前显存使用: {allocated:.2f}GB / {total:.2f}GB")
            else:
                # 显示CPU内存使用
                memory = psutil.virtual_memory()
                print(f"[{unique_id}] 处理切片 {i+1} 前内存使用: {memory.used/1024**3:.2f}GB / {memory.total/1024**3:.2f}GB ({memory.percent:.1f}%)")
            
            # 对当前切片进行转录
            # 使用 with torch.cuda.amp.autocast() 在半精度下运行推理
            with torch.no_grad():  # 确保不计算梯度
                if cuda_available:
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        output = local_asr_model.transcribe([chunk_path], timestamps=True)
                else:
                    # CPU模式下直接转录
                    output = local_asr_model.transcribe([chunk_path], timestamps=True)

            # 立即进行内存清理
            if AGGRESSIVE_MEMORY_CLEANUP:
                aggressive_memory_cleanup()
            else:
                if cuda_available:
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                gc.collect()
            
            # 记录chunk边界用于后续合并
            chunk_start_offset = chunk_info['start']
            chunk_boundaries.append(chunk_start_offset)
            
            if output and output[0].timestamp:
                # 修正并收集 segment 时间戳（使用chunk在原音频中的真实起始时间）
                if 'segment' in output[0].timestamp:
                    for seg in output[0].timestamp['segment']:
                        seg['start'] += chunk_start_offset
                        seg['end'] += chunk_start_offset
                        all_segments.append(seg)
                
                # 修正并收集 word 时间戳
                if 'word' in output[0].timestamp:
                     for word in output[0].timestamp['word']:
                        word['start'] += chunk_start_offset
                        word['end'] += chunk_start_offset
                        all_words.append(word)
            
            # 立即删除已处理的chunk文件以节省磁盘空间和内存
            if num_chunks > 1 and os.path.exists(chunk_path):
                try:
                    os.remove(chunk_path)
                    temp_files_to_clean.remove(chunk_path)
                    print(f"[{unique_id}] 已删除处理完成的切片文件: chunk_{i}")
                except Exception as e:
                    print(f"[{unique_id}] 删除切片文件时出错: {e}")

        print(f"[{unique_id}] 所有切片转录完成，正在合并结果。")
        
        # --- 4.5. 处理重叠区域并合并segments ---
        if ENABLE_OVERLAP_CHUNKING and len(chunk_boundaries) > 1:
            print(f"[{unique_id}] 处理重叠区域，去除重复内容...")
            all_segments = merge_overlapping_segments(all_segments, chunk_boundaries, CHUNK_OVERLAP_SECONDS)
            print(f"[{unique_id}] 重叠处理完成，最终segments数量: {len(all_segments)}")

        # --- 5. 格式化最终输出 ---
        if not all_segments:
            return jsonify({"error": "转录失败，模型未返回任何有效内容"}), 500

        # 构建完整的转录文本
        full_text = " ".join([seg['segment'].strip() for seg in all_segments if seg['segment'].strip()])
        
        # 根据 response_format 返回不同格式
        if response_format == 'text':
            return Response(full_text, mimetype='text/plain')
        elif response_format == 'srt':
            srt_result = segments_to_srt(all_segments)
            return Response(srt_result, mimetype='text/plain')
        elif response_format == 'vtt':
            vtt_result = segments_to_vtt(all_segments)
            return Response(vtt_result, mimetype='text/plain')
        elif response_format == 'verbose_json':
            # 详细的 JSON 格式，包含更多信息
            response_data = {
                "task": "transcribe",
                "language": language or "en",
                "duration": total_duration,
                "text": full_text,
                "segments": [
                    {
                        "id": i,
                        "seek": int(seg['start'] * 100),  # 转换为 centiseconds
                        "start": seg['start'],
                        "end": seg['end'],
                        "text": seg['segment'].strip(),
                        "tokens": [],  # NeMo 不提供 tokens，留空
                        "temperature": temperature,
                        "avg_logprob": -0.5,  # 模拟值
                        "compression_ratio": 1.0,  # 模拟值
                        "no_speech_prob": 0.0,  # 模拟值
                        "words": [
                            {
                                "word": word['word'],
                                "start": word['start'],
                                "end": word['end'],
                                "probability": 0.9  # 模拟值
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
            # 默认 JSON 格式 (response_format == 'json')
            response_data = {
                "text": full_text
            }
            return jsonify(response_data)

    except Exception as e:
        print(f"处理过程中发生严重错误: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "服务器内部错误", "details": str(e)}), 500
    finally:
        # --- 6. 清理所有临时文件 ---
        print(f"[{unique_id}] 清理临时文件...")
        for f_path in temp_files_to_clean:
            if os.path.exists(f_path):
                os.remove(f_path)
        print(f"[{unique_id}] 临时文件已清理。")
        
        # --- 7. 强制清理内存，避免累积 ---
        print(f"[{unique_id}] 执行最终内存清理...")
        if cuda_available:
            allocated_before, _, total = get_gpu_memory_usage()
            print(f"[{unique_id}] 清理前显存使用: {allocated_before:.2f}GB / {total:.2f}GB")
        else:
            memory_before = psutil.virtual_memory()
            print(f"[{unique_id}] 清理前内存使用: {memory_before.used/1024**3:.2f}GB / {memory_before.total/1024**3:.2f}GB")
        
        aggressive_memory_cleanup()
        
        if cuda_available:
            allocated_after, _, total = get_gpu_memory_usage()
            print(f"[{unique_id}] 清理后显存使用: {allocated_after:.2f}GB / {total:.2f}GB")
            if allocated_before > 0:
                print(f"[{unique_id}] 释放显存: {allocated_before - allocated_after:.2f}GB")
        else:
            memory_after = psutil.virtual_memory()
            print(f"[{unique_id}] 清理后内存使用: {memory_after.used/1024**3:.2f}GB / {memory_after.total/1024**3:.2f}GB")
        print(f"[{unique_id}] 内存清理完成。")


def segments_to_vtt(segments: list) -> str:
    """将 NeMo 的分段时间戳转换为 VTT 格式字符串"""
    vtt_content = ["WEBVTT", ""]
    
    for i, segment in enumerate(segments):
        start_time = format_vtt_time(segment['start'])
        end_time = format_vtt_time(segment['end'])
        text = segment['segment'].strip()
        
        if text:  # 仅添加有内容的字幕
            vtt_content.append(f"{start_time} --> {end_time}")
            vtt_content.append(text)
            vtt_content.append("")  # 空行分隔
            
    return "\n".join(vtt_content)


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


    print(f"🚀 服务器启动中...")
    print(f"API 端点: POST http://{host}:{port}/v1/audio/transcriptions")
    print(f"服务将使用 {threads} 个线程运行。")
    print("")
    print("=== 显存优化配置 ===")
    print(f"激进显存清理: {'启用' if AGGRESSIVE_MEMORY_CLEANUP else '禁用'}")
    print(f"梯度检查点: {'启用' if ENABLE_GRADIENT_CHECKPOINTING else '禁用'}")
    print(f"强制清理阈值: {FORCE_CLEANUP_THRESHOLD*100:.0f}%")
    print(f"最大chunk内存: {MAX_CHUNK_MEMORY_MB}MB")
    print(f"默认chunk时长: {CHUNK_MINITE} 分钟")
    # 初始化CUDA兼容性检查
    print("正在检查CUDA兼容性...")
    cuda_available = check_cuda_compatibility()
    
    if cuda_available:
        _, _, total_memory = get_gpu_memory_usage()
        print(f"GPU总显存: {total_memory:.1f}GB")
    else:
        memory = psutil.virtual_memory()
        print(f"系统内存: {memory.total/1024**3:.1f}GB")
    print("=" * 25)
    print("")
    print("=== Tensor Core 配置 ===")
    print(f"Tensor Core: {'启用' if ENABLE_TENSOR_CORE else '禁用'}")
    print(f"cuDNN Benchmark: {'启用' if ENABLE_CUDNN_BENCHMARK else '禁用'}")
    print(f"精度模式: {TENSOR_CORE_PRECISION}")
    if cuda_available:
        print(f"GPU支持: {get_tensor_core_info()}")
    else:
        print("GPU支持: N/A - CUDA不可用或不兼容")
    print("=" * 25)
    print("")
    print("=== 句子完整性优化 ===")
    print(f"重叠分割: {'启用' if ENABLE_OVERLAP_CHUNKING else '禁用'}")
    if ENABLE_OVERLAP_CHUNKING:
        print(f"重叠时长: {CHUNK_OVERLAP_SECONDS}s")
        print(f"边界阈值: {SENTENCE_BOUNDARY_THRESHOLD}")
    print("=" * 25)
    serve(app, host=host, port=port, threads=threads)