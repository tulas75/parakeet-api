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

import nemo.collections.asr as nemo_asr
import torch
import gc

# --- 全局设置与模型状态 ---
asr_model = None
last_request_time = None
model_lock = threading.Lock()


# 确保临时上传目录存在
if not os.path.exists('/app/temp_uploads'):
    os.makedirs('/app/temp_uploads')

def load_model_if_needed():
    """按需加载模型，如果模型未加载，则进行加载。"""
    global asr_model
    # 使用锁确保多线程环境下模型只被加载一次
    with model_lock:
        if asr_model is None:
            print("="*50)
            print("模型当前未加载，正在从磁盘加载...")
            print("模型名称: nvidia/parakeet-tdt-0.6b-v2")
            try:
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

                if torch.cuda.is_available():
                    print(f"检测到 CUDA，将使用 GPU 加速并开启半精度(FP16)优化。")
                    # 先在CPU上加载模型，然后转移到GPU并启用FP16
                    loaded_model = nemo_asr.models.ASRModel.restore_from(restore_path=model_path, map_location=torch.device('cpu'))
                    loaded_model = loaded_model.cuda()
                    loaded_model = loaded_model.half()
                else:
                    print("未检测到 CUDA，将使用 CPU 运行。")
                    loaded_model = nemo_asr.models.ASRModel.restore_from(restore_path=model_path)
                
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
    global asr_model, last_request_time
    with model_lock:
        if asr_model is not None:
            print(f"模型闲置超过 {IDLE_TIMEOUT_MINUTES} 分钟，正在从显存中卸载...")
            asr_model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
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
        CHUNK_DURATION_SECONDS = CHUNK_MINITE * 60  
        total_duration = get_audio_duration(target_wav_path)
        if total_duration == 0:
            return jsonify({"error": "无法处理时长为0的音频"}), 400

        # 检查是否需要切片，如果音频时长小于切片阈值，则直接处理
        if total_duration <= CHUNK_DURATION_SECONDS:
            print(f"[{unique_id}] 文件总时长: {total_duration:.2f}s. 小于切片阈值({CHUNK_DURATION_SECONDS}s)，无需切片。")
            chunk_paths = [target_wav_path]
            num_chunks = 1
        else:
            num_chunks = math.ceil(total_duration / CHUNK_DURATION_SECONDS)
            chunk_paths = []
            print(f"[{unique_id}] 文件总时长: {total_duration:.2f}s. 将切分为 {num_chunks} 个片段。")
            
            for i in range(num_chunks):
                start_time = i * CHUNK_DURATION_SECONDS
                chunk_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}_chunk_{i}.wav")
                chunk_paths.append(chunk_path)
                temp_files_to_clean.append(chunk_path)
                
                print(f"[{unique_id}] 正在创建切片 {i+1}/{num_chunks}...")
                chunk_command = [
                    'ffmpeg', '-y', '-i', target_wav_path,
                    '-ss', str(start_time),
                    '-t', str(CHUNK_DURATION_SECONDS),
                    '-c', 'copy',
                    chunk_path
                ]
                subprocess.run(chunk_command, capture_output=True, text=True)
            
        # --- 4. 循环转录并合并结果 ---
        all_segments = []
        all_words = []
        cumulative_time_offset = 0.0

        for i, chunk_path in enumerate(chunk_paths):
            print(f"[{unique_id}] 正在转录切片 {i+1}/{num_chunks}...")
            
            # 对当前切片进行转录
            # 使用 with torch.cuda.amp.autocast() 在半精度下运行推理
            if torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    output = local_asr_model.transcribe([chunk_path], timestamps=True)
            else:
                 output = local_asr_model.transcribe([chunk_path], timestamps=True)

            # 立即清理显存，避免累积
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            if output and output[0].timestamp:
                # 修正并收集 segment 时间戳
                if 'segment' in output[0].timestamp:
                    for seg in output[0].timestamp['segment']:
                        seg['start'] += cumulative_time_offset
                        seg['end'] += cumulative_time_offset
                        all_segments.append(seg)
                
                # 修正并收集 word 时间戳
                if 'word' in output[0].timestamp:
                     for word in output[0].timestamp['word']:
                        word['start'] += cumulative_time_offset
                        word['end'] += cumulative_time_offset
                        all_words.append(word)

            # 更新下一个切片的时间偏移量
            # 使用实际切片时长来更新，更精确
            chunk_actual_duration = get_audio_duration(chunk_path)
            cumulative_time_offset += chunk_actual_duration

        print(f"[{unique_id}] 所有切片转录完成，正在合并结果。")

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
        
        # --- 7. 强制清理显存，避免累积 ---
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print(f"[{unique_id}] 显存已清理。")


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
    serve(app, host=host, port=port, threads=threads)