#!/usr/bin/env python3
"""
sherpa-onnx SenseVoice-Small 语音识别测试。

使用 sherpa_onnx.OfflineRecognizer.from_sense_voice 加载 SenseVoice-Small int8 模型，
配合 WebRTC VAD 实现麦克风"说话 → 静音 → 自动识别 → 输出带标点结果"的交互式识别。

SenseVoice-Small 特点：
  - 多语言（中/英/日/粤/韩），自带标点（无需额外 punc_model）
  - int8 量化，CPU 实时率 RTF < 0.1，非常快
  - 离线识别模式（配合 VAD 实现交互式段落识别）

模型下载：
  https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2

用法:
    # 在 hermes-voice-agent/ 目录下运行（src 已通过 pip install -e . 安装，无需 sys.path）
    # 麦克风 VAD 交互式识别（默认）
    python -u tests/test_sensevoice_asr.py --record

    # 自定义 VAD 参数
    python -u tests/test_sensevoice_asr.py --record --vad-timeout 1500 --vad-mode 2

    # 识别 WAV 文件
    python -u tests/test_sensevoice_asr.py --file test.wav

    # 生成测试音并识别
    python -u tests/test_sensevoice_asr.py --sine 3

    # 指定语言 / 文本规整
    python -u tests/test_sensevoice_asr.py --file test.wav --language zh --itn

    # 同步进行 CAM++ 说话人识别（需先准备说话人注册文件 speaker.txt，
    # 每行: 名字 wav路径，同一名字可多行取平均）
    python -u tests/test_sensevoice_asr.py --file test.wav --speaker-file speaker.txt

    # 调节说话人识别阈值（默认 0.6，越高越严格）
    python -u tests/test_sensevoice_asr.py --record --speaker-file speaker.txt \
        --speaker-threshold 0.7

依赖:
    pip install sherpa-onnx sounddevice scipy numpy webrtcvad

CAM++ 说话人识别模型（自动下载，~28MB）:
    https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models
"""

import argparse
import logging
import sys
import tarfile
import time
import urllib.request
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Tuple

from voice_agent.vad import has_confirmed_run, init_vad, is_speech_frame

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sensevoice_asr_test")

SAMPLE_RATE = 16000
CHUNK_SIZE = 6400  # bytes, ~0.2s at 16kHz 16-bit

# VAD 帧参数（webrtcvad 要求帧长 10/20/30ms）
VAD_FRAME_MS = 30
VAD_FRAME_SIZE = int(SAMPLE_RATE * VAD_FRAME_MS / 1000)  # 480 样本

# VAD 缓冲优化：始终滚动保留最近 0.2 秒录音，
# 语音被确认后预填充进缓冲，避免丢失开头语音（首音/爆破音等）
PRE_ROLL_SECONDS = 0.2

MODEL_NAME = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "sherpa-asr"

MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    f"asr-models/{MODEL_NAME}.tar.bz2"
)

# CAM++ 说话人识别模型（sherpa-onnx 官方发布，CPU 推理，无需 torch）
SPEAKER_MODEL_NAME = "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx"
SPEAKER_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "sherpa-spk"
SPEAKER_MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    f"speaker-recongition-models/{SPEAKER_MODEL_NAME}"
)

# 说话人分离（diarization）pyannote segmentation 模型
SEG_MODEL_DIRNAME = "sherpa-onnx-pyannote-segmentation-3-0"
SEG_MODEL_FILENAME = "model.onnx"
SEG_MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
)


# ---------------------------------------------------------------------------
# 模型下载
# ---------------------------------------------------------------------------

def download_model(model_dir: Path = None) -> Path:
    """下载并解压 SenseVoice-Small int8 模型，返回模型目录路径。"""
    if model_dir is None:
        model_dir = MODEL_DIR
    model_path = model_dir / MODEL_NAME
    if model_path.exists():
        logger.info("模型已存在: %s", model_path)
        return model_path

    logger.info("📥 下载 SenseVoice-Small int8 模型（~140MB）...")
    model_dir.mkdir(parents=True, exist_ok=True)
    tarball = model_dir / "model.tar.bz2"
    try:
        urllib.request.urlretrieve(MODEL_URL, tarball)
    except Exception as e:
        tarball.unlink(missing_ok=True)
        logger.error("自动下载失败: %s", e)
        logger.error(
            "请手动下载: %s\n解压后放到: %s", MODEL_URL, model_path
        )
        sys.exit(1)
    logger.info("   解压中...")
    with tarfile.open(tarball, "r:bz2") as tar:
        tar.extractall(path=model_dir)
    tarball.unlink()
    logger.info("   ✅ %s", model_path)
    return model_path


def find_model_files(model_path: Path):
    """查找 SenseVoice 模型文件。"""
    model = model_path / "model.int8.onnx"
    tokens = model_path / "tokens.txt"
    for f in [model, tokens]:
        if not f.exists():
            raise FileNotFoundError(f"模型文件缺失: {f}")
    return model, tokens


def download_speaker_model(model_dir: Path = None) -> Path:
    """下载 CAM++ 说话人识别模型（~28MB），返回 onnx 文件路径。"""
    if model_dir is None:
        model_dir = SPEAKER_MODEL_DIR
    model_path = model_dir / SPEAKER_MODEL_NAME
    if model_path.exists():
        logger.info("说话人模型已存在: %s", model_path)
        return model_path

    logger.info("📥 下载 CAM++ 说话人识别模型（~28MB）...")
    model_dir.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(SPEAKER_MODEL_URL, model_path)
    except Exception as e:
        model_path.unlink(missing_ok=True)
        logger.error("自动下载失败: %s", e)
        logger.error(
            "请手动下载: %s\n放到: %s", SPEAKER_MODEL_URL, model_path
        )
        sys.exit(1)
    logger.info("   ✅ %s", model_path)
    return model_path


def download_segmentation_model(model_dir: Path = None) -> Path:
    """下载 pyannote 说话人分离 segmentation 模型，返回 onnx 文件路径。"""
    if model_dir is None:
        model_dir = SPEAKER_MODEL_DIR
    model_path = model_dir / SEG_MODEL_DIRNAME / SEG_MODEL_FILENAME
    if model_path.exists():
        logger.info("说话人分离模型已存在: %s", model_path)
        return model_path

    logger.info("📥 下载 pyannote 说话人分离模型（~7MB）...")
    model_dir.mkdir(parents=True, exist_ok=True)
    tarball = model_dir / "seg-model.tar.bz2"
    try:
        urllib.request.urlretrieve(SEG_MODEL_URL, tarball)
        import tarfile
        with tarfile.open(tarball, "r:bz2") as tar:
            tar.extractall(path=model_dir)
    except Exception as e:
        tarball.unlink(missing_ok=True)
        logger.error("自动下载失败: %s", e)
        logger.error(
            "请手动下载: %s\n解压后放到: %s", SEG_MODEL_URL, model_path
        )
        sys.exit(1)
    finally:
        tarball.unlink(missing_ok=True)
    logger.info("   ✅ %s", model_path)
    return model_path


# ---------------------------------------------------------------------------
# 音频工具
# ---------------------------------------------------------------------------

def read_wave(wave_path: str) -> Tuple[np.ndarray, int]:
    """
    读取单声道 16-bit WAV 文件，返回 float32 样本（归一化到 [-1, 1]）和采样率。
    """
    import wave as _wave

    with _wave.open(wave_path) as f:
        assert f.getnchannels() == 1, f"仅支持单声道，当前 {f.getnchannels()}"
        assert f.getsampwidth() == 2, f"仅支持 16-bit，当前 {f.getsampwidth()}"
        num_samples = f.getnframes()
        raw = f.readframes(num_samples)
        samples_int16 = np.frombuffer(raw, dtype=np.int16)
        samples_float32 = samples_int16.astype(np.float32) / 32768
        return samples_float32, f.getframerate()


def load_wav_to_pcm(path: str) -> Tuple[np.ndarray, int]:
    """从 WAV 加载音频并重采样到 16kHz，返回 float32 样本。"""
    from scipy.io import wavfile
    from scipy import signal

    sr, data = wavfile.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1).astype(data.dtype)
    if data.dtype == np.int16:
        samples = data.astype(np.float32) / 32768
    else:
        samples = data.astype(np.float32)
        if samples.max() > 1.0:
            samples = samples / 32768
    if sr != SAMPLE_RATE:
        n = round(len(samples) * SAMPLE_RATE / sr)
        samples = signal.resample(samples, n).astype(np.float32)
    return samples, SAMPLE_RATE


def generate_test_tone(duration_sec: float = 3.0) -> np.ndarray:
    """生成测试音（440Hz 正弦波 + 前导静音），返回 float32 [-1, 1]。"""
    silence_len = int(SAMPLE_RATE * 0.5)
    silence = np.zeros(silence_len, dtype=np.float32)
    tone_len = int(SAMPLE_RATE * (duration_sec - 0.5))
    t = np.linspace(0, duration_sec - 0.5, tone_len)
    tone = (np.sin(2 * np.pi * 440 * t) * 0.8).astype(np.float32)
    return np.concatenate([silence, tone])


# ---------------------------------------------------------------------------
# ASR 识别
# ---------------------------------------------------------------------------

def create_recognizer(model_path: Path, num_threads: int = 2,
                      language: str = "auto", use_itn: bool = False):
    """
    创建 sherpa-onnx SenseVoice-Small 离线识别器。

    Args:
        model_path: 模型目录路径。
        num_threads: 推理线程数。
        language: 语言（auto/zh/en/yue/ja/ko）。
        use_itn: 是否启用逆文本规整（数字/日期等转文字）。
    """
    import sherpa_onnx

    model, tokens = find_model_files(model_path)
    logger.info("加载模型:")
    logger.info("  模型:   %s", model)
    logger.info("  词表:   %s", tokens)
    logger.info("  语言:   %s", language)
    logger.info("  ITN:    %s", "启用" if use_itn else "禁用")

    recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=str(model),
        tokens=str(tokens),
        num_threads=num_threads,
        provider="cpu",
        sample_rate=SAMPLE_RATE,
        feature_dim=80,
        decoding_method="greedy_search",
        language=language,
        use_itn=use_itn,
    )
    return recognizer


# ---------------------------------------------------------------------------
# CAM++ 说话人识别（声纹 embedding + 身份比对）
# ---------------------------------------------------------------------------

SPEAKER_THRESHOLD = 0.6   # 余弦相似度阈值，高于此值判定为同一说话人


def create_speaker_extractor(model_path: Path, num_threads: int = 2):
    """
    创建 sherpa-onnx CAM++ 说话人 embedding 提取器。

    返回 SpeakerEmbeddingExtractor，可对音频提取 192 维说话人 embedding。
    """
    import sherpa_onnx

    config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=str(model_path),
        num_threads=num_threads,
        debug=False,
        provider="cpu",
    )
    if not config.validate():
        raise ValueError(f"无效的说话人模型配置: {config}")
    extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
    logger.info("CAM++ 说话人模型加载成功（embedding 维度 %d）", extractor.dim)
    return extractor


def extract_speaker_embedding(extractor, samples: np.ndarray,
                              sample_rate: int = SAMPLE_RATE):
    """
    对一段音频提取 CAM++ 说话人 embedding。

    音频太短（不足模型最短输入）时返回 None。
    """
    stream = extractor.create_stream()
    stream.accept_waveform(sample_rate=sample_rate, waveform=samples)
    stream.input_finished()
    if not extractor.is_ready(stream):
        return None
    embedding = extractor.compute(stream)
    return np.array(embedding)


def load_speaker_file(speaker_file: Path) -> Dict[str, List[str]]:
    """
    解析说话人注册文件。

    每行两列：说话人名字 + wav 路径（同一名字可有多行，注册时取平均）。
    空行和 # 注释行被忽略。
    """
    ans = defaultdict(list)
    with open(speaker_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) != 2:
                logger.warning("跳过无效行: %s", line)
                continue
            name, wav_path = fields
            ans[name].append(wav_path)
    return dict(ans)


def build_speaker_manager(extractor, speaker_file: Path):
    """
    从注册文件构建 SpeakerEmbeddingManager。

    每个说话人用其全部 wav 的平均 embedding 注册；未提供注册文件或
    文件为空时返回 None（此时只提取 embedding，不输出身份）。
    """
    import sherpa_onnx

    speakers = load_speaker_file(speaker_file)
    if not speakers:
        logger.info("未提供说话人注册文件（--speaker-file），跳过身份比对")
        return None

    manager = sherpa_onnx.SpeakerEmbeddingManager(extractor.dim)
    for name, wav_paths in speakers.items():
        emb_sum = None
        n_ok = 0
        for wav_path in wav_paths:
            try:
                samples, sr = load_wav_to_pcm(wav_path)
            except Exception as e:
                logger.warning("读取 %s 失败: %s", wav_path, e)
                continue
            emb = extract_speaker_embedding(extractor, samples, sr)
            if emb is None:
                logger.warning("音频过短，无法注册: %s", wav_path)
                continue
            emb_sum = emb if emb_sum is None else emb_sum + emb
            n_ok += 1
        if n_ok == 0:
            logger.warning("说话人 %s 没有可用音频，跳过注册", name)
            continue
        emb_avg = emb_sum / n_ok
        if not manager.add(name, emb_avg):
            logger.warning("注册说话人 %s 失败", name)
        else:
            logger.info("已注册说话人: %s（%d 段音频）", name, n_ok)
    return manager


# ---------------------------------------------------------------------------
# 说话人分离（OfflineSpeakerDiarization，多人分段）
# ---------------------------------------------------------------------------

def create_diarizer(segmentation_model: Path, embedding_model: Path,
                    num_threads: int = 2, min_duration_on: float = 0.3,
                    min_duration_off: float = 0.5):
    """
    创建 sherpa-onnx 说话人分离器。

    需要两个模型：
      - segmentation_model: pyannote segmentation（划分语音/非语音、说话人变化点）
      - embedding_model:    CAM++ 等说话人 embedding 模型（区分不同说话人）

    返回 OfflineSpeakerDiarization，可用 process(samples) 对整段音频分离出
    各说话人的起止时间。
    """
    import sherpa_onnx

    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=str(segmentation_model),
            ),
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(embedding_model),
            num_threads=num_threads,
            provider="cpu",
        ),
        clustering=sherpa_onnx.FastClusteringConfig(),
        min_duration_on=min_duration_on,
        min_duration_off=min_duration_off,
    )
    if not config.validate():
        raise ValueError(f"无效的说话人分离配置: {config}")
    diarizer = sherpa_onnx.OfflineSpeakerDiarization(config)
    logger.info("说话人分离器加载成功（segmentation + CAM++ embedding）")
    return diarizer


def run_diarization(diarizer, samples: np.ndarray, sample_rate: int,
                    label: str = "", recognizer=None,
                    speaker_extractor=None, speaker_manager=None,
                    speaker_threshold: float = SPEAKER_THRESHOLD):
    """
    对一段音频做说话人分离，并把 ASR 结果的 timestamps/tokens 与说话人段
    对齐后按时间顺序输出（说话人变化时换行）。

    参数:
      diarizer: OfflineSpeakerDiarization 实例。
      recognizer: SenseVoice 离线识别器（可选）。提供时输出与说话人对齐的
                  token 文本；不提供时仅打印各说话人起止时间段。
      speaker_extractor / speaker_manager: 提供时把分离出的 "speaker 0/1/..."
                  映射为注册姓名。

    输出示例（说话人变化换行）:
        [测试人] 今天的天气晴有很多云，
        [测试人] 还下雨了，哎呦。
    """
    import numpy as np

    if label:
        print(f"\n📂 {label}")
    print(f"   音频长度: {len(samples) / sample_rate:.1f}s ({len(samples)} 样本)")

    # 重采样到 16kHz（分离器要求 16k 单声道）
    if sample_rate != SAMPLE_RATE:
        from scipy import signal
        n = round(len(samples) * SAMPLE_RATE / sample_rate)
        samples = signal.resample(samples, n).astype(np.float32)

    print("\n🔊 说话人分离中...")
    t0 = time.time()
    result = diarizer.process(samples)
    elapsed = time.time() - t0
    # process 返回 OfflineSpeakerDiarizationResult，用 sort_by_start_time()
    # 取按时间排序的 segments 列表
    segments = result.sort_by_start_time()

    duration = len(samples) / SAMPLE_RATE
    rtf = elapsed / duration if duration > 0 else 0

    print(f"\n{'='*50}")
    print(f"⏱  耗时: {elapsed:.2f}s")
    print(f"📏 音频时长: {duration:.2f}s")
    print(f"⚡ RTF: {elapsed:.2f}/{duration:.2f} = {rtf:.3f}")
    print(f"🎙️ 说话人段数: {len(segments)}")
    print(f"{'='*50}")

    # 对每个说话人段提取 embedding 与注册声纹比对，得到姓名
    seg_names = []  # 与 segments 对齐的显示名
    for i, seg in enumerate(segments):
        start = int(seg.start * SAMPLE_RATE)
        end = int(seg.end * SAMPLE_RATE)
        seg_samples = samples[start:end]
        name = f"speaker {seg.speaker}"
        if speaker_extractor is not None and speaker_manager is not None \
                and len(seg_samples) > 0:
            emb = extract_speaker_embedding(speaker_extractor, seg_samples)
            if emb is not None:
                hit = speaker_manager.search(emb, threshold=speaker_threshold)
                if hit:
                    name = f"{hit}（{seg.speaker}）"
        seg_names.append(name)
        print(f"  🗣️ 说话人: {name}, 开始: {seg.start:.2f}s, 结束: {seg.end:.2f}s")

    # 若提供了 recognizer：做 ASR，把 timestamps/tokens 与说话人段对齐输出
    if recognizer is not None:
        _print_aligned_transcript(recognizer, samples, segments, seg_names)

    print(f"{'='*50}")
    return segments


def _print_aligned_transcript(recognizer, samples: np.ndarray,
                              segments, seg_names: List[str]):
    """
    对音频做 SenseVoice ASR，把每个 token 按时间戳归属到说话人段，
    按时间顺序输出，说话人变化时换行。
    """
    stream = recognizer.create_stream()
    stream.accept_waveform(SAMPLE_RATE, samples)
    t0 = time.time()
    recognizer.decode_stream(stream)
    asr_elapsed = time.time() - t0
    res = stream.result
    timestamps = list(res.timestamps)
    tokens = list(res.tokens)

    def seg_of(ts: float) -> int:
        """返回时间戳 ts 所属 segment 下标；不在任何段内返回 -1。"""
        for i, seg in enumerate(segments):
            if seg.start <= ts < seg.end:
                return i
        return -1

    print(f"\n📝 识别文本（与说话人对齐，ASR {asr_elapsed:.2f}s）:")
    lines = []
    cur_seg = -1
    for ts, tok in zip(timestamps, tokens):
        idx = seg_of(ts)
        if idx == -1:
            # 时间戳落在说话人段之外（如段间静音处的标点），沿用上一说话人
            idx = cur_seg if cur_seg != -1 else 0
        if idx != cur_seg:
            name = seg_names[idx] if 0 <= idx < len(seg_names) else "未知"
            if 0 <= idx < len(segments):
                seg = segments[idx]
                time_tag = f"[时间:{seg.start:.2f}s-{seg.end:.2f}s]"
            else:
                time_tag = ""
            cur_seg = idx
            lines.append(f"{time_tag}[说话人:{name}] {tok}")
        else:
            lines[-1] += tok
    for line in lines:
        print(f"  {line}")


# 防幻觉参数
MIN_AUDIO_DURATION = 0.5      # 最短识别音频（秒）
MIN_SIGNAL_RMS = 0.004        # 最低 RMS 能量（低于视为纯静音）


def _is_hallucination(text: str) -> bool:
    """判断是否为 SenseVoice 幻觉输出（填充词/无意义短词）。"""
    cleaned = text.strip().rstrip("。,.?!?! ")
    if not cleaned:
        return True
    # 纯 ASCII 短词（< 3 字符）
    if cleaned.isascii() and len(cleaned) <= 3:
        return True
    # 单字符日文/中文
    if len(cleaned) <= 1:
        return True
    return False


def _signal_energy(samples: np.ndarray) -> float:
    """计算音频 RMS 能量。"""
    if len(samples) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples))))


def recognize_audio(recognizer, samples: np.ndarray,
                    sample_rate: int = SAMPLE_RATE,
                    speaker_extractor=None, speaker_manager=None,
                    speaker_threshold: float = SPEAKER_THRESHOLD,
                    ) -> Tuple[str, float, str]:
    """
    识别一段音频，返回 (识别文本, 耗时秒, 说话人身份)。

    SenseVoice 输出自带标点，结果中可能包含 <|zh|><|NEUTRAL|> 等标记，
    这里会清理掉这些标记只保留纯文本。

    同步进行 CAM++ 说话人识别：若提供 speaker_extractor / speaker_manager，
    则对同一段音频提取 192 维声纹 embedding，并与已注册说话人比对，
    返回命中身份；未命中或未提供注册表时返回 "未知"。
    """
    import re

    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate, samples)

    t0 = time.time()
    recognizer.decode_stream(stream)
    elapsed = time.time() - t0

    # stream.result 返回 OfflineRecognitionResult 对象
    print(stream.result)
    # {"lang": "<|zh|>", "emotion": "<|NEUTRAL|>", "event": "<|Speech|>",
    # "text": "今天的天气晴有很多云，还下雨了，哎呦。",
    # "timestamps": [0.24, 0.36, 0.54, 0.72, 0.90, 1.20, 1.56, 1.74, 1.92, 2.10, 2.34, 2.40, 2.64, 2.82, 3.00, 3.18, 3.66, 3.84, 4.26],
    # "durations": [],
    # "tokens":["今", "天", "的", "天", "气", "晴", "有", "很", "多", "云", "，", "还", "下", "雨", "了", "，", "哎", "呦", "。"],
    # "ys_log_probs": [], "words": []}
    text = stream.result.text.strip()
    # 清理 SenseVoice 输出的标记（如 <|zh|><|NEUTRAL|><|Speech|> 等）

    # 同步进行 CAM++ 说话人识别（与 ASR 共用同一段音频）
    speaker = "未知"
    if speaker_extractor is not None and speaker_manager is not None:
        embedding = extract_speaker_embedding(speaker_extractor, samples, sample_rate)
        if embedding is not None:
            name = speaker_manager.search(embedding, threshold=speaker_threshold)
            if name:
                speaker = name
            else:
                speaker = "未知"

    return text, elapsed, speaker


def recognize_microphone_vad(recognizer, vad_mode: int = 3,
                              silence_timeout_ms: int = 1000,
                              speech_confirm_frames: int = 5,
                              max_duration_ms: int = 15000,
                              pre_roll_seconds: float = PRE_ROLL_SECONDS,
                              speaker_extractor=None, speaker_manager=None,
                              speaker_threshold: float = SPEAKER_THRESHOLD):
    """
    麦克风 VAD 录音识别（SenseVoice-Small 离线模式）。

    使用 WebRTC VAD 检测人声，并维护一个最近 pre_roll_seconds 秒的滚动缓冲：
    连续 speech_confirm_frames 帧（默认 5 帧 ≈ 150ms）判定为人声 → 用滚动缓冲
    预填充（避免丢失开头语音）→ 继续录音缓冲 → 最近 silence_timeout_ms 内不再
    出现连续确认语音帧 → 对缓冲音频做离线识别（同步进行 CAM++ 说话人识别）。

    结束判定基于最近 silence_timeout_frames 帧的滑动窗口：只有窗口内存在连续
    speech_confirm_frames 个 is_speech 帧才视为对话进行中；偶然的孤立噪音帧
    不足 confirm_frames 帧连续，不会重置静音计时，避免对话窗口被噪音无限延长、
    增加 ASR 延迟。

    行为:
      🔇 静音等待 → 🎤 连续 5 帧人声（含前 0.2s 前缀）→ 录音中
      → 🔇 最近窗口无连续人声 → 立即识别并打印带标点结果（含说话人身份）
    """
    import sounddevice as sd

    vad = init_vad(vad_mode)
    devices = sd.query_devices()
    if len(devices) == 0:
        logger.error("未检测到麦克风")
        return

    default_input = sd.default.device[0]
    logger.info("使用默认麦克风: %s", devices[default_input]["name"])

    mic_sample_rate = SAMPLE_RATE
    samples_per_read = VAD_FRAME_SIZE  # 30ms

    silence_timeout_frames = silence_timeout_ms // VAD_FRAME_MS
    max_duration_frames = max_duration_ms // VAD_FRAME_MS
    pre_roll_frames = max(1, round(pre_roll_seconds * 1000 / VAD_FRAME_MS))
    speech_confirm = 0
    in_speech = False
    speech_frames = 0

    buf: List[np.ndarray] = []
    # 滚动缓冲：始终保留最近 pre_roll_seconds 秒录音（maxlen 自动丢弃最旧帧），
    # 语音被确认后将其预填充进 buf，避免丢失开头语音
    recent_buf: deque = deque(maxlen=pre_roll_frames)
    # 最近 silence_timeout_frames 帧的 is_speech 标记滑动窗口（maxlen 自动丢弃
    # 最旧帧）：窗口内存在连续 speech_confirm_frames 个 is_speech 帧才视为对话
    # 进行中；孤立噪音帧不会重置静音计时，避免对话窗口被噪音无限延长
    recent_flags: deque = deque(maxlen=max(silence_timeout_frames, speech_confirm_frames))
    round_results: List[str] = []

    print(f"\n🎤 VAD 录音中（静音 {silence_timeout_ms/1000:.1f}s 自动结束本轮）")
    print("   SenseVoice-Small 离线识别，自带标点")
    print("   Ctrl+C 退出\n")

    try:
        with sd.InputStream(channels=1, dtype="int16",
                            samplerate=mic_sample_rate) as s:
            while True:
                samples, _ = s.read(samples_per_read)
                samples = samples.reshape(-1)
                pcm_bytes = samples.tobytes()
                audio_float = samples.astype(np.float32) / 32768.0

                # 每帧都更新最近 pre_roll_seconds 秒滚动缓冲
                recent_buf.append(audio_float)

                is_speech = is_speech_frame(vad, pcm_bytes)
                recent_flags.append(is_speech)

                if is_speech:
                    speech_confirm += 1

                    if not in_speech and speech_confirm >= speech_confirm_frames:
                        # 语音开始 → 用最近 pre_roll_seconds 秒滚动缓冲预填充，
                        # 保留确认前的开头语音（首音/爆破音等）
                        in_speech = True
                        buf = list(recent_buf)  # 已含当前确认帧，不再重复追加
                        speech_frames = len(buf)
                        print("\n🎤", end="", flush=True)
                    elif in_speech:
                        buf.append(audio_float)
                        speech_frames += 1
                else:
                    speech_confirm = 0
                    if in_speech:
                        # 静音期间继续录（保持尾部）
                        buf.append(audio_float)

                # 对话结束判定（录音中）：最近 silence_timeout_frames 帧内不存在
                # 连续 speech_confirm_frames 个 is_speech 帧 → 对话已结束，立即
                # 识别；或超出最长录音保护。孤立噪音帧不构成确认语音，不会延长
                # 对话窗口（旧机制中任一 is_speech 帧都会重置静音计数）
                if in_speech and (speech_frames >= max_duration_frames
                                  or not has_confirmed_run(recent_flags, speech_confirm_frames)):
                    # 结束本轮 → 处理防幻觉 → 离线识别
                    audio = np.concatenate(buf).astype(np.float32)
                    duration = len(audio) / SAMPLE_RATE
                    rms = _signal_energy(audio)

                    # 防幻觉：音频太短 / 能量太低 直接丢弃
                    if duration < MIN_AUDIO_DURATION:
                        print("\r  🔇 音频过短，忽略", end="", flush=True)
                    elif rms < MIN_SIGNAL_RMS:
                        print("\r  🔇 静音，忽略", end="", flush=True)
                    else:
                        t0 = time.time()
                        text, elapsed, speaker = recognize_audio(
                            recognizer, audio,
                            speaker_extractor=speaker_extractor,
                            speaker_manager=speaker_manager,
                            speaker_threshold=speaker_threshold,
                        )
                        if text and not _is_hallucination(text):
                            round_results.append(text)
                            tag = f" [{speaker}]" if speaker != "未知" else ""
                            print(f"\r  ✅{tag} {text}  ({elapsed:.2f}s)")
                        else:
                            print("\r  🔇 无有效语音", end="", flush=True)
                    in_speech = False
                    speech_confirm = 0
                    speech_frames = 0
                    buf = []
                    recent_flags.clear()
                    print("\n🔊 等待下轮说话...", end="", flush=True)

    except KeyboardInterrupt:
        # 识别最后一段未结束的录音
        if in_speech and buf:
            audio = np.concatenate(buf).astype(np.float32)
            duration = len(audio) / SAMPLE_RATE
            rms = _signal_energy(audio)

            if duration >= MIN_AUDIO_DURATION and rms >= MIN_SIGNAL_RMS:
                text, elapsed, speaker = recognize_audio(
                    recognizer, audio,
                    speaker_extractor=speaker_extractor,
                    speaker_manager=speaker_manager,
                    speaker_threshold=speaker_threshold,
                )
                if text and not _is_hallucination(text):
                    round_results.append(text)
                    tag = f" [{speaker}]" if speaker != "未知" else ""
                    print(f"\n  ✅{tag} {text}  ({elapsed:.2f}s)")
                else:
                    print("\n  🔇 无有效语音")
            else:
                print("\n  🔇 音频过短或静音，忽略")

        print(f"\n\n📊 本轮结果: {len(round_results)} 次")
        for i, t in enumerate(round_results, 1):
            print(f"   {i}. {t}")
        print("⏹  已停止")


def recognize_microphone(recognizer, speaker_extractor=None,
                         speaker_manager=None,
                         speaker_threshold: float = SPEAKER_THRESHOLD):
    """
    麦克风持续录音，Ctrl+C 停止后一次性识别（无 VAD）。

    识别时同步进行 CAM++ 说话人识别（如提供 extractor/manager）。
    """
    import sounddevice as sd

    devices = sd.query_devices()
    if len(devices) == 0:
        logger.error("未检测到麦克风")
        return

    default_input = sd.default.device[0]
    logger.info("使用默认麦克风: %s", devices[default_input]["name"])

    mic_sample_rate = SAMPLE_RATE
    samples_per_read = int(0.1 * mic_sample_rate)   # 100ms

    buffer: List[np.ndarray] = []
    record_seconds = [0.0]

    print(f"\n🎤 录音中...（Ctrl+C 停止并识别）\n", flush=True)

    try:
        with sd.InputStream(channels=1, dtype="float32",
                            samplerate=mic_sample_rate) as s:
            while True:
                samples, _ = s.read(samples_per_read)
                samples = samples.reshape(-1)
                buffer.append(samples)
                record_seconds[0] += samples_per_read / mic_sample_rate
                print(f"\r  ⏺️ {record_seconds[0]:.1f}s", end="", flush=True)
    except KeyboardInterrupt:
        print(f"\n\n⏹  录音结束（{record_seconds[0]:.1f}s）")

    if not buffer:
        print("  无音频数据")
        return

    audio = np.concatenate(buffer).astype(np.float32)
    print("🔊 识别中...")
    text, elapsed, speaker = recognize_audio(
        recognizer, audio,
        speaker_extractor=speaker_extractor,
        speaker_manager=speaker_manager,
        speaker_threshold=speaker_threshold,
    )
    if text:
        tag = f" [{speaker}]" if speaker != "未知" else ""
        print(f"\n  ✅{tag} {text}  ({elapsed:.2f}s)")
    else:
        print("\n  🔇 无识别结果")


# ---------------------------------------------------------------------------
# 文件识别
# ---------------------------------------------------------------------------

def run_file_test(recognizer, samples: np.ndarray, sample_rate: int,
                  label: str = "", speaker_extractor=None,
                  speaker_manager=None,
                  speaker_threshold: float = SPEAKER_THRESHOLD):
    """运行文件识别测试（同步进行 CAM++ 说话人识别）。"""
    if label:
        print(f"\n📂 {label}")
    print(f"   音频长度: {len(samples) / sample_rate:.1f}s ({len(samples)} 样本)")

    # 重采样到 16kHz
    if sample_rate != SAMPLE_RATE:
        from scipy import signal
        n = round(len(samples) * SAMPLE_RATE / sample_rate)
        samples = signal.resample(samples, n).astype(np.float32)

    print("\n🔊 识别中...")
    text, elapsed, speaker = recognize_audio(
        recognizer, samples,
        speaker_extractor=speaker_extractor,
        speaker_manager=speaker_manager,
        speaker_threshold=speaker_threshold,
    )

    duration = len(samples) / SAMPLE_RATE
    rtf = elapsed / duration if duration > 0 else 0

    print(f"\n{'='*50}")
    print(f"⏱  耗时: {elapsed:.2f}s")
    print(f"📏 音频时长: {duration:.2f}s")
    print(f"⚡ RTF: {elapsed:.2f}/{duration:.2f} = {rtf:.3f}")
    if speaker != "未知":
        print(f"🗣️ 说话人: {speaker}")
    print(f"\n✅ 识别结果（带标点）: {text or '<空>'}")
    print(f"{'='*50}")

    return text


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="sherpa-onnx SenseVoice-Small 语音识别测试（自带标点）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument("--file", type=str, metavar="PATH", help="WAV 文件路径")
    g.add_argument("--sine", type=float, metavar="SEC", help="生成测试音（秒）")
    g.add_argument("--record", action="store_true", help="麦克风录音识别")
    g.add_argument("--diarize", type=str, metavar="PATH",
                   help="WAV 文件路径（说话人分离模式：按说话人分段输出起止时间）")

    # 模型参数
    p.add_argument("--model-dir", type=str, default=MODEL_DIR,
                   help="模型目录")
    p.add_argument("--num-threads", type=int, default=2,
                   help="推理线程数")
    p.add_argument("--language", type=str, default="zh",
                   choices=["auto", "zh", "en", "yue", "ja", "ko"],
                   help="语言（默认 %(default)s）")
    p.add_argument("--itn", action="store_true", dest="itn",
                   help="启用逆文本规整（默认启用，输出带标点）")
    p.set_defaults(itn=True)

    # VAD 参数（仅 --record 模式有效）
    p.add_argument("--vad-mode", type=int, default=1, choices=[0, 1, 2, 3],
                   help="VAD 灵敏度（0-3，3 最敏感，默认 %(default)s）")
    p.add_argument("--vad-timeout", type=int, default=1000,
                   help="VAD 静音超时毫秒（默认 %(default)sms）")
    p.add_argument("--vad-confirm", type=int, default=5,
                   help="VAD 语音确认连续帧数（默认 %(default)s，约 150ms）")
    p.add_argument("--max-duration", type=int, default=15,
                   help="单轮最长录音秒数（默认 %(default)s）")
    p.add_argument("--pre-roll", type=float, default=PRE_ROLL_SECONDS,
                   help="语音开始前保留的录音秒数，避免丢失开头（默认 %(default)s）")

    # VAD 无参数模式（仅录音不自动结束）
    p.add_argument("--no-vad", action="store_true",
                   help="--record 模式下禁用 VAD（Ctrl+C 停止后识别）")

    # CAM++ 说话人识别参数
    p.add_argument("--speaker-file", type=str, default=None,
                   metavar="PATH",
                   help="说话人注册文件（每行: 名字 wav路径，同一名字可多行取平均）；"
                        "不指定则不进行身份比对")
    p.add_argument("--speaker-threshold", type=float, default=SPEAKER_THRESHOLD,
                   help="说话人识别余弦相似度阈值（默认 %(default)s）")
    p.add_argument("--speaker-model-dir", type=str, default=SPEAKER_MODEL_DIR,
                   help="CAM++ 说话人模型目录")

    args = p.parse_args()

    # --- 确定模型路径 ---
    model_path = Path(args.model_dir) / MODEL_NAME
    if not model_path.exists():
        logger.error("模型目录不存在: %s", model_path)
        logger.info("尝试自动下载模型...")
        model_path = download_model(Path(args.model_dir))

    recognizer = create_recognizer(
        model_path,
        num_threads=args.num_threads,
        language=args.language,
        use_itn=args.itn,
    )

    # --- 装配 CAM++ 说话人识别（可选） ---
    speaker_extractor = speaker_manager = None
    if args.speaker_file:
        speaker_model = download_speaker_model(Path(args.speaker_model_dir))
        speaker_extractor = create_speaker_extractor(
            speaker_model, num_threads=args.num_threads)
        speaker_manager = build_speaker_manager(
            speaker_extractor, Path(args.speaker_file))

    # --- 装配说话人分离器（--diarize 模式） ---
    diarizer = None
    if args.diarize:
        seg_model = download_segmentation_model(Path(args.speaker_model_dir))
        spk_model = download_speaker_model(Path(args.speaker_model_dir))
        diarizer = create_diarizer(seg_model, spk_model,
                                   num_threads=args.num_threads)

    # --- 确定音频来源 ---
    if args.file:
        if args.file.endswith(".wav"):
            samples, sr = read_wave(args.file)
            logger.info("📂 加载 WAV: %s", args.file)
            logger.info("   采样率: %d Hz, 长度: %.1f s", sr, len(samples) / sr)
        else:
            samples, sr = load_wav_to_pcm(args.file)
            logger.info("📂 加载 PCM: %s", args.file)
            logger.info("   长度: %d 样本 (%.1f s @ %d Hz)", len(samples),
                        len(samples) / SAMPLE_RATE, SAMPLE_RATE)

        run_file_test(recognizer, samples, sr, label=args.file,
                      speaker_extractor=speaker_extractor,
                      speaker_manager=speaker_manager,
                      speaker_threshold=args.speaker_threshold)

    elif args.diarize:
        if args.diarize.endswith(".wav"):
            samples, sr = read_wave(args.diarize)
            logger.info("📂 加载 WAV: %s", args.diarize)
            logger.info("   采样率: %d Hz, 长度: %.1f s", sr, len(samples) / sr)
        else:
            samples, sr = load_wav_to_pcm(args.diarize)
            logger.info("📂 加载 PCM: %s", args.diarize)
            logger.info("   长度: %d 样本 (%.1f s @ %d Hz)", len(samples),
                        len(samples) / SAMPLE_RATE, SAMPLE_RATE)

        run_diarization(diarizer, samples, sr, label=args.diarize,
                        recognizer=recognizer,
                        speaker_extractor=speaker_extractor,
                        speaker_manager=speaker_manager,
                        speaker_threshold=args.speaker_threshold)

    elif args.sine:
        logger.info("🔊 生成测试音 %s 秒", args.sine)
        samples = generate_test_tone(args.sine)
        run_file_test(recognizer, samples, SAMPLE_RATE, label=f"测试音 {args.sine}s",
                      speaker_extractor=speaker_extractor,
                      speaker_manager=speaker_manager,
                      speaker_threshold=args.speaker_threshold)

    elif args.record:
        try:
            import sounddevice  # noqa
            import sherpa_onnx  # noqa
        except ImportError as e:
            logger.error("❌ 缺少依赖: %s\n   pip install sherpa-onnx sounddevice", e)
            sys.exit(1)

        if args.no_vad:
            recognize_microphone(recognizer,
                                 speaker_extractor=speaker_extractor,
                                 speaker_manager=speaker_manager,
                                 speaker_threshold=args.speaker_threshold)
        else:
            recognize_microphone_vad(
                recognizer,
                vad_mode=args.vad_mode,
                silence_timeout_ms=args.vad_timeout,
                speech_confirm_frames=args.vad_confirm,
                max_duration_ms=args.max_duration * 1000,
                pre_roll_seconds=args.pre_roll,
                speaker_extractor=speaker_extractor,
                speaker_manager=speaker_manager,
                speaker_threshold=args.speaker_threshold,
            )

    else:
        # 默认：使用 test.wav
        test_wav = Path(__file__).parent / "test.wav"
        if test_wav.exists():
            logger.info("📂 使用默认测试文件: %s", test_wav)
            samples, sr = read_wave(str(test_wav))
            run_file_test(recognizer, samples, sr, label=str(test_wav),
                          speaker_extractor=speaker_extractor,
                          speaker_manager=speaker_manager,
                          speaker_threshold=args.speaker_threshold)
        else:
            logger.info("🔊 默认生成 3s 测试音（未指定 --file / --sine / --record）")
            logger.info("   提示：也可以使用已有 WAV 文件识别")
            samples = generate_test_tone(3.0)
            run_file_test(recognizer, samples, SAMPLE_RATE, label="测试音 3s",
                          speaker_extractor=speaker_extractor,
                          speaker_manager=speaker_manager,
                          speaker_threshold=args.speaker_threshold)


if __name__ == "__main__":
    main()