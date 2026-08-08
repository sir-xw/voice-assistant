#!/usr/bin/env python3
"""
sherpa-onnx 流式 ASR 识别测试。

使用 sherpa_onnx.OnlineRecognizer 进行本地语音识别（无需腾讯云）。
暂时不采用，测试发现准确率比腾讯云差，且没有断句标点。
模型下载：
https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2

用法:
    # 识别 WAV 文件
    python -u tests/test_sherpa_asr.py --file test.wav

    # 生成测试音并识别
    python -u tests/test_sherpa_asr.py --sine 3

    # 麦克风实时识别
    python -u tests/test_sherpa_asr.py --record                       # 持续识别
    python -u tests/test_sherpa_asr.py --record --vad                 # VAD 自动开始/结束
    python -u tests/test_sherpa_asr.py --record --vad --vad-timeout 1500

    # 指定自定义模型
    python -u tests/test_sherpa_asr.py --file test.wav --model-dir /path/to/model

依赖:
    pip install sherpa-onnx sounddevice  (麦克风模式)
    pip install scipy                     (WAV 加载)
"""

import argparse
import logging
import sys
import tarfile
import time
import urllib.request
from collections import deque
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from vad import has_confirmed_run, init_vad, is_speech_frame

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sherpa_asr_test")

SAMPLE_RATE = 16000
CHUNK_SIZE = 6400  # bytes, ~0.2s at 16kHz 16-bit

# VAD 帧参数（webrtcvad 要求帧长 10/20/30ms）
VAD_FRAME_MS = 30
VAD_FRAME_SIZE = int(SAMPLE_RATE * VAD_FRAME_MS / 1000)  # 480 样本
VAD_FRAME_BYTES = VAD_FRAME_SIZE * 2  # 960 bytes (int16)

MODEL_NAME = "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "sherpa-asr"


# ---------------------------------------------------------------------------
# 模型下载
# ---------------------------------------------------------------------------

def download_model(model_dir: Path = None) -> Path:
    """下载并解压 sherpa-onnx ASR 模型，返回模型目录路径。"""
    if model_dir is None:
        model_dir = MODEL_DIR
    model_path = model_dir / MODEL_NAME
    if model_path.exists():
        logger.info("模型已存在: %s", model_path)
        return model_path

    logger.info("📥 下载 ASR 模型 (~38MB)...")
    model_dir.mkdir(parents=True, exist_ok=True)
    tarball = model_dir / "model.tar.bz2"
    urllib.request.urlretrieve(MODEL_URL, tarball)
    logger.info("   解压中...")
    with tarfile.open(tarball, "r:bz2") as tar:
        tar.extractall(path=model_dir)
    tarball.unlink()
    logger.info("   ✅ %s", model_path)
    return model_path


def find_model_files(model_path: Path):
    """查找模型文件（encoder, decoder, joiner, tokens），优先 int8。"""
    enc = model_path / f"encoder-epoch-99-avg-1.onnx"
    enc_int8 = model_path / f"encoder-epoch-99-avg-1.int8.onnx"
    if enc_int8.exists():
        enc = enc_int8

    dec = model_path / f"decoder-epoch-99-avg-1.onnx"
    joi = model_path / f"joiner-epoch-99-avg-1.onnx"
    joi_int8 = model_path / f"joiner-epoch-99-avg-1.int8.onnx"
    if joi_int8.exists():
        joi = joi_int8

    tok = model_path / "tokens.txt"

    for f in [enc, dec, joi, tok]:
        if not f.exists():
            raise FileNotFoundError(f"模型文件缺失: {f}")

    return enc, dec, joi, tok


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
# VAD 工具（仅用于控制 ASR 生命周期，不过滤音频样本）
# 与 src/voice_frontend.py 共用 src/vad.py：VAD 只决定何时启动/停止 ASR，
# 识别期间所有帧全量送入，不做静音去除。
# ---------------------------------------------------------------------------





# ---------------------------------------------------------------------------
# ASR 识别
# ---------------------------------------------------------------------------

def create_recognizer(model_path: Path, num_threads: int = 2):
    """创建 sherpa-onnx 流式识别器。"""
    import sherpa_onnx

    enc, dec, joi, tok = find_model_files(model_path / MODEL_NAME)
    logger.info("加载模型:")
    logger.info("  编码器: %s", enc)
    logger.info("  解码器: %s", dec)
    logger.info("  联合器: %s", joi)
    logger.info("  词表:   %s", tok)

    recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=str(tok),
        encoder=str(enc),
        decoder=str(dec),
        joiner=str(joi),
        num_threads=num_threads,
        provider="cpu",
        sample_rate=SAMPLE_RATE,
        feature_dim=80,
        decoding_method="greedy_search",
    )
    return recognizer


def recognize_file(recognizer, samples: np.ndarray, sample_rate: int,
                   show_interim: bool = True) -> Tuple[str, float, List[str]]:
    """
    识别一段音频，返回 (最终文本, 耗时秒, 所有中间结果)。

    模拟流式：分块送入以观察中间结果变化。
    """
    stream = recognizer.create_stream()
    chunk_samples = int(0.2 * sample_rate)

    t0 = time.time()
    interims: List[str] = []
    last_text = ""

    pos = 0
    while pos < len(samples):
        chunk = samples[pos:pos + chunk_samples]
        stream.accept_waveform(sample_rate, chunk)
        pos += chunk_samples

        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)

        result = recognizer.get_result(stream)
        if result != last_text:
            interims.append(result)
            if show_interim:
                print(f"  📝 [interim] {result}", flush=True)
            last_text = result

    # 尾部填充（确保尾音被处理）
    tail_paddings = np.zeros(int(0.66 * sample_rate), dtype=np.float32)
    stream.accept_waveform(sample_rate, tail_paddings)
    stream.input_finished()

    while True:
        if recognizer.is_ready(stream):
            recognizer.decode_stream(stream)
        else:
            break

    final_text = recognizer.get_result(stream)
    elapsed = time.time() - t0

    if final_text and final_text != last_text:
        interims.append(final_text)

    return final_text, elapsed, interims


def recognize_file_streaming_callback(recognizer, samples: np.ndarray,
                                       sample_rate: int,
                                       on_interim=None, on_final=None,
                                       on_complete=None):
    """
    流式识别，支持回调（类似 test_tencent_asr.py 的回调风格）。
    """
    stream = recognizer.create_stream()
    chunk_samples = int(0.2 * sample_rate)
    t0 = time.time()
    last_text = ""

    pos = 0
    while pos < len(samples):
        chunk = samples[pos:pos + chunk_samples]
        stream.accept_waveform(sample_rate, chunk)
        pos += chunk_samples

        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)

        result = recognizer.get_result(stream)
        if result != last_text:
            if on_interim:
                on_interim(result, time.time() - t0)
            last_text = result

    tail_paddings = np.zeros(int(0.66 * sample_rate), dtype=np.float32)
    stream.accept_waveform(sample_rate, tail_paddings)
    stream.input_finished()

    while True:
        if recognizer.is_ready(stream):
            recognizer.decode_stream(stream)
        else:
            break

    final_text = recognizer.get_result(stream)
    if on_final:
        on_final(final_text, time.time() - t0)
    if on_complete:
        on_complete(final_text, time.time() - t0)

    return final_text


def recognize_microphone(recognizer):
    """
    麦克风实时识别（无 VAD，持续识别）。

    使用 sounddevice 采集音频，实时送入 sherpa-onnx 解码并打印结果。
    """
    import sounddevice as sd

    devices = sd.query_devices()
    if len(devices) == 0:
        logger.error("未检测到麦克风")
        return

    default_input = sd.default.device[0]
    logger.info("使用默认麦克风: %s", devices[default_input]["name"])

    mic_sample_rate = 48000
    samples_per_read = int(0.1 * mic_sample_rate)

    stream = recognizer.create_stream()
    last_result = ""

    print(f"\n🎤 监听中（Ctrl+C 退出）...\n", flush=True)

    try:
        with sd.InputStream(channels=1, dtype="float32",
                            samplerate=mic_sample_rate) as s:
            while True:
                samples, _ = s.read(samples_per_read)
                samples = samples.reshape(-1)
                stream.accept_waveform(mic_sample_rate, samples)
                while recognizer.is_ready(stream):
                    recognizer.decode_stream(stream)
                result = recognizer.get_result(stream)
                if result != last_result:
                    print(f"\r  📝 {result}", end="", flush=True)
                    last_result = result
    except KeyboardInterrupt:
        print(f"\n\n  ✅ 最终: {last_result}")
        print("⏹  已停止")


def recognize_microphone_vad(recognizer, vad_mode: int = 3,
                              silence_timeout_ms: int = 1000,
                              speech_confirm_frames: int = 3):
    """
    麦克风 VAD 录音识别。

    使用 WebRTC VAD 检测人声，仅在说话时送入 sherpa-onnx 识别。
    最近 silence_timeout_ms 内不再出现连续确认语音帧后自动停止本轮 ASR，
    等待下次说话。结束判定基于最近 silence_timeout_frames 帧的滑动窗口：
    只有窗口内存在连续 speech_confirm_frames 个 is_speech 帧才视为对话进行中，
    偶然的孤立噪音帧不足 confirm_frames 帧连续，不会重置静音计时，避免对话
    窗口被噪音无限延长、增加 ASR 延迟。

    行为（类似 test_tencent_asr.py 的 VAD 录音）:
      🔇 静音等待 → 🎤 检测到人声 → 实时识别 → 🔇 最近窗口无连续人声 → 打印本轮结果
    """
    import sounddevice as sd

    vad = init_vad(vad_mode)
    devices = sd.query_devices()
    if len(devices) == 0:
        logger.error("未检测到麦克风")
        return

    default_input = sd.default.device[0]
    logger.info("使用默认麦克风: %s", devices[default_input]["name"])

    # 麦克风用 16kHz 避免额外重采样
    mic_sample_rate = SAMPLE_RATE
    samples_per_read = VAD_FRAME_SIZE  # 30ms

    silence_timeout_frames = silence_timeout_ms // VAD_FRAME_MS
    speech_confirm = 0
    in_speech = False

    # 最近 silence_timeout_frames 帧的 is_speech 标记滑动窗口（maxlen 自动丢弃
    # 最旧帧）：窗口内存在连续 speech_confirm_frames 个 is_speech 帧才视为对话
    # 进行中；孤立噪音帧不会重置静音计时，避免对话窗口被噪音无限延长
    recent_flags: deque = deque(maxlen=max(silence_timeout_frames, speech_confirm_frames))

    stream = recognizer.create_stream()
    last_result = ""
    round_results: List[str] = []

    print(f"\n🎤 VAD 录音中（静音 {silence_timeout_ms/1000:.1f}s 自动结束本轮）")
    print("   Ctrl+C 退出\n")

    try:
        with sd.InputStream(channels=1, dtype="int16",
                            samplerate=mic_sample_rate) as s:
            while True:
                samples, _ = s.read(samples_per_read)
                samples = samples.reshape(-1)
                pcm_bytes = samples.tobytes()

                is_speech = is_speech_frame(vad, pcm_bytes)
                recent_flags.append(is_speech)

                if is_speech:
                    speech_confirm += 1

                    if not in_speech and speech_confirm >= speech_confirm_frames:
                        # 语音开始 → 创建新 stream，开始识别
                        in_speech = True
                        stream = recognizer.create_stream()
                        last_result = ""
                        print("\n🎤", end="", flush=True)

                    if in_speech:
                        # 送入识别的音频需要 float32 [-1, 1]
                        audio_float = samples.astype(np.float32) / 32768.0
                        stream.accept_waveform(mic_sample_rate, audio_float)
                        while recognizer.is_ready(stream):
                            recognizer.decode_stream(stream)
                        result = recognizer.get_result(stream)
                        if result and result != last_result:
                            print(f"\r  📝 {result}", end="", flush=True)
                            last_result = result
                else:
                    speech_confirm = 0
                    if in_speech:
                        # 静音期间继续送帧（保持 ASR 尾部处理）
                        audio_float = samples.astype(np.float32) / 32768.0
                        stream.accept_waveform(mic_sample_rate, audio_float)
                        while recognizer.is_ready(stream):
                            recognizer.decode_stream(stream)

                # 对话结束判定（录音中）：最近 silence_timeout_frames 帧内不存在
                # 连续 speech_confirm_frames 个 is_speech 帧 → 对话已结束，立即
                # 结束本轮并取最终结果。孤立噪音帧不构成确认语音，不会延长对话
                # 窗口（旧机制中任一 is_speech 帧都会重置静音计数）
                if in_speech and not has_confirmed_run(recent_flags, speech_confirm_frames):
                    # 结束本轮
                    tail = np.zeros(int(0.5 * SAMPLE_RATE), dtype=np.float32)
                    stream.accept_waveform(SAMPLE_RATE, tail)
                    stream.input_finished()
                    while True:
                        if recognizer.is_ready(stream):
                            recognizer.decode_stream(stream)
                        else:
                            break
                    final = recognizer.get_result(stream)
                    if final:
                        round_results.append(final)
                        print(f"\r  ✅ {final}")
                    else:
                        print("\r  🔇 无识别结果")
                    in_speech = False
                    speech_confirm = 0
                    recent_flags.clear()
                    print("\n🔊 等待下轮说话...", end="", flush=True)

    except KeyboardInterrupt:
        # 打印本轮最终结果
        if in_speech:
            tail = np.zeros(int(0.5 * SAMPLE_RATE), dtype=np.float32)
            stream.accept_waveform(SAMPLE_RATE, tail)
            stream.input_finished()
            while True:
                if recognizer.is_ready(stream):
                    recognizer.decode_stream(stream)
                else:
                    break
            final = recognizer.get_result(stream)
            if final:
                round_results.append(final)
                print(f"\n  ✅ {final}")

        print(f"\n\n📊 本轮结果: {len(round_results)} 次")
        for i, t in enumerate(round_results, 1):
            print(f"   {i}. {t}")
        print("⏹  已停止")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def run_file_test(samples: np.ndarray, sample_rate: int,
                  model_path: Path, label: str = ""):
    """运行文件识别测试（简洁模式）。"""
    import sherpa_onnx  # noqa: F401

    if label:
        print(f"\n📂 {label}")
    print(f"   音频长度: {len(samples) / sample_rate:.1f}s ({len(samples)} 样本)")

    recognizer = create_recognizer(model_path)

    print("\n🔊 识别中...")
    final_text, elapsed, interims = recognize_file(recognizer, samples, sample_rate)

    print(f"\n{'='*50}")
    print(f"⏱  耗时: {elapsed:.2f}s")
    duration = len(samples) / sample_rate
    rtf = elapsed / duration if duration > 0 else 0
    print(f"📏 音频时长: {duration:.2f}s")
    print(f"⚡ RTF: {elapsed:.2f}/{duration:.2f} = {rtf:.3f}")

    if interims:
        print(f"\n📝 中间结果 ({len(interims)} 次变化):")
        for t in interims:
            print(f"   • {t}")

    print(f"\n✅ 最终结果: {final_text or '<空>'}")
    print(f"{'='*50}")

    return final_text


def run_file_callback_test(samples: np.ndarray, sample_rate: int,
                           model_path: Path):
    """带回调风格的识别测试（类似 test_tencent_asr.py 风格）。"""
    import sherpa_onnx  # noqa

    recognizer = create_recognizer(model_path)
    results = []
    t0 = time.time()

    def on_interim(text, ts):
        results.append(("interim", text, ts))
        print(f"  📝 [{ts:5.2f}s] {text}")

    def on_final(text, ts):
        results.append(("final", text, ts))
        print(f"  ✅ [{ts:5.2f}s] {text}")

    def on_complete(text, ts):
        results.append(("complete", text, ts))

    print("\n🔊 识别中（回调模式）...")
    final = recognize_file_streaming_callback(
        recognizer, samples, sample_rate,
        on_interim=on_interim, on_final=on_final, on_complete=on_complete,
    )

    elapsed = time.time() - t0
    print(f"\n{'='*50}")
    print(f"⏱  总耗时: {elapsed:.1f}s")
    finals = [r for r in results if r[0] == "final"]
    interims = [r for r in results if r[0] == "interim"]
    print(f"📊 中间: {len(interims)} | 最终: {len(finals)}")

    if finals:
        print(f"\n✅ 测试完成 — 最终结果: {finals[-1][1]}")
    else:
        print(f"\n⚠️  未收到最终结果")
    print(f"{'='*50}")

    return final


def main():
    p = argparse.ArgumentParser(
        description="sherpa-onnx 流式 ASR 识别测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument("--file", type=str, metavar="PATH", help="WAV 文件路径")
    g.add_argument("--sine", type=float, metavar="SEC", help="生成测试音（秒）")
    g.add_argument("--record", action="store_true", help="麦克风实时识别")

    p.add_argument("--model-dir", type=str, default=MODEL_DIR,
                   help="模型目录")
    p.add_argument("--num-threads", type=int, default=1,
                   help="推理线程数")
    p.add_argument("--callback", action="store_true",
                   help="使用回调风格 API（类似 test_tencent_asr.py）")

    # VAD 参数（仅 --record 模式有效，控制 ASR 启动/停止，不过滤音频）
    p.add_argument("--vad", action="store_true",
                   help="--record 模式下启用 VAD（检测到人声自动开始，静音超时自动结束）")
    p.add_argument("--vad-mode", type=int, default=3, choices=[0, 1, 2, 3],
                   help="VAD 灵敏度（0-3，3 最敏感，默认 %(default)s）")
    p.add_argument("--vad-timeout", type=int, default=1000,
                   help="VAD 静音超时毫秒（默认 %(default)sms）")
    p.add_argument("--vad-confirm", type=int, default=3,
                   help="VAD 语音确认连续帧数（默认 %(default)s）")

    args = p.parse_args()

    # --- 确定模型路径 ---
    model_path = Path(args.model_dir)
    if not model_path.exists():
        logger.error("模型目录不存在: %s", model_path)
        sys.exit(1)

    # --- 确定音频来源 ---
    # VAD 仅用于 --record 模式控制 ASR 生命周期，不过滤文件音频
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

        if args.callback:
            run_file_callback_test(samples, sr, model_path)
        else:
            run_file_test(samples, sr, model_path, label=args.file)

    elif args.sine:
        logger.info("🔊 生成测试音 %s 秒", args.sine)
        samples = generate_test_tone(args.sine)

        if args.callback:
            run_file_callback_test(samples, SAMPLE_RATE, model_path)
        else:
            run_file_test(samples, SAMPLE_RATE, model_path,
                          label=f"测试音 {args.sine}s")

    elif args.record:
        try:
            import sounddevice  # noqa
            import sherpa_onnx  # noqa
        except ImportError as e:
            logger.error("❌ 缺少依赖: %s\n   pip install sherpa-onnx sounddevice", e)
            sys.exit(1)

        recognizer = create_recognizer(model_path, args.num_threads)
        if args.vad:
            recognize_microphone_vad(
                recognizer,
                vad_mode=args.vad_mode,
                silence_timeout_ms=args.vad_timeout,
                speech_confirm_frames=args.vad_confirm,
            )
        else:
            recognize_microphone(recognizer)

    else:
        # 默认：使用 test.wav
        test_wav = Path(__file__).parent / "test.wav"
        if test_wav.exists():
            logger.info("📂 使用默认测试文件: %s", test_wav)
            samples, sr = read_wave(str(test_wav))
            run_file_test(samples, sr, model_path, label=str(test_wav))
        else:
            logger.info("🔊 默认生成 3s 测试音（未指定 --file / --sine / --record）")
            samples = generate_test_tone(3.0)
            run_file_test(samples, SAMPLE_RATE, model_path,
                          label="测试音 3s")


if __name__ == "__main__":
    main()