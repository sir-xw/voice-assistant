#!/usr/bin/env python
"""
手动测试：腾讯云实时语音识别（边录边发 + VAD 自动结束）。

录音和 ASR 发送同步进行，VAD 检测到静音超时后自动停止。
通过 threading.Event 阻塞等待 SDK 回调，无需固定 sleep。

用法:
    # VAD 录音（静音 1s 自动停止）
    python -u tests/test_tencent_asr.py --record

    # 自定义参数
    python -u tests/test_tencent_asr.py --record --vad-timeout 2 --max-duration 30

    # 使用已有文件或测试音
    python -u tests/test_tencent_asr.py --file test.wav
    python -u tests/test_tencent_asr.py --sine 3

依赖:
    pip install sounddevice scipy webrtcvad  (录音模式)
"""

import argparse
import logging
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from asr_engine import TencentCloudASREngine, TencentASRConfig, ASRState
from config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("real_asr_test")

SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_SIZE = int(SAMPLE_RATE * FRAME_MS / 1000)
CHUNK_SIZE = 6400


def generate_test_pcm(duration_sec: float = 3.0) -> bytes:
    import numpy as np
    n = int(SAMPLE_RATE * duration_sec)
    silence = np.zeros(int(SAMPLE_RATE * 0.5), dtype=np.int16)
    t = np.linspace(0, duration_sec - 0.5, int(SAMPLE_RATE * (duration_sec - 0.5)))
    tone = (np.sin(2 * np.pi * 440 * t) * 8000).astype(np.int16)
    return np.concatenate([silence, tone]).tobytes()


def load_wav_to_pcm(path: str) -> bytes:
    from scipy.io import wavfile
    import numpy as np
    from scipy import signal
    sr, data = wavfile.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1).astype(data.dtype)
    if sr != SAMPLE_RATE:
        n = round(len(data) * SAMPLE_RATE / sr)
        data = signal.resample(data, n).astype(data.dtype)
    return data.tobytes()


def make_engine(engine_model="16k_zh", needvad=False):
    """从 .env + config.yaml 创建 ASR 引擎。"""
    from dotenv import load_dotenv
    import os

    dotenv_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(dotenv_path)

    sid = os.getenv("VOICE_SecretId", "")
    skey = os.getenv("VOICE_SecretKey", "")
    appid = os.getenv("VOICE_AppId", "")

    if not sid or not skey:
        print("❌ 缺少 .env 中的 VOICE_SecretId / VOICE_SecretKey")
        sys.exit(1)
    if not appid:
        print("❌ 缺少 .env 中的 VOICE_AppId")
        print("   登录 https://console.cloud.tencent.com/cam/capi 查看")
        sys.exit(1)

    cfg = TencentASRConfig(
        secret_id=sid, secret_key=skey, app_id=appid,
        engine_model=engine_model, needvad=needvad, voice_format=1,
    )
    return TencentCloudASREngine(cfg)


def run_vad_live(vad_timeout=1.0, max_dur=30.0, vad_mode=3):
    """
    VAD 录音 + 实时 ASR 识别。

    使用 sounddevice InputStream 回调模式，每帧同时做两件事：
      1. VAD 判断是否有人声
      2. 累积到 6400 字节后通过 feed_audio 发送

    连续静音超过 vad_timeout 秒 → 结束。
    收到 on_complete 回调 → threading.Event 放行。
    """
    import sounddevice as sd
    import webrtcvad
    import numpy as np

    vad = webrtcvad.Vad(vad_mode)
    engine = make_engine()

    results = []
    t0 = time.time()
    done_evt = threading.Event()
    buf = bytearray()

    def on_start():
        print("  ✅ ASR 已连接")

    def on_interim(text):
        results.append(("interim", text, time.time() - t0))

    def on_final(text):
        results.append(("final", text, time.time() - t0))
        print(f"\n  ✅ 最终: {text}")

    def on_complete():
        results.append(("complete", "", time.time() - t0))
        done_evt.set()

    def on_error(msg):
        print(f"\n  ❌ {msg}")
        done_evt.set()

    engine.on_start = on_start
    engine.on_interim = on_interim
    engine.on_final = on_final
    engine.on_complete = on_complete
    engine.on_error = on_error

    print("\n🔊 连接腾讯云 ASR...")
    engine.start_recognition()

    # 等连接
    for _ in range(50):
        if engine.state in (ASRState.RECORDING, ASRState.ERROR):
            break
        time.sleep(0.1)
    if engine.state == ASRState.ERROR:
        print("❌ 连接失败")
        return results, time.time() - t0

    # 录音回调
    speech_frames = 0
    silence_frames = 0
    speech_started = False
    limit_silence = int(vad_timeout / (FRAME_MS / 1000))
    chunk_buf = bytearray()

    def callback(indata, frames, ti, status):
        nonlocal speech_frames, silence_frames, speech_started

        pcm = indata.tobytes()
        is_speech = vad.is_speech(pcm, SAMPLE_RATE)

        if is_speech:
            silence_frames = 0
            speech_frames += 1
            if not speech_started and speech_frames >= 7:
                speech_started = True
                print("🎤", end="", flush=True)
        else:
            if speech_started:
                silence_frames += 1

        # 累积到 chunk 发送
        chunk_buf.extend(pcm)
        while len(chunk_buf) >= CHUNK_SIZE:
            engine.feed_audio(bytes(chunk_buf[:CHUNK_SIZE]))
            del chunk_buf[:CHUNK_SIZE]

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="int16",
        blocksize=FRAME_SIZE, callback=callback,
    )
    stream.start()

    # 等待结束条件
    print(f"\n🎤 VAD 录音中（静音 {vad_timeout}s 自动停）...")
    try:
        while not done_evt.is_set():
            elapsed = time.time() - t0
            if elapsed >= max_dur:
                print(f"\n  ⏰ 超时 {max_dur}s")
                break
            if speech_started and silence_frames >= limit_silence:
                actual = silence_frames * FRAME_MS / 1000
                print(f"\n  🔇 静音 {actual:.1f}s")
                break
            done_evt.wait(0.2)
    except KeyboardInterrupt:
        print("\n  ⏹ 中断")
    finally:
        stream.stop()
        stream.close()
        # 发完缓冲区
        if chunk_buf:
            engine.feed_audio(bytes(chunk_buf))
        engine.stop_recognition()
        # 阻塞等待 on_complete
        if not done_evt.wait(10):
            print("⚠️  ASR 未在 10s 内回调 complete")

    if not speech_started:
        print("⚠️  未检测到人声")

    return results, time.time() - t0


def run_file(pcm_data, chunk_size=CHUNK_SIZE):
    """使用已有 PCM 数据识别。"""
    engine = make_engine()
    results = []
    t0 = time.time()
    done_evt = threading.Event()

    def on_start():
        print("  ✅ ASR 已连接")

    def on_interim(text):
        results.append(("interim", text, time.time() - t0))

    def on_final(text):
        results.append(("final", text, time.time() - t0))
        print(f"\n  ✅ 最终: {text}")

    def on_complete():
        results.append(("complete", "", time.time() - t0))
        done_evt.set()

    def on_error(msg):
        print(f"\n  ❌ {msg}")
        done_evt.set()

    engine.on_start = on_start
    engine.on_interim = on_interim
    engine.on_final = on_final
    engine.on_complete = on_complete
    engine.on_error = on_error

    print("\n🔊 连接腾讯云 ASR...")
    engine.start_recognition()

    for _ in range(50):
        if engine.state in (ASRState.RECORDING, ASRState.ERROR):
            break
        time.sleep(0.1)
    if engine.state == ASRState.ERROR:
        print("❌ 连接失败")
        return results, time.time() - t0

    print(f"📤 发送 {len(pcm_data)} bytes...")
    for i in range(0, len(pcm_data), chunk_size):
        chunk = pcm_data[i:i + chunk_size]
        if len(chunk) < chunk_size:
            chunk = chunk + b"\x00" * (chunk_size - len(chunk))
        engine.feed_audio(chunk)
        time.sleep(chunk_size / SAMPLE_RATE / 2)  # 模拟实时率

    engine.stop_recognition()

    if not done_evt.wait(10):
        print("⚠️  ASR 未在 10s 内回调 complete")

    return results, time.time() - t0


def print_results(results, elapsed):
    print(f"\n{'='*50}")
    print(f"⏱  总耗时: {elapsed:.1f}s")
    finals = [r for r in results if r[0] == "final"]
    interims = [r for r in results if r[0] == "interim"]
    print(f"📊 中间: {len(interims)} | 最终: {len(finals)}")

    for kind, text, t in results:
        icon = {"interim": "📝", "final": "✅", "complete": "🏁"}.get(kind, "•")
        print(f"   {icon} [{t:5.1f}s] {text or kind}")

    if finals:
        print("\n✅ 测试完成")
    else:
        print("\n⚠️  未收到最终结果")


def main():
    cfg = load_config().get("voice", {})
    vad_cfg = cfg.get("vad", {})
    asr_cfg = cfg.get("asr", {})
    df_vad_timeout = vad_cfg.get("silence_threshold_ms", 1000) / 1000.0
    df_vad_mode = vad_cfg.get("mode", 3)
    df_model = asr_cfg.get("engine_model", "16k_zh")

    p = argparse.ArgumentParser(
        description="腾讯云实时语音识别测试（边录边发 + VAD）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument("--record", action="store_true", help="VAD 录音（默认）")
    g.add_argument("--file", type=str, metavar="PATH", help="WAV 文件")
    g.add_argument("--sine", type=float, metavar="SEC", help="测试音")

    p.add_argument("--vad-timeout", type=float, default=df_vad_timeout,
                   help=f"静音超时秒（默认 {df_vad_timeout}s，来自 config）")
    p.add_argument("--max-duration", type=float, default=30.0,
                   help="最长录音秒（默认 30）")
    p.add_argument("--vad-mode", type=int, default=df_vad_mode, choices=[0,1,2,3],
                   help=f"VAD 模式（默认 {df_vad_mode}，来自 config）")
    p.add_argument("--model", type=str, default=df_model,
                   help=f"引擎模型（默认 {df_model}，来自 config）")
    p.add_argument("--save", type=str, metavar="PATH", help="保存录音为 WAV")

    args = p.parse_args()

    # --- 确定音频来源 ---
    if args.file:
        print(f"📂 加载 {args.file}")
        pcm = load_wav_to_pcm(args.file)
        print(f"   长度: {len(pcm)} bytes ({len(pcm)/32000:.1f}s)")
        results, elapsed = run_file(pcm)
        print_results(results, elapsed)

    elif args.sine:
        print(f"🔊 测试音 {args.sine}s")
        pcm = generate_test_pcm(args.sine)
        results, elapsed = run_file(pcm)
        print_results(results, elapsed)

    else:
        # 默认：VAD 录音
        try:
            import sounddevice  # noqa
            import webrtcvad   # noqa
        except ImportError as e:
            print(f"❌ 缺少依赖: {e}\n   pip install sounddevice scipy webrtcvad")
            return

        results, elapsed = run_vad_live(
            vad_timeout=args.vad_timeout,
            max_dur=args.max_duration,
            vad_mode=args.vad_mode,
        )
        print_results(results, elapsed)


if __name__ == "__main__":
    main()

