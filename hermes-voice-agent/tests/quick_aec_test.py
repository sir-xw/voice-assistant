#!/usr/bin/env python3
"""
快速 AEC 验证：验证 PipeWire 系统级回声消除效果。

两种回声源模式：
  1. 文件模式（默认）：用预设 WAV 文件作为回声源
  2. 录音模式（--record）：VAD 录音作为回声源（需 webrtcvad）

不调用任何软件 AEC 库，依赖系统 PipeWire 提供的回声消除。

用法:
    python -u tests/quick_aec_test.py                          # 文件模式，默认 assets/prompt.wav
    python -u tests/quick_aec_test.py --file assets/farewell.wav
    python -u tests/quick_aec_test.py --volume 0.3             # 播放音量倍率
    python -u tests/quick_aec_test.py --record                 # 录音模式（VAD）代替预设音频
    python -u tests/quick_aec_test.py --record --out-dir /tmp
"""

import argparse
import logging
import threading
import time
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("quick_aec_test")

SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_SIZE = int(SAMPLE_RATE * FRAME_MS / 1000)


def load_wav(path: str) -> np.ndarray:
    """从文件加载 WAV 音频。"""
    with wave.open(str(path), "rb") as wf:
        data = wf.readframes(wf.getnframes())
    return np.frombuffer(data, dtype=np.int16)


def save_wav(path: str, data: np.ndarray):
    """保存音频为 WAV 文件。"""
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(data.tobytes())


def record_vad(vad_timeout: float = 1.0, vad_mode: int = 3, max_duration: float = 15.0) -> np.ndarray:
    """VAD 录音：检测人声开始，静音超时结束。

    需要安装 webrtcvad: pip install webrtcvad
    """
    try:
        import webrtcvad
    except ImportError:
        logger.error("录音模式需要 webrtcvad，请先安装：pip install webrtcvad")
        raise

    vad = webrtcvad.Vad(vad_mode)
    frames, silent, speech_frames = [], 0, 0
    speech_detected = False
    limit = int(vad_timeout / (FRAME_MS / 1000))
    t0 = time.time()

    def cb(indata, frames_count, ti, status):
        nonlocal silent, speech_frames, speech_detected
        if status:
            return
        pcm = indata.tobytes()
        is_sp = vad.is_speech(pcm, SAMPLE_RATE)
        if is_sp:
            silent = 0
            speech_frames += 1
            if not speech_detected and speech_frames >= 7:
                speech_detected = True
                print("🎤", end="", flush=True)
        else:
            if speech_detected:
                silent += 1
        frames.append(pcm)

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="int16",
        blocksize=FRAME_SIZE, latency="low", callback=cb,
    )
    stream.start()
    print(f"\n🎤 录音（静音 {vad_timeout}s 自动停）...  请说话", end="")
    try:
        while True:
            if time.time() - t0 >= max_duration:
                break
            if speech_detected and silent >= limit:
                print(f"\n  🔇 静音 {silent * FRAME_MS / 1000:.1f}s")
                break
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n  ⏹")
    finally:
        stream.stop()
        stream.close()

    if not speech_detected:
        print("\n  ⚠️  未检测到人声")
    audio = np.frombuffer(b"".join(frames), dtype=np.int16)
    print(f"  ✅ {len(audio) / SAMPLE_RATE:.1f}s")
    return audio


def test_aec(source_path: str, volume: float = 1.0, out_dir: str = ".", record_mode: bool = False):
    """播放回声源（文件或 VAD 录音）+ 同时录制，验证 PipeWire AEC 消除效果。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # === 获取回声源 ===
    if record_mode:
        print(f"\n{'=' * 50}")
        print("📢 模式：VAD 录音（作为回声源）")
        print(f"{'=' * 50}")
        source = record_vad()
        name = "echo_source"
        save_wav(str(out / "echo_source.wav"), source)
        print(f"  源文件: {out / 'echo_source.wav'}")
    else:
        source = load_wav(source_path)
        name = Path(source_path).stem
        print(f"\n📢 模式：文件加载")
        print(f"  源音频: {Path(source_path).name} ({len(source) / SAMPLE_RATE:.1f}s)")

    print(f"  音量:   {volume:.0%}")
    print(f"  AEC:    PipeWire 系统级（默认启用）")

    # === 播放 + 同时录制 ===
    print(f"\n{'=' * 50}")
    print("📢 播放回声源 + 录制（含系统 PipeWire AEC）")
    print(f"{'=' * 50}")
    print("  🔊 开始播放... 请保持安静")

    raw_frames = []
    rec_ready = threading.Event()

    def rec_cb(indata, frames, ti, status):
        if status:
            return
        raw_frames.append(indata.tobytes())
        rec_ready.set()

    rec = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="int16",
        blocksize=FRAME_SIZE, latency="low", callback=rec_cb,
    )
    rec.start()
    rec_ready.wait(2)

    audio_out = (source.astype(np.float32) * volume).astype(np.int16)
    sd.play(audio_out, SAMPLE_RATE)
    sd.wait()

    time.sleep(0.3)
    rec.stop()
    rec.close()

    raw = np.frombuffer(b"".join(raw_frames), dtype=np.int16)

    # === 保存结果 ===
    raw_path = out / f"aec_raw_{name}.wav"
    save_wav(str(raw_path), raw)
    print(f"\n💾 已保存: {raw_path}  ({len(raw) / SAMPLE_RATE:.1f}s)")

    # === 播放结果对比 ===
    print(f"\n{'=' * 50}")
    print("📢 对比播放录制结果（含 PipeWire AEC）")
    print(f"{'=' * 50}")
    print("\n▶ 正在播放...")
    time.sleep(0.5)
    sd.play(raw, SAMPLE_RATE)
    sd.wait()

    # === 汇总 ===
    print(f"\n{'=' * 50}")
    print("📊 测试汇总")
    print(f"{'=' * 50}")
    print(f"   模式:    {'录音 (VAD)' if record_mode else '文件'}")
    if not record_mode:
        print(f"   源文件:  {Path(source_path).name} ({len(source) / SAMPLE_RATE:.1f}s)")
    print(f"   录制结果: {raw_path.name} ({len(raw) / SAMPLE_RATE:.1f}s)")
    print(f"   音量:    {volume:.0%}")
    print(f"\n   判断标准：")
    print(f"     人声应清晰，播放的回声应被 PipeWire AEC 消除")


def main():
    assets = Path(__file__).resolve().parent.parent / "assets"
    p = argparse.ArgumentParser(
        description="AEC 回声消除测试（PipeWire 系统级）",
        epilog="提示：录音模式需要 pip install webrtcvad",
    )
    p.add_argument("--file", default=str(assets / "prompt.wav"), help="源音频（录音模式忽略）")
    p.add_argument("--volume", type=float, default=1.0, help="播放音量倍率 (0.0~1.0)")
    p.add_argument("--out-dir", default="/tmp", help="输出目录")
    p.add_argument("--record", action="store_true", help="启用 VAD 录音模式（代替 --file）")
    args = p.parse_args()
    test_aec(
        source_path=args.file,
        volume=args.volume,
        out_dir=args.out_dir,
        record_mode=args.record,
    )


if __name__ == "__main__":
    main()