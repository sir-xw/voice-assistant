#!/usr/bin/env python

"""
播放-录音延迟探测工具。

播放白噪声，从麦克风录制，通过互相关分析计算
扬声器到麦克风的路径延迟。

注意：AEC 已由 Linux PipeWire 系统级处理，不再需要配置延迟参数。
此工具仅保留用于系统延迟诊断。

用法:
    python -u tools/measure_delay.py                    # 默认 3 轮
    python -u tools/measure_delay.py --rounds 5         # 跑 5 轮取平均
    python -u tools/measure_delay.py --duration 2.0     # 每轮 2 秒噪声
    python -u tools/measure_delay.py --volume 0.5       # 音量 50%

依赖:
    pip install sounddevice scipy numpy
"""

import argparse
import logging
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import sounddevice as sd
from scipy import signal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("measure_delay")

SAMPLE_RATE = 16000
BLOCK_SIZE = 1024


# ─── 核心探测 ────────────────────────────────────────────

def gen_chirp(duration: float = 1.0, volume: float = 0.3,
              silence_sec: float = 0.2) -> np.ndarray:
    """
    生成线性扫频信号（Chirp），200Hz→4000Hz，前导静音做时间锚点。

    Args:
        duration: 扫频时长（秒）
        volume: 音量 0~1
        silence_sec: 前导静音（秒），用于区分电路串扰和真实回声

    Returns:
        float32 数组（长度 = silence_sec + duration）
    """
    # 前导静音（时间锚点）
    silence = np.zeros(int(SAMPLE_RATE * silence_sec), dtype=np.float32)
    # 扫频
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    f0, f1 = 200.0, 4000.0
    sweep = np.sin(2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * duration)))
    sweep = (sweep * volume * 0.9).astype(np.float32)
    return np.concatenate([silence, sweep])


def play_signal(signal_float: np.ndarray, duration: float = 1.0, volume: float = 0.3) -> tuple:
    """
    播放信号并同时录音。

    Args:
        signal_float: float32 播放信号
        volume: 音量 0~1

    Returns:
        (played_int16, recorded_int16)
    """
    num_samples = len(signal_float)

    # 生成 int16 参考信号（用于互相关）
    played = (signal_float / 0.9 * 16000).astype(np.int16)

    recorded = np.zeros(num_samples, dtype=np.int16)
    recording_started = threading.Event()
    recorded_chunks: list[np.ndarray] = []
    chunk_lock = threading.Lock()

    def record_callback(indata, frames, time_info, status):
        if status:
            logger.warning(f"Record status: {status}")
        with chunk_lock:
            recorded_chunks.append(indata.copy())
        recording_started.set()

    # 录音流（低延迟模式）
    record_stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=BLOCK_SIZE,
        latency="low",
        callback=record_callback,
    )

    # 播放流（低延迟模式）
    play_stream = sd.OutputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=BLOCK_SIZE,
        latency="low",
    )

    record_stream.start()
    # 等录音启动
    recording_started.wait(timeout=2)

    play_stream.start()
    play_stream.write(signal_float)

    # 等播放完 + 录音收尾
    play_time = duration + 0.3
    time.sleep(play_time)

    play_stream.stop()
    play_stream.close()
    record_stream.stop()
    record_stream.close()

    with chunk_lock:
        if recorded_chunks:
            recorded = np.concatenate(recorded_chunks).flatten()[:num_samples]
        else:
            recorded = np.zeros(num_samples, dtype=np.int16)

    return played.flatten(), recorded.flatten()


def estimate_delay(played: np.ndarray, recorded: np.ndarray,
                   max_delay_ms: float = 500.0) -> float:
    """
    通过互相关计算播放到录制的延迟。

    Args:
        played: 播放信号 (int16)
        recorded: 录制信号 (int16)
        max_delay_ms: 最大预期延迟（毫秒）

    Returns:
        估算的延迟（毫秒），负值表示不可靠
    """
    max_lag = int(SAMPLE_RATE * max_delay_ms / 1000)

    # 统一做归一化
    p = played.astype(np.float64)
    p = (p - p.mean()) / (p.std() + 1e-10)
    r = recorded.astype(np.float64)
    r = (r - r.mean()) / (r.std() + 1e-10)

    # 互相关
    correlation = signal.correlate(r, p, mode="same", method="fft")
    lags = signal.correlation_lags(len(r), len(p), mode="same")

    # 只搜前 max_lag 个 lag（播放先于录制）
    valid = (lags >= 0) & (lags <= max_lag)
    if not valid.any():
        return -1.0

    peak_idx = np.argmax(correlation * valid)
    peak_lag = lags[peak_idx]
    peak_value = correlation[peak_idx]

    # 信噪比检查：峰值需大于阈值
    noise_floor = np.std(correlation[valid]) * 3
    if peak_value < noise_floor:
        return -1.0

    # 扣除前导静音（200ms）
    delay_ms = peak_lag / SAMPLE_RATE * 1000 - 200
    return max(0, delay_ms)


# ─── 多轮探测 ────────────────────────────────────────────

@dataclass
class DelayResult:
    delays: list[float] = field(default_factory=list)
    mean: float = 0.0
    min: float = 0.0
    max: float = 0.0
    std: float = 0.0


def run_measurement(rounds: int = 3, duration: float = 1.0,
                    volume: float = 0.3, max_delay_ms: float = 500.0) -> DelayResult:
    """
    运行多轮延迟探测。

    Returns:
        DelayResult 包含全部统计
    """
    result = DelayResult()

    print(f"\n{'='*50}")
    print(f"🔊 播放-录音延迟探测")
    print(f"{'='*50}")
    print(f"   轮数: {rounds}  扫频时长: {duration}s  音量: {volume:.0%}")
    print(f"   采样率: {SAMPLE_RATE}Hz  最大搜索: {max_delay_ms}ms")
    print(f"   ⚠️ 确保扬声器和麦克风都打开，环境安静")
    print(f"{'='*50}\n")

    for i in range(rounds):
        print(f"▶ 第 {i+1}/{rounds} 轮...", end=" ", flush=True)

        chirp = gen_chirp(duration, volume)
        played, recorded = play_signal(chirp, duration, volume)
        delay = estimate_delay(played, recorded, max_delay_ms)

        if delay >= 0:
            result.delays.append(delay)
            print(f"延迟: {delay:.1f}ms")
        else:
            print(f"❌ 未检测到有效信号（音量太低或麦克风未开？）")

        # 轮间间隔
        if i < rounds - 1:
            time.sleep(0.5)

    if result.delays:
        result.mean = float(np.mean(result.delays))
        result.min = float(np.min(result.delays))
        result.max = float(np.max(result.delays))
        result.std = float(np.std(result.delays))

    return result


# ─── 入口 ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="播放-录音延迟探测工具（系统诊断）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -u tools/measure_delay.py
  python -u tools/measure_delay.py --rounds 5 --duration 1.5
  python -u tools/measure_delay.py --volume 0.5 --max-delay 300

注意: AEC 已由 PipeWire 系统级处理，无需配置延迟参数。
此工具仅保留用于系统延迟诊断。
        """,
    )

    parser.add_argument("--rounds", type=int, default=3,
                        help="探测轮数（默认 3）")
    parser.add_argument("--duration", type=float, default=1.0,
                        help="每轮噪声时长秒（默认 1.0）")
    parser.add_argument("--volume", type=float, default=0.3,
                        help="播放音量 0~1（默认 0.3）")
    parser.add_argument("--max-delay", type=float, default=500.0,
                        help="最大搜索延迟毫秒（默认 500）")

    args = parser.parse_args()

    result = run_measurement(
        rounds=args.rounds,
        duration=args.duration,
        volume=args.volume,
        max_delay_ms=args.max_delay,
    )

    print(f"\n{'='*50}")
    print(f"📊 延迟统计")
    print(f"{'='*50}")
    if result.delays:
        print(f"   有效样本: {len(result.delays)}/{args.rounds}")
        print(f"   平均值:   {result.mean:.1f} ms")
        print(f"   最小值:   {result.min:.1f} ms")
        print(f"   最大值:   {result.max:.1f} ms")
        print(f"   标准差:   {result.std:.1f} ms")
        print(f"\n   注意: AEC 已由 PipeWire 系统级处理，无需配置延迟参数。")
        print(f"   此工具仅保留用于系统延迟诊断。")
    else:
        print(f"   ❌ 无有效数据")
        print(f"   建议:")
        print(f"     1. 调大 --volume")
        print(f"     2. 检查麦克风是否开启")
        print(f"     3. 检查扬声器是否出声")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
