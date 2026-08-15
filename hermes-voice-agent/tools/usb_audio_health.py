#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
usb_audio_health.py —— USB 音箱音频链路健康检测（QEMU usb-host 透传环境专用）

背景
----
本机是 PVE/KVM 虚拟机，USB 音箱（AIMIC-M4）经 QEMU usb-host 透传进入。
长时间运行后 QEMU 虚拟 USB 控制器的等时（isoc）传输状态积累异常，
表现：「播放时完全听不到音乐音调，只有持续低噪 + 偶尔爆音」。
此时 guest 软件层（PipeWire）的数据看似正常，但数据实际没有送达音箱。

本脚本通过两类信号检测异常：
1. 被动检查：读取 PipeWire sink 节点的 error 字段（USB 传输错误会在此暴露）
2. 主动自检：空闲时播放 2 秒低音量测试音，从 sink monitor 录回并分析。
   故障时数据链路断裂，monitor 录音中的测试音会缺失或严重中断——
   这是最直接可靠的"数据是否送达"探针。

检测到异常后（--auto 模式）自动调用 repair_usb_audio.sh 重建虚拟 USB 控制器。

用法
----
    python3 usb_audio_health.py            # 只检测，打印结果（exit 0=健康 1=异常）
    python3 usb_audio_health.py --auto     # 检测，异常时自动修复
    python3 usb_audio_health.py --selftest # 强制主动自检（忽略空闲判断）

配套：systemd 用户级 timer（usb-audio-health.timer）每小时触发一次。
"""

import json
import os
import subprocess
import sys
import tempfile
import wave
import struct
import math
import time
from pathlib import Path

# ---------------------------------------------------------------
# 常量
# ---------------------------------------------------------------
TOOLS_DIR = Path(__file__).resolve().parent
REPAIR_SCRIPT = TOOLS_DIR / "repair_usb_audio.sh"
TONE_PATH = Path("/tmp/usb_audio_health_tone.wav")   # 预生成测试音
TONE_FREQ = 1000.0                                     # 1kHz
TONE_DUR = 2.0                                         # 2 秒
TONE_AMP = 0.10                                        # -20 dBFS，低音量
REC_DUR = 3.0                                          # monitor 录音时长
PLAY_DELAY = 0.3                                       # 录音开始后延迟播放
REC_RATE = 48000

# 健康判定阈值
TONE_BAND_HZ = 40.0     # 1kHz 检测带宽 ±40Hz
TONE_MIN_DB = -12.0     # 播放时段 1kHz 峰值相对最强成分，低于此值视为测试音缺失
SILENT_RATIO_MAX = 0.30  # 播放时段静音帧（< 峰值10%）比例，超过视为信号中断

# 状态文件：记录上次异常时间，防止短时间重复修复
STATE_FILE = Path("/tmp/usb_audio_health_state.json")
REPAIR_MIN_INTERVAL = 300  # 两次修复最小间隔（秒）


# ---------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------
def log(msg):
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


def run(cmd, timeout=15):
    """运行命令，返回 (returncode, stdout, stderr)。"""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return 124, "", "timeout"


def gen_tone():
    """生成测试音 wav（48kHz 单声道 1kHz，带淡入淡出防爆音）。"""
    if TONE_PATH.exists():
        return True
    sr = REC_RATE
    n = int(sr * TONE_DUR)
    fade = int(0.02 * sr)
    frames = bytearray()
    for i in range(n):
        t = i / sr
        v = TONE_AMP * math.sin(2 * math.pi * TONE_FREQ * t)
        env = 1.0
        if i < fade:
            env = i / fade
        elif i > n - fade:
            env = (n - i) / fade
        s = int(v * env * 32767)
        frames += struct.pack("<h", s)
    try:
        with wave.open(str(TONE_PATH), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(bytes(frames))
        return True
    except OSError as e:
        log(f"生成测试音失败: {e}")
        return False


def pw_dump_sink():
    """从 pw-dump 获取默认输出 sink 的节点信息，返回 dict 或 None。"""
    rc, out, err = run(["pw-dump"])
    if rc != 0:
        return None
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        return None
    for obj in data:
        info = obj.get("info", {})
        props = info.get("props", {})
        if props.get("media.class") == "Audio/Sink" and "usb" in str(props.get("node.name", "")):
            return info
    return None


def get_monitor_name():
    """获取默认 sink 的 monitor 源名称（PulseAudio 层）。"""
    rc, out, err = run(["pactl", "list", "sources", "short"])
    if rc != 0:
        return None
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) > 1 and ".monitor" in parts[1]:
            return parts[1]
    return None


def _has_running_stream(cmd):
    """解析 pactl 输出，判断是否存在 RUNNING 状态的流。

    paused（CORKED）、空闲（IDLE）的流不算活跃，不阻止自检。
    """
    rc, out, _ = run(cmd)
    if rc != 0:
        return False
    in_stream = False
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("信宿输入") or line.startswith("信源输出") \
           or line.startswith("Sink Input") or line.startswith("Source Output"):
            in_stream = True
            continue
        if in_stream and line.startswith("状态"):
            if "RUNNING" in line:
                return True
            in_stream = False
    return False


def has_active_streams():
    """是否有活跃（RUNNING）的播放/录音流（有则说明系统正忙，不适合自检）。"""
    return _has_running_stream(["pactl", "list", "sink-inputs"]) or \
        _has_running_stream(["pactl", "list", "source-outputs"])


def passive_check(sink_info):
    """被动检查：sink 节点的 error 字段。

    返回 (异常bool, 描述str)。
    """
    if sink_info is None:
        return False, "sink 信息不可用（跳过被动检查）"
    err = sink_info.get("error")
    if err is not None:
        return True, f"sink error 字段非空: {err}"
    return False, "sink error 字段正常"


def analyze_recording(raw_path, rate):
    """分析 monitor 录音：播放时段内 1kHz 是否主导、信号是否连续。

    返回 (健康bool, 描述str)。
    """
    data = np_fromfile(raw_path)
    if data is None or len(data) < rate:
        return False, "录音数据无效"

    sig = data.astype(np.float64)
    total_dur = len(sig) / rate
    # 播放时段：从 PLAY_DELAY 到 PLAY_DELAY+TONE_DUR
    t0 = int(PLAY_DELAY * rate)
    t1 = min(int((PLAY_DELAY + TONE_DUR) * rate), len(sig))
    if t1 - t0 < rate // 2:
        return False, "播放时段过短"

    seg = sig[t0:t1]
    # 1) 频谱：1kHz 峰值相对最强成分
    w = np.hanning(len(seg))
    spec = np.abs(np.fft.rfft(seg * w))
    freqs = np.fft.rfftfreq(len(seg), 1.0 / rate)
    peak_all = spec.max()
    band = (freqs > TONE_FREQ - TONE_BAND_HZ) & (freqs < TONE_FREQ + TONE_BAND_HZ)
    peak_tone = spec[band].max()
    db = 20 * math.log10((peak_tone + 1e-12) / (peak_all + 1e-12))

    # 2) 连续性：5ms 帧 RMS，静音帧比例
    f_len = int(rate * 0.005)
    rms = [math.sqrt(np.mean(seg[i:i + f_len] ** 2)) for i in range(0, len(seg) - f_len, f_len)]
    if rms:
        thr = max(rms) * 0.10
        silent = sum(1 for r in rms if r < thr) / len(rms)
    else:
        silent = 1.0

    healthy = (db > TONE_MIN_DB) and (silent <= SILENT_RATIO_MAX)
    desc = (f"1kHz 相对能量 {db:.1f}dB（要求 >{TONE_MIN_DB}），"
            f"静音帧占比 {silent*100:.0f}%（要求 ≤{SILENT_RATIO_MAX*100:.0f}%）")
    return healthy, desc


def np_fromfile(path):
    """读取 16bit 原始 PCM（单声道）。"""
    try:
        import numpy as np
    except ImportError:
        log("需要 numpy")
        return None
    try:
        raw = np.fromfile(path, dtype=np.int16)
        return raw
    except OSError:
        return None


def selftest():
    """主动自检：播放测试音 + monitor 录音分析。

    返回 (健康bool, 描述str)。录音/播放失败视为异常。
    """
    monitor = get_monitor_name()
    if not monitor:
        return False, "找不到 monitor 源"
    if not gen_tone():
        return False, "测试音生成失败"

    raw_path = f"/tmp/usb_audio_health_rec_{os.getpid()}.pcm"
    try:
        # 并行：播放 + 录音
        rec = subprocess.Popen(
            ["timeout", str(int(REC_DUR) + 1), "parec",
             "--device=" + monitor, "--rate", str(REC_RATE),
             "--channels", "1", "--format", "s16le", "--raw", raw_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        time.sleep(PLAY_DELAY)
        play = subprocess.run(
            ["timeout", str(int(TONE_DUR) + 1), "pw-play", str(TONE_PATH)],
            capture_output=True, timeout=int(TONE_DUR) + 5,
        )
        rec.wait(timeout=int(REC_DUR) + 5)

        if play.returncode != 0:
            # pw-play 失败：可能 sink 异常
            return False, f"pw-play 播放失败 rc={play.returncode}"

        healthy, desc = analyze_recording(raw_path, REC_RATE)
        return healthy, desc
    finally:
        try:
            os.unlink(raw_path)
        except OSError:
            pass


def repair():
    """调用修复脚本。返回成功与否。"""
    log("触发自动修复...")
    if not REPAIR_SCRIPT.exists():
        log(f"找不到修复脚本 {REPAIR_SCRIPT}")
        return False
    rc = subprocess.run(["bash", str(REPAIR_SCRIPT)]).returncode
    return rc == 0


def load_state():
    try:
        return json.loads(STATE_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def save_state(st):
    try:
        STATE_FILE.write_text(json.dumps(st))
    except OSError:
        pass


def main():
    auto = "--auto" in sys.argv
    force = "--selftest" in sys.argv

    problems = []
    notes = []

    # 1) 被动检查
    sink = pw_dump_sink()
    bad, desc = passive_check(sink)
    log(f"[被动] {desc}")
    if bad:
        problems.append("sink error 异常")

    # 2) 主动自检（空闲时；--selftest 强制）
    if force or not has_active_streams():
        healthy, desc = selftest()
        log(f"[自检] {desc}")
        if not healthy:
            problems.append(f"自检异常：{desc}")
    else:
        log("[自检] 系统正忙（有播放/录音流），跳过主动自检，仅被动检查")

    # 3) 汇总
    if not problems:
        log("结果：健康 ✅")
        save_state({"last_problem": None})
        return 0

    log("结果：异常 ⚠️  " + "；".join(problems))

    if not auto:
        return 1

    # 自动修复：限制最短间隔，避免反复打断
    st = load_state()
    last = st.get("last_repair", 0)
    now = time.time()
    if now - last < REPAIR_MIN_INTERVAL:
        log(f"距上次修复仅 {int(now-last)}s，跳过本次自动修复（防循环）")
        return 1
    if repair():
        st["last_repair"] = now
        st["last_problem"] = problems
        save_state(st)
        log("自动修复完成 ✅")
        return 0
    else:
        log("自动修复失败，请人工处理！")
        return 1


if __name__ == "__main__":
    try:
        import numpy as np  # noqa: F401  提前导入，保证主流程可用
    except ImportError:
        log("缺少 numpy，无法分析录音")
        sys.exit(1)
    sys.exit(main())
