"""
WebRTC VAD 工具集（本地语音活动检测）。

被 voice_frontend.py 与 tests/ 共用，避免各测试脚本重复实现：
  - init_vad: 初始化 webrtcvad（未安装时返回 None，调用方据此禁用 VAD）
  - is_speech_frame: 判断一帧 int16 PCM 是否含人声
  - has_confirmed_run: 滑动窗口连续确认帧判定（用于语音结束检测）

VAD 只决定何时启动/停止 ASR，不过滤音频样本；识别期间所有帧全量送入，
不做静音去除。
"""

from collections import deque


def init_vad(mode: int = 3):
    """初始化 WebRTC VAD，未安装 webrtcvad 时返回 None。"""
    try:
        import webrtcvad
        return webrtcvad.Vad(mode)
    except ImportError:
        return None


def is_speech_frame(vad, pcm_int16: bytes, sample_rate: int = 16000) -> bool:
    """判断一帧 int16 PCM 是否含人声（vad 为 None 时视为语音）。"""
    if vad is None:
        return True
    try:
        return vad.is_speech(pcm_int16, sample_rate)
    except Exception:
        return False


def has_confirmed_run(recent_flags: deque, confirm_frames: int) -> bool:
    """检查窗口内是否存在连续 confirm_frames 个 is_speech 帧。

    用于对话结束判定：最近 silence_timeout_frames 帧内只要出现过连续确认语音帧，
    就认为对话仍在进行；偶然的孤立噪音帧（不足 confirm_frames 帧连续）不会命中，
    从而不会延长对话窗口、增加 ASR 延迟。
    """
    run = 0
    for flag in recent_flags:
        run = run + 1 if flag else 0
        if run >= confirm_frames:
            return True
    return False
