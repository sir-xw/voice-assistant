"""
进程内播报桥：插件回调（hermes agent 线程）→ VoiceApp（asyncio 播放队列）。

hermes 的 post_api_request 钩子回调运行在 agent 的调用线程里，无法直接拿到
VoiceApp 实例。因此 VoiceApp 启动时通过 :func:`set_sink` 注册自己，插件
（.hermes/plugins/speech-relay）的回调里调用 :func:`emit` 把回复载荷转交
给 VoiceApp 处理（VoiceApp 内部再投递到 asyncio 队列播放）。

桥只做线程安全的转发，不做任何业务判断。
"""

import logging
import threading

logger = logging.getLogger("speech_bridge")

_lock = threading.Lock()
_sink = None  # Callable[[dict], None] | None


def set_sink(callback) -> None:
    """注册播报接收方（VoiceApp 启动时调用）。"""
    global _sink
    with _lock:
        _sink = callback


def clear_sink() -> None:
    """注销播报接收方（VoiceApp 停止时调用）。"""
    global _sink
    with _lock:
        _sink = None


def emit(payload: dict) -> None:
    """插件回调入口：把 post_api_request 载荷转发给已注册的接收方。"""
    sink = None
    with _lock:
        sink = _sink
    if sink is None:
        return
    try:
        sink(payload)
    except Exception:
        logger.exception("播报桥转发失败")
