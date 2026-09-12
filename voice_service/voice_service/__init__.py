"""
voice_service — 语音服务（Voice Service）。

独立进程：音频采集 / sherpa 唤醒词 / VAD / 腾讯云 ASR / 腾讯云 TTS / 播放，
对外提供 WebSocket 接入点。零 hermes 依赖。

架构设计见 docs/voice-service-websocket-architecture.md。
"""

__version__ = "0.1.0"
