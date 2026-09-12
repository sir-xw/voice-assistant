"""
music_coordinator — 音乐状态协调器。

MPD 唯一写入口：intent 状态机 + hold 计数 → effective 状态 → 实际 MPD 操作。
零 hermes 依赖。架构设计见 docs/voice-service-websocket-architecture.md §10。
"""

__version__ = "0.1.0"
