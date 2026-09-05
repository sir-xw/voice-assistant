"""
MPD 连接与后端（RealMpd / DummyMpd）。

RealMpd：python-mpd2 封装（连接参数取环境变量 MPD_HOST/MPD_PORT）。
DummyMpd：无 MPD 时的记录器（本地自检/开发用），不执行任何真实操作。
"""

from __future__ import annotations

import logging
import os
import socket
from typing import List, Optional, Tuple

from .coordinator import Intent, MpdBackend

logger = logging.getLogger("music_coordinator.mpd_conn")


class RealMpd(MpdBackend):
    """真实 MPD 后端：每次操作建立短连接（同旧 mpd_tool 的做法）。"""

    def __init__(self, host: Optional[str] = None, port: Optional[int] = None):
        self.host = host or os.environ.get("MPD_HOST", "localhost")
        self.port = port or int(os.environ.get("MPD_PORT", "6600"))

    def _client(self):
        import mpd as mpd_client  # python-mpd2（惰性导入）
        client = mpd_client.MPDClient()
        client.timeout = 10
        client.idletimeout = None
        client.connect(self.host, self.port)
        return client

    def _with_client(self, fn, *args):
        client = self._client()
        try:
            return fn(client, *args)
        finally:
            try:
                client.close()
                client.disconnect()
            except Exception:
                pass

    def apply(self, intent: Intent) -> None:
        def _do(client):
            if intent == Intent.PLAYING:
                client.pause(0)  # play 也允许在暂停位恢复
            elif intent == Intent.PAUSED:
                client.pause(1)
            elif intent == Intent.STOPPED:
                client.stop()
        try:
            self._with_client(_do)
            logger.info("[mc] MPD apply %s", intent.value)
        except Exception as exc:
            logger.warning("[mc] MPD apply %s 失败: %s", intent.value, exc)

    def status(self) -> dict:
        def _do(client):
            return dict(client.status())
        try:
            return self._with_client(_do)
        except Exception as exc:
            logger.warning("[mc] MPD status 失败: %s", exc)
            return {}

    def transport_op(self, name: str) -> None:
        def _do(client):
            if name == "next":
                client.next()
            elif name == "previous":
                client.previous()
        try:
            self._with_client(_do)
            logger.info("[mc] MPD %s", name)
        except Exception as exc:
            logger.warning("[mc] MPD %s 失败: %s", name, exc)

    # ─── 只读查询（供 MCP 工具）──────────────────────────

    def currentsong(self) -> dict:
        def _do(client):
            return _format_song(client.currentsong() or {})
        try:
            return self._with_client(_do)
        except Exception as exc:
            logger.warning("[mc] MPD currentsong 失败: %s", exc)
            return {}

    def playlist(self) -> List[dict]:
        def _do(client):
            songs = client.playlistinfo() or []
            return [_format_song(s) for s in songs[:50]]
        try:
            return self._with_client(_do)
        except Exception as exc:
            logger.warning("[mc] MPD playlistinfo 失败: %s", exc)
            return []

    def search(self, filters: dict) -> List[dict]:
        """按条件搜索（条件值做转义，同旧 mpd_tool 语义）。"""
        def _do(client):
            def esc(v: str) -> str:
                return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
            parts = [f"({key} == {esc(value)})" for key, value in filters.items()]
            songs = client.search(" AND ".join(parts)) or []
            return [_format_song(s) for s in songs[:50]]
        try:
            return self._with_client(_do)
        except Exception as exc:
            logger.warning("[mc] MPD search 失败: %s", exc)
            return []


def _format_song(song: dict) -> dict:
    """格式化歌曲信息，只保留关键字段（同旧 mpd_tool._format_song）。"""
    return {
        "file": song.get("file", ""),
        "artist": song.get("artist", "unknown"),
        "album": song.get("album", "unknown"),
        "title": song.get("title", song.get("file", "unknown")),
        "duration": song.get("duration", "0"),
    }


class DummyMpd(MpdBackend):
    """记录型后端（自检/开发）：ops 为 [(kind, detail), ...]。"""

    def __init__(self, initial_state: str = "play"):
        self.ops: List[Tuple[str, str]] = []
        self._state = initial_state  # play / pause / stop

    def apply(self, intent: Intent) -> None:
        self.ops.append(("apply", intent.value))
        self._state = {"playing": "play", "paused": "pause",
                       "stopped": "stop"}[intent.value]

    def status(self) -> dict:
        return {"state": self._state}

    def transport_op(self, name: str) -> None:
        self.ops.append(("transport", name))

    # 只读查询（供 MCP 工具）：Dummy 返回空/预设
    def currentsong(self) -> dict:
        return {"file": "dummy.mp3", "artist": "Dummy", "album": "测试",
                "title": "无 MPD（Dummy 模式）", "duration": "0"}

    def playlist(self) -> List[dict]:
        return [self.currentsong()]

    def search(self, filters: dict) -> List[dict]:
        return [self.currentsong()]
