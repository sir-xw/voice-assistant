"""
Music Coordinator hold 客户端（Voice Service 侧，Unix socket JSON lines）。

每次出声（TTS/提示音/资产/等待音）前 hold()、静默后 release()，由协调器
按「intent + hold 计数」决定音乐是否真的暂停/恢复 —— Voice Service 不需要
判断会话结束后音乐该播放还是暂停（架构文档 §10）。

- 连接不可达时静默降级（记录一次告警，不阻断语音）；
- 短连接式：每次操作建一次 Unix socket 连接，简单可靠（低频率调用）。
"""

from __future__ import annotations

import json
import logging
import socket
from typing import Optional

logger = logging.getLogger("voice_service.music_client")


class MusicHoldClient:
    def __init__(self, socket_path: str):
        self.socket_path = socket_path or ""
        self._held = False
        self._warned = False

    def _rpc(self, op: str, reason: str = "tts") -> Optional[dict]:
        if not self.socket_path:
            return None
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                s.settimeout(2.0)
                s.connect(self.socket_path)
                s.sendall((json.dumps({"op": op, "reason": reason}) + "\n").encode())
                resp = s.recv(4096).decode("utf-8", "replace")
            return json.loads(resp.splitlines()[0]) if resp.strip() else None
        except Exception as exc:
            if not self._warned:
                logger.warning("[music] 协调器不可达（%s），音乐避让降级: %s",
                               self.socket_path, exc)
                self._warned = True
            return None

    def hold(self, reason: str = "tts") -> bool:
        """出声前调用：暂停音乐（幂等，只发一次 hold）。"""
        if self._held:
            return True
        resp = self._rpc("hold", reason)
        if resp is None or not resp.get("ok", False):
            return False
        self._held = True
        return True

    def release(self, reason: str = "tts") -> bool:
        """静默后调用：交给协调器按 intent 决定恢复/保持。"""
        if not self._held:
            return True
        self._rpc("release", reason)
        self._held = False
        return True

    def close(self) -> None:
        self.release()
