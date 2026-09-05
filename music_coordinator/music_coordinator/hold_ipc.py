"""
hold IPC：Unix socket 服务（Voice Service → Music Coordinator）。

帧格式：一行 JSON（\n 分隔），请求 {op, reason?, seq?}：
- {"op":"hold",    "reason":"tts"}
- {"op":"release", "reason":"tts"}
- {"op":"status"}
应答：{"ok":true, "intent":..., "hold_count":..., "effective":...}
或 {"ok":false, "error":"..."}

全部操作在 asyncio 事件循环内串行执行（单实例协调器无并发写问题）。
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, Optional

from .coordinator import MusicCoordinator

logger = logging.getLogger("music_coordinator.hold_ipc")

VALID_OPS = ("hold", "release", "status")


class HoldIpcServer:
    """基于 asyncio Unix stream server 的 hold/release/status 服务。"""

    def __init__(self, coordinator: MusicCoordinator, socket_path: str):
        self.coordinator = coordinator
        self.socket_path = socket_path
        self._server: Optional[asyncio.AbstractServer] = None

    async def start(self) -> None:
        import os
        try:
            os.unlink(self.socket_path)
        except FileNotFoundError:
            pass
        os.makedirs(os.path.dirname(self.socket_path) or ".", exist_ok=True)
        self._server = await asyncio.start_unix_server(
            self._handle_conn, self.socket_path)
        os.chmod(self.socket_path, 0o666)  # Voice Service 可能不同用户
        logger.info("[mc] hold IPC 监听 %s", self.socket_path)

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    async def _handle_conn(self, reader: asyncio.StreamReader,
                           writer: asyncio.StreamWriter) -> None:
        peer = writer.get_extra_info("peername")
        try:
            while True:
                raw = await reader.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", "replace").strip()
                if not line:
                    continue
                resp = self._dispatch(line)
                writer.write((json.dumps(resp, ensure_ascii=False) + "\n").encode())
                await writer.drain()
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        except Exception as exc:
            logger.warning("[mc] hold IPC 连接异常(%s): %s", peer, exc)
        finally:
            writer.close()

    def _dispatch(self, line: str) -> Dict[str, Any]:
        try:
            req = json.loads(line)
            op = req.get("op")
            if op not in VALID_OPS:
                return {"ok": False, "error": f"未知 op: {op}"}
            reason = str(req.get("reason") or "tts")
            if op == "hold":
                self.coordinator.hold(reason)
            elif op == "release":
                self.coordinator.release(reason)
            snap = self.coordinator.snapshot()
            return {"ok": True, **snap}
        except Exception as exc:
            logger.warning("[mc] hold IPC 请求解析失败: %s", exc)
            return {"ok": False, "error": str(exc)}
