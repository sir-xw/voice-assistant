"""
Voice Service 服务端（最小骨架版，M1 前置）。

当前实现（不含语音装配）：
- WebSocket 监听 + 可选 token 鉴权（Authorization: Bearer）与 HTTP GET /healthz；
- 单客户端语义：第二个并发连接被拒（409），预留 client_id 多路扩展；
- `hello` 握手（返回 welcome，附服务端实际 KWS 词表占位）与 `ping/pong` 心跳；
- `speak/control/interrupt` 先回 ack 并记日志（播放装配在 M3，见 README 进度）。

后续演进：语音状态机编排（inbound.py）与播报队列（playback.py）装配进来后，
本模块负责连接管理/帧路由；必要时再拆出 hub.py / session.py。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

from . import protocol as P
from .service_config import ServiceConfig

logger = logging.getLogger("voice_service.server")

# 非 upgrade 的 HTTP GET（探活）——若路径不是 /healthz 返回 404
HEALTHZ_PATH = "/healthz"


class VoiceServer:
    """Voice Service WS 服务端。"""

    def __init__(self, svc: ServiceConfig):
        self.svc = svc
        self._server: Optional[Any] = None
        # 事件循环（start 时记录；供 SDK/音频线程经 threadsafe_event 上行）
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        # 当前活动客户端（v1 单 client）：client_id → connection
        self._client: Optional[Any] = None
        self._client_id: str = ""
        # outbound：Playback（M3 装配后由 app 注入；None=仅骨架）
        self.playback: Optional[Any] = None
        # 说话人身份绑定（SpeakerAliases；app 在启用声纹时注入）：
        # 供 speaker_alias 帧读写 names.json，None=说话人识别未启用
        self.speaker_admin: Optional[Any] = None
        # 实际助手表（wake_word.assistants，config 唯一源；app 装配时注入）。
        # welcome.my_wakewords 回报给客户端：gateway 依此知道有哪些助手会话。
        self._assistants: list = []

    def set_assistants(self, assistants: list) -> None:
        """注入实际助手表（[{name, keywords:[...]}, ...]），welcome 回报用。"""
        self._assistants = list(assistants or [])

    # ─── 生命周期 ────────────────────────────────────────

    async def start(self) -> None:
        from websockets.asyncio.server import serve
        self.loop = asyncio.get_running_loop()
        self._server = await serve(
            self._handle_connection,
            self.svc.host,
            self.svc.port,
            process_request=self._process_request,
            ping_interval=None,       # 心跳走应用层 ping/pong
            max_size=2 ** 20,         # 1 MiB（语音帧文本足够）
        )
        logger.info("[voice_service] 监听 ws://%s:%d (token=%s)",
                    self.svc.host, self.svc.port, "已启用" if self.svc.token else "未启用(仅本机)")

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    # ─── HTTP / 鉴权 ─────────────────────────────────────

    async def _process_request(self, connection, request) -> Optional[Any]:
        """处理 HTTP 请求（/healthz 探活）与 WS token 鉴权（websockets>=17 API）。

        新 API 签名：process_request(connection, request)；request 为
        websockets.http11.Request（request.path / request.headers）。
        返回 Response 则不进入 WS 升级；返回 None 继续握手。
        """
        from websockets.datastructures import Headers
        from websockets.http11 import Response

        is_upgrade = (request.headers.get("Upgrade", "") or "").lower() == "websocket"

        if not is_upgrade:
            # 纯 HTTP 探活
            if request.path == HEALTHZ_PATH:
                return Response(200, "OK",
                                Headers({"Content-Type": "text/plain; charset=utf-8"}),
                                b"ok")
            return Response(404, "Not Found",
                            Headers({"Content-Type": "text/plain; charset=utf-8"}),
                            b"not found")

        # WS 升级：token 鉴权（Authorization: Bearer <token>）
        if self.svc.token:
            auth = request.headers.get("Authorization", "") or ""
            bearer = auth[7:] if auth.lower().startswith("bearer ") else ""
            if bearer.strip() != self.svc.token:
                return Response(401, "Unauthorized",
                                Headers({"Content-Type": "text/plain; charset=utf-8"}),
                                b"unauthorized")
        return None

    # ─── 连接处理 ────────────────────────────────────────

    async def _handle_connection(self, connection) -> None:
        try:
            # 单客户端占用检查
            if self._client is not None:
                logger.warning("[voice_service] 已有客户端 %s，拒绝第二个连接 %s",
                               self._client_id, getattr(connection, "remote_address", "?"))
                await connection.close(code=4090, reason="already occupied")
                return

            # 首帧必须是 hello
            try:
                first = await asyncio.wait_for(connection.recv(), timeout=10.0)
            except asyncio.TimeoutError:
                await connection.close(code=4091, reason="hello timeout")
                return

            frame = P.loads(first)
            if frame.get("type") != P.CMD_HELLO:
                await connection.close(code=4092, reason="first frame must be hello")
                return
            data = frame.get("data") or {}
            client_id = data.get("client_id") or frame.get("client_id") or "unknown"
            caps = data.get("caps") or {}

            self._client = connection
            self._client_id = client_id
            logger.info("[voice_service] 客户端接入: client_id=%s caps=%s",
                        client_id, caps)
            await self.send_frame(P.make_frame(P.EVT_WELCOME, {
                "ok": True,
                "my_wakewords": self._assistants,  # 服务端实际助手表（config 唯一源）
                "version": P.PROTOCOL_VERSION,
            }))

            # 主循环：等待消息；超过心跳窗口未收到任何帧视为失联
            while True:
                try:
                    raw = await asyncio.wait_for(
                        connection.recv(), timeout=self.svc.heartbeat_timeout_sec)
                except asyncio.TimeoutError:
                    logger.warning("[voice_service] 客户端 %s 心跳超时，断开", client_id)
                    break
                if raw is None:        # 连接关闭
                    break
                try:
                    frame = P.loads(raw)
                except Exception as exc:
                    logger.warning("[voice_service] 非法帧: %s", exc)
                    await self.send_frame(P.ack_for(0, False, "bad frame"))
                    continue
                await self._dispatch(frame)
        except Exception as exc:
            # 正常关闭（对端 close / bye）会抛 ConnectionClosed —— 不算异常
            if type(exc).__name__ == "ConnectionClosed":
                logger.debug("[voice_service] 连接正常关闭: %s", exc)
            else:
                logger.warning("[voice_service] 连接异常: %s", exc)
        finally:
            # 只清"当前登记连接"的状态；被拒/未登记的连接不得污染活动客户端
            if self._client is connection:
                logger.info("[voice_service] 客户端 %s 已断开", self._client_id or "?")
                self._client = None
                self._client_id = ""

    # ─── 帧路由 ──────────────────────────────────────────

    async def _dispatch(self, frame: Dict[str, Any]) -> None:
        type_ = frame.get("type")
        data = frame.get("data") or {}
        seq = frame.get("seq")
        if type_ == P.CMD_PING:
            await self.send_frame(P.make_frame(P.EVT_PONG, {}, seq=seq))
            return
        if type_ == P.CMD_BYE:
            await self.send_frame(P.ack_for(seq, True))
            await self._client.close(code=1000, reason="bye")
            return
        if type_ == P.CMD_SPEAK:
            # 播报（M3）：入队 → ack
            if self.playback is not None:
                await self.playback.enqueue_speak(
                    id=data.get("id", ""), wake=data.get("wake", ""),
                    kind=data.get("kind", P.SPEAK_FINAL),
                    segments=data.get("segments") or [],
                    turn_seq=int(data.get("turn_seq") or 0))
                logger.info("[voice_service] speak 入队: kind=%s wake=%s",
                            data.get("kind"), data.get("wake"))
            else:
                logger.info("[voice_service] 收到 speak（无 playback，仅记录）: %s",
                            str(data)[:120])
            await self.send_frame(P.ack_for(seq, True))
            return
        if type_ == P.CMD_CONTROL:
            action = (data or {}).get("action")
            if action == P.CTRL_CLOSE_WINDOW and self.playback is not None:
                await self.playback.close_window()
            elif action in (P.CTRL_DISMISS_REPLY, P.CTRL_RELOAD_KWS):
                logger.info("[voice_service] control %s 暂未启用（可选增强）", action)
            else:
                logger.info("[voice_service] control: %s", action)
            await self.send_frame(P.ack_for(seq, True))
            return
        if type_ == P.CMD_INTERRUPT:
            if self.playback is not None:
                await self.playback.interrupt()
            else:
                logger.info("[voice_service] 收到 interrupt（无 playback，仅记录）")
            await self.send_frame(P.ack_for(seq, True))
            return
        if type_ == P.CMD_SPEAKER_ALIAS:
            # 说话人身份绑定（agent 工具 → 服务端写 names.json 并热更新显示名）。
            # 成功/失败都带数据（失败时附 previous 等，便于 agent 向用户解释）。
            result = self._handle_speaker_alias(data)
            ok = bool(result.pop("ok", False))
            extra = {"seq": seq, "ok": ok}
            if ok:
                extra.update(result)
            else:
                extra["error"] = result.pop("error", None) or "speaker_alias 失败"
                extra.update(result)
            await self.send_frame(P.make_frame(P.EVT_ACK, extra, seq=seq))
            return
        if type_ == P.EVT_ACK:
            # 服务端通常不主动向客户端发命令；收到 ack 只记日志（未来 speak_done 也走 S→C）
            logger.debug("[voice_service] 客户端 ack: %s", data)
            return
        logger.warning("[voice_service] 未知帧类型: %s", type_)
        await self.send_frame(P.ack_for(seq, False, f"unknown type: {type_}"))

    def _handle_speaker_alias(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """说话人绑定/解绑（`speaker_alias` 帧）。

        返回给 ack 的 payload（`ok` 由调用方取走）：成功带 `spk_id/name/previous/
        also_bound` 等；失败只有 `ok=False, error`。绑定写入 names.json 并立即生效
        （下一句 `asr_result` 的标签就会用新名字），无需重启。
        """
        if self.speaker_admin is None:
            return {"ok": False, "error": "说话人识别未启用（voiceprint 未开启）"}
        action = str((data or {}).get("action") or "").strip().lower()
        spk_id = (data or {}).get("spk_id")
        if action == P.ALIAS_SET:
            result = self.speaker_admin.set_alias(
                spk_id, (data or {}).get("name") or "",
                overwrite=bool((data or {}).get("overwrite")))
        elif action == P.ALIAS_UNSET:
            result = self.speaker_admin.unset_alias(spk_id)
        else:
            result = {"ok": False, "error": f"未知 action: {action!r}"}
        if result.get("ok"):
            logger.info("[voice_service] speaker_alias %s → %s", action, result)
        else:
            logger.warning("[voice_service] speaker_alias %s 失败: %s",
                           action, result.get("error"))
        return result

    # ─── 上行事件（供 inbound 装配后调用）────────────────

    async def send_frame(self, frame: Dict[str, Any]) -> bool:
        """向当前客户端发一帧；无客户端时丢弃并返回 False。"""
        if self._client is None:
            logger.debug("[voice_service] 无客户端，丢弃上行帧: %s", frame.get("type"))
            return False
        try:
            await self._client.send(P.dumps(frame))
            return True
        except Exception as exc:
            logger.warning("[voice_service] 上行帧发送失败: %s", exc)
            return False

    async def broadcast_event(self, type_: str, data: Dict[str, Any]) -> bool:
        """上行事件（wake_word/asr_result/...），v1 仅单客户端。"""
        return await self.send_frame(P.make_frame(type_, data))

    def threadsafe_event(self, type_: str, data: Dict[str, Any]) -> bool:
        """线程安全上行：供 SDK/音频线程调用（内部调度到事件循环）。"""
        if self.loop is None or self._client is None:
            logger.debug("[voice_service] 无客户端/事件循环，丢弃事件 %s", type_)
            return False
        try:
            return asyncio.run_coroutine_threadsafe(
                self.broadcast_event(type_, data), self.loop).result()
        except Exception as exc:
            logger.warning("[voice_service] 线程安全上行失败: %s", exc)
            return False
