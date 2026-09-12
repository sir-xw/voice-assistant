"""
Voice Gateway WS 客户端（可独立测试，不依赖 hermes）。

职责：连接 Voice Service 的 WS 接入点，完成 hello/welcome 握手、心跳，
并把帧编解码与线程安全的发送封装好；业务处理（MessageEvent / speak 帧）
由上层 adapter 通过回调注入。

协议契约复用 voice_service.protocol —— 安装本包时需同 venv 安装 voice_service。
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

from voice_service import protocol as P

logger = logging.getLogger("hermes_gateway_plugin.client")

FrameHandler = Callable[[Dict[str, Any]], None]  # 收到上行事件帧（线程安全：在 asyncio 线程回调）


class VoiceGatewayClient:
    """Voice Service WS 客户端。"""

    def __init__(
        self,
        url: str,
        *,
        token: str = "",
        client_id: str = "voice-gateway-1",
        caps: Optional[Dict[str, Any]] = None,
        ping_interval_sec: float = 20.0,
    ):
        self.url = url
        self.token = token
        self.client_id = client_id
        self.caps = caps or {}
        self.ping_interval_sec = ping_interval_sec

        self._ws: Optional[Any] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        # 线程安全发送：外部线程（post_api_request 钩子等）只放队列
        self._out_queue: asyncio.Queue = asyncio.Queue()
        self._writer_task: Optional[asyncio.Task] = None
        self._ping_task: Optional[asyncio.Task] = None
        self._on_event: Optional[FrameHandler] = None
        self._closing = False
        # 请求/应答：seq → (Event, box)。工具 handler 在 agent 线程阻塞等 ack
        self._seq = itertools.count(1)
        self._ack_lock = threading.Lock()
        self._ack_waiters: Dict[int, Tuple[threading.Event, Dict[str, Any]]] = {}

    # ─── 生命周期 ────────────────────────────────────────

    @property
    def is_connected(self) -> bool:
        """是否保持 OPEN 连接（供上层幂等判断，避免 connect 重入撞单客户端）。"""
        ws = self._ws
        if ws is None or self._closing:
            return False
        try:
            return getattr(ws, "state", None) is not None and \
                ws.state.name == "OPEN"
        except Exception:
            return False

    async def connect(self, handler: Optional[FrameHandler] = None) -> bool:
        """连接 + hello 握手。**幂等**：已有 OPEN 连接时直接返回 True。

        （reconnect_forever 每轮都会调用本方法；只有断线后才真正重连。）
        """
        import websockets

        if self.is_connected:
            logger.debug("[voice client] 已连接，跳过重复 connect")
            return True
        self._loop = asyncio.get_running_loop()
        self._on_event = handler
        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        self._ws = await websockets.connect(
            self.url, additional_headers=headers,
            ping_interval=None,          # 心跳走应用层 ping/pong
            open_timeout=10,
        )
        # hello：认证 + 上报能力。唤醒词→助手映射归 Voice Service 维护，
        # 客户端不再上报期望词表；服务端 welcome 回报实际助手表（my_wakewords）。
        await self._ws.send(P.dumps(P.make_frame(P.CMD_HELLO, {
            "client_id": self.client_id,
            "caps": self.caps,
        }, client_id=self.client_id)))
        welcome_raw = await asyncio.wait_for(self._ws.recv(), timeout=10)
        welcome = P.loads(welcome_raw)
        if welcome.get("type") != P.EVT_WELCOME or not (welcome.get("data") or {}).get("ok"):
            logger.error("[voice client] hello 被拒绝: %s", welcome_raw[:200])
            await self.close()
            return False
        logger.info("[voice client] 已连接 %s（服务端助手表: %s）",
                    self.url, (welcome.get("data") or {}).get("my_wakewords"))
        self._writer_task = asyncio.create_task(self._writer_loop())
        self._ping_task = asyncio.create_task(self._ping_loop())
        return True

    async def close(self) -> None:
        self._closing = True
        self._fail_pending("连接已关闭")
        for t in (self._writer_task, self._ping_task):
            if t is not None:
                t.cancel()
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

    # ─── 主接收循环 ──────────────────────────────────────

    async def run(self) -> None:
        """接收循环：阻塞直到断开。帧经 P.loads 后回调 _on_event。"""
        assert self._ws is not None, "先 connect()"
        try:
            async for raw in self._ws:
                if raw is None:
                    break
                try:
                    frame = P.loads(raw)
                except Exception as exc:
                    logger.warning("[voice client] 非法帧: %s", exc)
                    continue
                type_ = frame.get("type")
                if type_ == P.EVT_PONG:
                    logger.debug("[voice client] pong")
                    continue
                if type_ == P.EVT_ACK:
                    if self._resolve_ack(frame):
                        continue
                    logger.debug("[voice client] ack（无等待者）: %s", frame.get("data"))
                    continue
                if self._on_event is not None:
                    try:
                        self._on_event(frame)
                    except Exception:
                        logger.exception("[voice client] 事件回调异常")
                else:
                    logger.info("[voice client] 事件(未处理): %s", str(frame)[:160])
        finally:
            # 断线：唤醒所有等 ack 的工具调用，避免它们干等到超时
            self._fail_pending("连接断开")

    async def reconnect_forever(self, handler: Optional[FrameHandler] = None,
                                *, max_backoff_sec: float = 30.0) -> None:
        """断线自动重连主循环（指数退避 1s→30s）。"""
        backoff = 1.0
        while True:
            try:
                ok = await self.connect(handler)
                if ok:
                    backoff = 1.0
                    await self.run()
                logger.warning("[voice client] 连接断开，%.0fs 后重连", backoff)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("[voice client] 连接失败: %s（%.0fs 后重试）", exc, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, max_backoff_sec)

    # ─── 发送（线程安全） ────────────────────────────────

    def send_frame(self, frame: Dict[str, Any]) -> None:
        """任意线程可调：帧入队，由事件循环 writer 发送。"""
        if self._loop is None or self._ws is None:
            logger.warning("[voice client] 未连接，帧丢弃: %s", frame.get("type"))
            return
        try:
            asyncio.run_coroutine_threadsafe(
                self._out_queue.put(frame), self._loop)
        except Exception as exc:
            logger.warning("[voice client] 发送入队失败: %s", exc)

    def send_speak(self, *, id: str, wake: str, kind: str,
                   segments: List[P.EmotionSegment], turn_seq: int) -> None:
        """下发待朗读文本（speak 帧）。"""
        self.send_frame(P.make_frame(P.CMD_SPEAK, {
            "id": id, "wake": wake, "kind": kind,
            "segments": segments, "turn_seq": turn_seq,
        }, client_id=self.client_id))

    def send_control(self, action: str, **extra) -> None:
        self.send_frame(P.make_frame(P.CMD_CONTROL,
                                     {"action": action, **extra},
                                     client_id=self.client_id))

    # ─── 请求/应答（线程安全，供 agent 工具 handler 调用）──

    def request(self, frame_type: str, data: Dict[str, Any], *,
                timeout: float = 5.0) -> Optional[Dict[str, Any]]:
        """发一帧并等它的 ``ack``，返回 ``ack.data``。

        **只能在非事件循环线程调用**（agent 工具线程）：内部用 ``threading.Event``
        阻塞等待，ack 由 ``run()``（事件循环）解析后唤醒。超时/未连接/断线返回
        ``None``（调用方按失败处理）。
        """
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is not None and running is self._loop:
            logger.error("[voice client] request() 不能在客户端事件循环线程内调用")
            return None
        if not self.is_connected:
            logger.warning("[voice client] 未连接，无法发送 %s", frame_type)
            return None

        seq = next(self._seq)
        ev = threading.Event()
        box: Dict[str, Any] = {}
        with self._ack_lock:
            self._ack_waiters[seq] = (ev, box)
        try:
            self.send_frame(P.make_frame(frame_type, data, seq=seq,
                                         client_id=self.client_id))
            if not ev.wait(timeout):
                logger.warning("[voice client] %s 等待 ack 超时（seq=%d, %.1fs）",
                               frame_type, seq, timeout)
                return None
            return box.get("data")
        finally:
            with self._ack_lock:
                self._ack_waiters.pop(seq, None)

    def _resolve_ack(self, frame: Dict[str, Any]) -> bool:
        """把 ack 交给对应等待者；没有等待者返回 False（如 speak 的 ack）。"""
        data = frame.get("data") or {}
        seq = data.get("seq")
        if not isinstance(seq, int):
            return False
        with self._ack_lock:
            waiter = self._ack_waiters.get(seq)
        if waiter is None:
            return False
        ev, box = waiter
        box["data"] = data
        ev.set()
        return True

    def _fail_pending(self, reason: str) -> None:
        """唤醒所有等待者并给出失败原因（断线/关闭）。"""
        with self._ack_lock:
            waiters = list(self._ack_waiters.items())
            self._ack_waiters.clear()
        for seq, (ev, box) in waiters:
            box["data"] = {"seq": seq, "ok": False, "error": reason}
            ev.set()

    # ─── 内部任务 ────────────────────────────────────────

    async def _writer_loop(self) -> None:
        """出站队列 → WS 发送。

        断线瞬间 send() 会抛 ConnectionClosed —— 正常路径（run() 退出 →
        reconnect_forever 重连后会重建本任务），这里捕获后静默退出，
        避免留下 "Task exception was never retrieved" 脏日志。
        """
        try:
            while True:
                frame = await self._out_queue.get()
                if self._ws is None:
                    continue
                await self._ws.send(P.dumps(frame))
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            # 连接关闭类：正常抖动，重连后重建；其它异常记 warning
            if type(exc).__name__.startswith("ConnectionClosed"):
                logger.debug("[voice client] writer 随连接关闭退出: %s", exc)
            else:
                logger.warning("[voice client] writer 异常退出: %s", exc)

    async def _ping_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self.ping_interval_sec)
                self.send_frame(P.make_frame(P.CMD_PING, {}, client_id=self.client_id))
        except asyncio.CancelledError:
            pass
