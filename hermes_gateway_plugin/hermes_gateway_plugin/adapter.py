"""
voice-platform 适配器（M4）：语音平台 = Voice Service 的 WS 客户端。

架构：Voice Service 独立承载音频与**会话节奏**（唤醒词/VAD/ASR/TTS/播放/
对话窗口/通知音/等待音）。本适配器只做「接线」：
- **唤醒词→助手映射归 Voice Service**（config wake_word.assistants，KWS
  命中返回 @助手名）——本适配器不再配置/上报 wakewords，只用上行事件里的
  助手名路由会话。
- **inbound**：Voice Service 上行事件 → gateway 会话。
  - `wake_word` → 记录当前活跃助手名（路由归属）；
  - `asr_result` → MessageEvent(chat_id="wake:<助手名>") → handle_message()。
- **outbound**：播报通道 = `post_api_request` 钩子（主通道，携带每轮 API 调用
  的上下文参数：session_id/finish_reason/assistant_message 等）+ `send()`
  （gateway delivery 兜底）。**当前为观察期**：on_post_api_request 会把完整
  参数摘要写入日志（前缀 `[voice] post_api_request:`），用于确认 memory
  审查轮（background review）与正常问答轮的字段差异 —— 待确认后据此过滤
  system 型消息（记忆更新/自我改进等），让语音只朗读"用户问题的直接回复"。

实现依赖 hermes gateway：必须在 hermes venv 中运行，且同 venv 需安装
voice_service（协议契约 voice_service.protocol）。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import socket
import threading
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from gateway.config import Platform
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)

logger = logging.getLogger(__name__)

# 语音会话 chat_id 前缀：助手名 <名> → chat_id "wake:<名>"（名来自 Voice Service 上行）
WAKE_CHAT_PREFIX = "wake:"

# gateway 系统通知内容特征：只在 send() 出现、不走 agent API → 静默不投递
_SYSTEM_NOTICE_PREFIXES = (
    "📬 No home channel is set",
    "♻️",
    "♻",
)

# 非朗读内容（固定格式的非对话性消息）—— 观察期仅登记，未接入判定；
# 待 on_post_api_request 参数分析后决定是否启用（见模块 docstring）。
_NON_READABLE_PATTERNS = (
    re.compile(r"^\s*💾\s*Self-improvement review:\s+\S[\s\S]*$", re.IGNORECASE),
    re.compile(
        r"^\s*💾\s+Skill\s+['\"].+?['\"]\s+"
        r"(?:created|updated|improved|patched)\.?\s*$", re.IGNORECASE),
    re.compile(r"^\s*⏳\s+Working\s+—\s+\d+\s+min(?:\s|$)", re.IGNORECASE),
    re.compile(
        r"^\s*\[Background process\s+\S+\s+"
        r"(?:finished with exit code|is still running~)[\s\S]*\]\s*$",
        re.IGNORECASE),
    re.compile(
        r"^\s*(?:✅|❌)\s+Hermes update\s+"
        r"(?:finished|failed|timed out)[\s\S]*$", re.IGNORECASE),
    re.compile(
        r"^\s*♻️?\s+Gateway\s+(?:restarted successfully|online\b)[\s\S]*$",
        re.IGNORECASE),
)


class VoiceAdapter(BasePlatformAdapter):
    """语音平台适配器（WS 客户端版）。"""

    # 语音平台的"消息"就是扬声器播报，不渲染 markdown 代码块
    supports_code_blocks: bool = False
    # 播放是 Voice Service 侧后台任务，支持 turn 结束后的异步递送
    supports_async_delivery: bool = True

    # 语音输入来自本地物理麦克风（Voice Service 采集），不是网络暴露的账号消息
    @property
    def authorization_is_upstream(self) -> bool:
        return True

    def __init__(self, config):
        super().__init__(config=config, platform=Platform("voice"))
        extra = config.extra or {}
        self._extra = extra
        # 唤醒词→助手映射归 Voice Service 维护（config wake_word.assistants），
        # 本适配器不再读取/校验 wakewords —— 只用上行事件携带的助手名路由会话。
        svc = extra.get("service") or {}
        self._service = {
            "url": svc.get("url", ""),
            "host": svc.get("host", "127.0.0.1"),
            "port": int(svc.get("port", 8765)),
            "token": svc.get("token", ""),
        }

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._client = None          # VoiceGatewayClient
        self._run_task: Optional[asyncio.Task] = None

        self._current_wake: Optional[str] = None
        # 每个会话最近一次 asr_result 的 turn_seq（speak 回填用）
        self._last_turn_by_wake: Dict[str, int] = {}
        # 钩子最近播报的最终回复（chat_id → 文本）：send() 去重用
        self._last_final_by_chat: Dict[str, str] = {}

    # ─── 配置解析 ────────────────────────────────────────

    def _service_url(self) -> str:
        svc = self._service
        if svc["url"]:
            return svc["url"]
        return f"ws://{svc['host']}:{svc['port']}"

    # ─── 生命周期 ────────────────────────────────────────

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        """连接 Voice Service：client 建立 WS + hello；成功后维持自动重连循环。"""
        self._loop = asyncio.get_running_loop()
        if self._client is not None and self._client.is_connected:
            logger.info("[voice] connect 幂等跳过（已有活跃连接）")
            return True
        try:
            from .client import VoiceGatewayClient

            self._client = VoiceGatewayClient(
                self._service_url(),
                token=self._service["token"],
                client_id=f"voice-gateway-{os.getpid()}",
                caps={"interim": True},
            )
            ok = await self._client.connect(handler=self._on_client_event)
            if not ok:
                logger.error("[voice] Voice Service 握手失败")
                return False
            self._run_task = asyncio.create_task(
                self._client.reconnect_forever(handler=self._on_client_event))
            _set_active(self)
            logger.info("[voice] voice 平台已连接 Voice Service（助手表见握手日志）")
            return True
        except Exception as exc:
            logger.error("[voice] connect 失败: %s", exc)
            return False

    async def disconnect(self) -> None:
        _set_active(None)
        if self._run_task is not None:
            self._run_task.cancel()
            try:
                await self._run_task
            except asyncio.CancelledError:
                pass
            self._run_task = None
        if self._client is not None:
            await self._client.close()
            self._client = None

    # ─── inbound：Voice Service 上行事件 → gateway 会话 ──

    def _on_client_event(self, frame: Dict[str, Any]) -> None:
        if self._loop is None:
            return
        type_ = frame.get("type")
        data = frame.get("data") or {}
        try:
            if type_ == "wake_word":
                self._current_wake = data.get("wake") or data.get("keyword") or ""
                logger.info("[voice] 助手唤醒: %s", self._current_wake)
            elif type_ == "asr_result":
                wake = data.get("wake") or self._current_wake or ""
                turn = data.get("turn_seq")
                if turn:
                    self._last_turn_by_wake[wake] = int(turn)
                text = (data.get("text") or "").strip()
                if not text:
                    return
                logger.info("[voice] 用户说了(%s): %s", wake, text[:60])
                asyncio.run_coroutine_threadsafe(
                    self._handle_asr_message(wake, text), self._loop)
            elif type_ == "speak_done":
                logger.info("[voice] 播报完成: %s", data)
            elif type_ == "error":
                logger.warning("[voice] Voice Service 事件错误: %s", data)
            else:
                logger.debug("[voice] 客户端事件: %s", type_)
        except Exception as exc:
            logger.warning("[voice] 事件处理异常: %s", exc)

    async def _handle_asr_message(self, wake: str, text: str) -> None:
        source = self.build_source(
            chat_id=self.wake_chat_id(wake),
            chat_name=wake,
            chat_type="dm",
            user_id="voice-user",
            user_name="用户",
        )
        event = MessageEvent(
            text=text,
            message_type=MessageType.TEXT,
            source=source,
            message_id=f"voice-{int(time.time() * 1000)}",
        )
        await self.handle_message(event)

    # ─── outbound：钩子主通道 + send() 兜底 ───────────────

    def _send_speak(self, wake: str, segments: List[tuple[str, str]],
                    is_final: bool) -> None:
        """线程安全：文本 → speak 帧。播报方式由 Voice Service 按会话状态决定。"""
        if self._client is None:
            return
        turn = self._last_turn_by_wake.get(wake, 0)
        self._client.send_speak(
            id=f"voice-{int(time.time() * 1000)}", wake=wake,
            kind="final" if is_final else "interim",
            segments=segments, turn_seq=turn)

    async def send(self, chat_id: str, content: str,
                   reply_to: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        """gateway delivery 兜底：最终回复主由钩子播报，这里只补漏并去重。"""
        text = (content or "").strip()
        if not text:
            return SendResult(success=True)
        if text.upper().strip("[]") == "FINISH":
            if self._client is not None:
                self._client.send_control("close_window")
            return SendResult(success=True)
        wake = chat_id[len(WAKE_CHAT_PREFIX):] \
            if chat_id.startswith(WAKE_CHAT_PREFIX) else chat_id
        if self._last_final_by_chat.get(chat_id) == text:
            logger.info("[voice] send 去重：最终回复已由钩子播报")
            return SendResult(success=True)
        if text.startswith(_SYSTEM_NOTICE_PREFIXES):
            logger.info("[voice] 系统通知静默: %.50s", text)
            return SendResult(success=True)
        segments = self._parse_segments(text)
        if not segments:
            return SendResult(success=True)
        # 兜底：钩子未播报的内容（工具轮文本/异常路径），按普通文本下发，
        # 是否朗读由 Voice Service 会话状态决定。
        self._send_speak(wake, segments, False)
        logger.info("[voice] send 兜底下发 → %s: %.60s", wake, text)
        return SendResult(success=True)

    # ─── 工具/辅助 ───────────────────────────────────────

    def _parse_segments(self, response: str) -> List[tuple[str, str]]:
        from voice_service.protocol import parse_emotion_segments
        return parse_emotion_segments(response)

    def toolsets_for_source(self, source) -> Optional[List[str]]:
        return None

    def wake_chat_id(self, wake_name: str) -> str:
        return f"{WAKE_CHAT_PREFIX}{wake_name}"

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        wake_name = (
            chat_id[len(WAKE_CHAT_PREFIX):]
            if chat_id.startswith(WAKE_CHAT_PREFIX) else chat_id)
        return {"name": wake_name or chat_id, "type": "dm"}


# ─── post_api_request 钩子（观察期：先记参数，再定过滤规则）─────────


def _summarize_kwargs(kwargs: Dict[str, Any]) -> str:
    """把钩子参数压缩成便于日志检索的摘要（避免整段 API 载荷刷屏）。"""
    parts: List[str] = []
    for key, value in kwargs.items():
        if key == "assistant_message":
            content = None
            if isinstance(value, dict):
                content = value.get("content")
            elif hasattr(value, "content"):
                content = getattr(value, "content")
            elif isinstance(value, str):
                content = value
            parts.append(f"assistant_message.content={str(content or '')[:100]!r}")
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            parts.append(f"{key}={str(value)[:100]!r}")
        elif isinstance(value, (list, tuple)):
            parts.append(f"{key}=<list[{len(value)}]>")
        elif isinstance(value, dict):
            parts.append(f"{key}=<dict{{{', '.join(list(value)[:8])}}}>")
        else:
            parts.append(f"{key}=<{type(value).__name__}>")
    return ", ".join(parts)


_active_lock = threading.Lock()
_active: Optional[VoiceAdapter] = None


def _set_active(adapter: Optional[VoiceAdapter]) -> None:
    global _active
    with _active_lock:
        _active = adapter


def is_connected() -> bool:
    with _active_lock:
        return _active is not None


def _get_active() -> Optional[VoiceAdapter]:
    with _active_lock:
        return _active


def on_post_api_request(**kwargs):
    """post_api_request 钩子（观察期 + 来源过滤）：只处理 voice 平台的回复。

    post_api_request 是 hermes 的全局 observer 钩子（每个会话的每次 LLM API
    请求都会触发），必须按 platform 过滤：非 voice 会话（cli/其它平台）直接
    return，避免 CLI 等会话的回复被语音朗读。

    播报语义：finish_reason=stop → final，tool_calls 带文字 → interim，
    [FINISH] → 关窗。待对比"正常问答轮"与"memory 审查轮"的字段差异后，
    再据此过滤 system 型消息（记忆更新/自我改进），让语音只朗读用户问题的
    直接回复。
    """
    if (kwargs.get("platform") or "") != "voice":
        return  # 非 voice 平台的 API 请求不播报
    logger.info("[voice] post_api_request 参数: %s", _summarize_kwargs(kwargs))
    adapter = _get_active()
    if adapter is None or adapter._client is None:
        return
    try:
        finish_reason = kwargs.get("finish_reason") or ""
        assistant = kwargs.get("assistant_message")
        if isinstance(assistant, dict):
            content = assistant.get("content") or ""
        else:
            content = getattr(assistant, "content", None) or ""
        text = (content or "").strip()
        if not text:
            return
        if text.upper().strip("[]") == "FINISH":
            adapter._client.send_control("close_window")
            return
        wake = adapter._current_wake or ""
        is_final = finish_reason == "stop"
        segments = adapter._parse_segments(text)
        if not segments:
            return
        if is_final:
            adapter._last_final_by_chat[adapter.wake_chat_id(wake)] = text
        adapter._send_speak(wake, segments, is_final)
        logger.info("[voice] 钩子播报(%s): %.60s",
                    "最终" if is_final else "中间", text)
    except Exception as exc:
        logger.warning("[voice] post_api_request 钩子异常: %s", exc)


# ─── 前置检查 / 校验 ─────────────────────────────────────


def check_requirements() -> bool:
    """前置检查：Voice Service 可达（凭据/模型归 Voice Service 自身）。"""
    inst = _get_active()
    if inst is None:
        return True  # 平台未建实例时放行；连接失败在 connect() 体现
    try:
        parsed = urlparse(inst._service_url())
        host, port = parsed.hostname or "127.0.0.1", parsed.port or 8765
        with socket.create_connection((host, int(port)), timeout=3):
            return True
    except Exception as exc:
        logger.warning("[voice] Voice Service 探测失败: %s", exc)
        return False


def validate_config(config) -> bool:
    """校验平台配置：voice 平台只需 WS 连接参数（唤醒词映射归 Voice Service）。

    不再要求 wakewords：接入参数（url/token）都在 extra.service，缺省时用
    默认 127.0.0.1:8765 —— 因此本校验始终放行，连接成败在 connect() 体现。
    """
    return True


def register(ctx) -> None:
    """插件入口：注册 voice 平台 + post_api_request 钩子（观察期）。"""
    ctx.register_platform(
        name="voice",
        label="语音",
        adapter_factory=lambda cfg: VoiceAdapter(cfg),
        check_fn=check_requirements,
        validate_config=validate_config,
        required_env=[],            # 凭据/模型归 Voice Service
        install_hint="pip install -e ./hermes_gateway_plugin ./voice_service",
        is_connected=is_connected,
        # 语音是本地物理麦克风输入，无网络暴露——授权由这两个 env 控制
        allowed_users_env="VOICE_ALLOWED_USERS",
        allow_all_env="VOICE_ALLOW_ALL_USERS",
        emoji="🎙️",
        platform_hint=(
            '【语音交互说明】\n'
            '1. 用户的输入来自语音识别（ASR），可能存在同音字、漏字、多字等错误。\n'
            '   如果问题听起来不合逻辑，结合上下文做合理推断，而不是逐字照搬。\n'
            '2. 回答要简洁，控制在 3 句话以内。\n'
            '3. 回答中自然融入确认——不是生硬复述，而是把确认编织在回答里。\n'
            '   如果确实听清了，不需要额外确认。\n'
            '4. 如果实在听不懂，直接表示没听清。\n'
            '\n'
            '【说话人标识说明】\n'
            '发送给你的每条用户消息会以 [说话人身份] 前缀标注这句话是谁说的，\n'
            '例如「[爸爸] 今天天气怎么样？」。说话人身份用于帮助理解对话上下文，\n'
            '回答时不需要复述说话人。\n'
            '\n'
            '【语音播报规则】\n'
            '你的每条文字回复都会被系统实时语音播报给用户（无需调用任何播报工具）。\n'
            '- 文字格式回复可以带情绪标识：格式为 (情绪)你要说的话。\n'
            '- 可用情绪：neutral sad happy angry fear story poetry sajiao disgusted '
            'amaze exciting aojiao jieshuo。\n'
            '- 播放控制类工具（若可用）执行后直接返回 [FINISH]。'
        ),
    )
    ctx.register_hook("post_api_request", on_post_api_request)
    logger.info("voice-platform 已注册（钩子观察期：记录 post_api_request 参数）")
