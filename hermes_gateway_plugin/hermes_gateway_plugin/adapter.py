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
- **工具**：`voice_speaker_bind`（工具集 `voice_speaker`）把当前语音说话人的声纹
  编号绑定到名字。agent 只拿到文本前缀；插件从 `asr_result.data.speakers` 记下
  每个会话最近一轮的说话人，工具调用时经 WS `speaker_alias` 交给 Voice Service
  写 names.json（唯一映射在服务侧），ack 回结果。

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

# 说话人绑定工具的取数窗口：最近一轮语音多久内算"当前说话人"；显式指定编号时
# 允许回溯的最近轮次范围（防止模型拿很久以前的编号来绑定）
SPEAKER_CURRENT_MAX_AGE_SEC = 120.0
SPEAKER_RECENT_MAX_AGE_SEC = 300.0
SPEAKER_RECENT_KEEP = 8
# 绑定请求等 ack 的超时（服务端只写一个小 json，1s 足够；留足抖动余量）
SPEAKER_BIND_TIMEOUT_SEC = 5.0

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
        # 每个会话（wake）最近一轮的说话人：asr_result.speakers + 时间戳 + 最近编号
        # （voice_speaker_bind 工具据此确定"当前说话人"）
        self._last_utt_by_wake: Dict[str, Dict[str, Any]] = {}
        # 钩子最近播报的最终回复（chat_id → 文本）：send() 去重用
        self._last_final_by_chat: Dict[str, str] = {}
        # session_id → 助手名（惰性学习）：回复归属用，避免被后续唤醒抢占
        self._session_wake: Dict[str, str] = {}

    # ─── 配置解析 ────────────────────────────────────────

    def _service_url(self) -> str:
        svc = self._service
        if svc["url"]:
            return svc["url"]
        return f"ws://{svc['host']}:{svc['port']}"

    def _wake_for_hook(self, kwargs: Dict[str, Any]) -> str:
        """把钩子回复归属到对应助手（按 session_id 惰性学习）。

        `post_api_request` 只给 session_id；某会话**首次**出现时，当前活跃助手就是
        它的归属方，记下映射；此后该会话的中间轮/最终回复都归它所有。这样用户在
        助手 A 思考/调工具期间唤醒助手 B 后，A 迟到的回复仍标记为 A（Voice Service
        据此丢弃 A 的 interim、给 A 的 final 加「我是A，」前缀），不会被错记到 B。
        """
        cur = self._current_wake or ""
        sid = str(kwargs.get("session_id") or "")
        if not sid:
            return cur
        wake = self._session_wake.get(sid)
        if wake:
            return wake
        if cur:
            self._session_wake[sid] = cur
        return cur

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
                self._remember_speakers(wake, data.get("speakers") or [],
                                        int(turn or 0))
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

    # ─── 说话人跟踪（voice_speaker_bind 工具用）────────────

    def _remember_speakers(self, wake: str, speakers: List[Dict[str, Any]],
                           turn: int = 0) -> None:
        """记下某会话（wake）本轮说话人：当前轮 + 最近编号（供显式指定时校验）。"""
        if not wake:
            return
        now = time.time()
        prev = self._last_utt_by_wake.get(wake) or {}
        recent = [(ts, sid) for ts, sid in (prev.get("recent") or [])
                  if now - ts <= SPEAKER_RECENT_MAX_AGE_SEC]
        for spk in speakers:
            sid = str((spk or {}).get("spk_id") or "")
            if sid and sid not in [s for _, s in recent]:
                recent.append((now, sid))
        self._last_utt_by_wake[wake] = {
            "turn_seq": turn, "speakers": list(speakers), "ts": now,
            "recent": recent[-SPEAKER_RECENT_KEEP:],
        }

    def _current_speakers(self, session_id: str) -> tuple[List[Dict[str, Any]], str, str]:
        """当前会话最近一轮的说话人。

        返回 ``(speakers, wake, error)``：``error`` 非空表示拿不到当前说话人
        （不是语音会话 / 没有语音记录 / 太旧 / 没识别出编号）。
        会话必须是**语音会话**（`_session_wake` 只对 platform=voice 的回调学习），
        因此 CLI 等会话调用本工具会在这里被挡住。
        """
        sid = str(session_id or "")
        wake = self._session_wake.get(sid) or ""
        if not wake:
            return [], "", "当前会话不是语音会话（voice_speaker_bind 只用于语音输入）"
        utt = self._last_utt_by_wake.get(wake)
        if not utt:
            return [], wake, f"还没有 {wake} 会话的语音记录"
        if time.time() - float(utt.get("ts") or 0) > SPEAKER_CURRENT_MAX_AGE_SEC:
            return [], wake, "最近一轮语音已过去较久，无法确定当前说话人，请让用户再说一次"
        speakers = [s for s in (utt.get("speakers") or []) if s.get("spk_id")]
        if not speakers:
            return [], wake, "这一轮没有识别到说话人编号（声纹未启用或未注册）"
        return speakers, wake, ""

    def _recent_speaker_ids(self, wake: str) -> List[str]:
        """最近若干轮出现过的编号（显式指定 spk_id 时的合法集合）。"""
        utt = self._last_utt_by_wake.get(wake) or {}
        now = time.time()
        return [sid for ts, sid in (utt.get("recent") or [])
                if now - ts <= SPEAKER_RECENT_MAX_AGE_SEC]

    def bind_speaker(self, args: Dict[str, Any], session_id: str = "",
                     **kwargs: Any) -> str:
        """`voice_speaker_bind` 工具：把当前说话人的声纹编号绑定到名字。

        编号默认取"当前会话最近一轮的说话人"；一轮里有多个说话人时必须由模型显式
        指定 `spk_id`（消息前缀里的编号）。写请求经 WS 交给 Voice Service（它持有
        唯一映射与声纹库），成功返回该编号与名字。
        """
        from tools.registry import tool_error, tool_result

        from voice_service import protocol as P

        name = str(args.get("name") or "").strip()
        if not name:
            return tool_error("name 不能为空：需要用户明确说出的身份名")

        speakers, wake, err = self._current_speakers(session_id)
        if err:
            return tool_error(err)

        want = str(args.get("spk_id") or "").strip()
        if want:
            spk_id = P.normalize_spk_id(want)
            if not spk_id:
                return tool_error(
                    f"spk_id 非法: {want!r}（应填消息前缀里的编号，如 101）")
            known = self._recent_speaker_ids(wake)
            if known and spk_id not in known:
                return tool_error(
                    f"编号 {want} 不属于最近这轮说话人（本轮: "
                    f"{', '.join(P.speaker_number(s) for s in known)}）；"
                    f"请用消息前缀里的编号")
        else:
            ids = [str(s.get("spk_id")) for s in speakers]
            if len(ids) != 1:
                shown = "、".join(f"{s.get('label') or s.get('spk_id')}"
                                  for s in speakers)
                return tool_error(
                    f"这一轮有多个说话人（{shown}），请带 spk_id 指定给谁绑定")
            spk_id = ids[0]

        if self._client is None:
            return tool_error("语音服务未连接，绑定未完成")
        ack = self._client.request(P.CMD_SPEAKER_ALIAS, {
            "action": P.ALIAS_SET,
            "spk_id": spk_id,
            "name": name,
            "overwrite": bool(args.get("overwrite")),
        }, timeout=SPEAKER_BIND_TIMEOUT_SEC)
        if ack is None:
            return tool_error("语音服务未响应（超时或连接断开），绑定未完成")
        if not ack.get("ok"):
            return tool_error(str(ack.get("error") or "绑定失败"),
                              spk_id=spk_id, name=name)
        label = f"{ack.get('name') or name} (ID: {P.speaker_number(spk_id)})"
        logger.info("[voice] 说话人绑定成功: %s → %s（previous=%s）",
                    spk_id, ack.get("name") or name, ack.get("previous"))
        return tool_result(
            ok=True, spk_id=spk_id, name=ack.get("name") or name, label=label,
            previous=ack.get("previous"), unchanged=bool(ack.get("unchanged")),
            also_bound=ack.get("also_bound") or [],
            note=f"以后该说话人的消息前缀会显示为 [{label}]",
        )

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
        wake = adapter._wake_for_hook(kwargs)
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


# ─── 说话人身份绑定工具（agent → WS speaker_alias → Voice Service）───

VOICE_SPEAKER_BIND_SCHEMA = {
    "name": "voice_speaker_bind",
    "description": (
        "把当前语音说话人的声纹编号绑定到一个名字（记住「这个声音是谁」）。"
        "只用于语音对话：仅当用户亲口说明了自己的身份后才调用，不要猜测。"
        "绑定成功后该说话人的消息前缀会从「未知 (ID: 101)」变成「辰辰 (ID: 101)」，"
        "此后不必再问同一个人是谁。若该编号已经绑定为别的名字，"
        "需要先向用户确认，再用 overwrite=true 更正（例如用户说“你认错了，我是辰辰”）。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "用户说出的身份名，如 辰辰、淘淘、爸爸",
            },
            "spk_id": {
                "type": "string",
                "description": (
                    "可选：说话人编号（消息前缀里 ID: 后面的数字，如 101）。"
                    "默认绑定当前说话人的编号；一轮里有多个说话人时必须指定。"
                ),
            },
            "overwrite": {
                "type": "boolean",
                "description": (
                    "该编号已绑定为别的名字时是否覆盖。仅在用户明确更正身份时置 true。"
                ),
            },
        },
        "required": ["name"],
    },
}

# 工具所属工具集（hermes 配置 platform_toolsets.voice 里需列出该名，见 README）
VOICE_TOOLSET = "voice_speaker"


def _handle_voice_speaker_bind(args: Dict[str, Any], **kwargs: Any) -> str:
    """工具入口：按 session_id 定位到当前语会话，交给活跃适配器处理。"""
    from tools.registry import tool_error

    adapter = _get_active()
    if adapter is None:
        return tool_error("语音平台未连接，无法绑定说话人身份")
    return adapter.bind_speaker(
        args, session_id=str(kwargs.get("session_id") or ""))


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
            '发送给你的每条用户消息会以 [说话人名字 (ID: 编号)] 前缀标注这句话是谁说的，\n'
            '例如「[爸爸 (ID: 100)] 今天天气怎么样？」；没识别出的说话人名字显示为「未知」，\n'
            '如「[未知 (ID: 101)] 你好」。编号是该说话人在声纹库里的固定 id，同一个人跨轮次\n'
            '编号不变，可据此区分不同说话人。说话人身份用于帮助理解对话上下文，\n'
            '回答时不需要复述说话人。\n'
            '\n'
            '【说话人身份确认】\n'
            '- 前缀已是「[辰辰 (ID: 100)]」这类具体名字 → 说明已认出是谁，直接用，不要再问。\n'
            '- 前缀是「[未知 (ID: 101)]」且这次任务需要知道对方身份（例如"查我的课表"，\n'
            '  而你手头有多人的数据）→ 先直接问清楚，例如"你是辰辰还是淘淘？"，不要替对方假定。\n'
            '- 对方明确说明身份后，调用 voice_speaker_bind(name="辰辰") 记住（默认绑定当前\n'
            '  说话人的编号），然后接着完成刚才的请求。只有对方亲口说明时才绑定，不要猜。\n'
            '- 声纹识别偏严：同一个人可能拿到新的编号（又会显示为「未知 (ID: 10x)」）。\n'
            '  用户说自己是某人时，即使那个人已有别的编号，也照样绑定（允许一人多编号）。\n'
            '- 若发现认错人（用户说"你认错了，我是辰辰"）→ 用\n'
            '  voice_speaker_bind(name="辰辰", overwrite=true) 更正。\n'
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
    # 说话人身份绑定工具（工具集 voice_speaker，需在 hermes config 的
    # platform_toolsets.voice 里列出该工具集才会进入 agent 的工具表）
    ctx.register_tool(
        name=VOICE_SPEAKER_BIND_SCHEMA["name"],
        toolset=VOICE_TOOLSET,
        schema=VOICE_SPEAKER_BIND_SCHEMA,
        handler=_handle_voice_speaker_bind,
        check_fn=is_connected,
        description=VOICE_SPEAKER_BIND_SCHEMA["description"],
        emoji="🗣️",
    )
    logger.info("voice-platform 已注册（含工具 %s / 工具集 %s）",
                VOICE_SPEAKER_BIND_SCHEMA["name"], VOICE_TOOLSET)
