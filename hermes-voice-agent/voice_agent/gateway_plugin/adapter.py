"""
voice-platform 适配器：语音平台（sherpa 唤醒词 + VAD + 腾讯云 ASR/TTS）。

实现进度（分阶段）：
- 阶段 0（本文件当前）：插件注册、配置解析、生命周期骨架、工具披露接口；
  语音组件（唤醒词/ASR/TTS/播放）在 connect() 中装配，缺失时降级并告警。
- 阶段 1：inbound —— 唤醒词/对话窗口期 VAD → 腾讯云 ASR → MessageEvent。
- 阶段 2：outbound —— send() 情绪分段解析 → 全局播报队列（串行）→ TTS
  流式播放 → 通知音 → 对话窗口；post_api_request 钩子做中间轮播报。
- 阶段 3：多唤醒词独立会话（chat_id=wake:<名>）+ 会话 system prompt 绑定。

配置（~/.hermes/config.yaml）::

    platforms:
      voice:
        enabled: true
        extra:
          wakewords:                     # 唤醒词 → 会话映射
            小布: { session_id: "hermes-voice-小布", system_prompt: "..." }
            翻译助手: { session_id: "hermes-voice-翻译助手" }
          conversation_window_sec: 5.0   # 最终回复播完后的连续对话窗口期
          identity_prefix: true          # 播报前加 "我是{唤醒词}，" 前缀
          kws:                           # sherpa 模型路径（默认取项目 models/）
            model_dir: "models/sherpa-kws"
            model_name: "sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20"
          voiceprint: { enabled: false } # 说话人识别（可选）

腾讯云凭据走环境变量 VOICE_SecretId / VOICE_SecretKey / VOICE_AppId
（与现有 voice_agent 配置一致）。
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)

logger = logging.getLogger(__name__)

# 语音会话 chat_id 前缀：唤醒词 <名> → chat_id "wake:<名>"
WAKE_CHAT_PREFIX = "wake:"

# gateway 系统通知内容特征：voice 频道只播 LLM 回复，命中以下前缀的通知静默。
# - "📬 No home channel is set"：home-channel 提示（首次对话且未配置 home channel）
# - "♻️"/"♻"：Gateway online / restarted 生命周期广播（另有
#   platforms.voice.gateway_restart_notification=false 配置层抑制，此处双保险）
_SYSTEM_NOTICE_PREFIXES = (
    "📬 No home channel is set",
    "♻️",
    "♻",
)

# 项目根（voice_agent/gateway_plugin/ → 项目根）
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
NOTIFICATION_PATH = PROJECT_ROOT / "assets" / "notification.wav"

# ─── (情绪)文字 分段解析（与 VoiceApp 保持一致）────────────

VALID_EMOTIONS = {
    "neutral", "sad", "happy", "angry", "fear",
    "story", "poetry", "sajiao", "disgusted", "amaze",
    "exciting", "aojiao", "jieshuo",
}
EMOTION_PATTERN = re.compile(
    r"\((?:%s)\)" % "|".join(sorted(VALID_EMOTIONS)), re.IGNORECASE)


def parse_emotion_segments(response: str) -> List[tuple[str, str]]:
    """解析 (情绪)文字内容 格式，支持多个情绪标记分段（与 VoiceApp 同一语义）。"""
    response = response.strip().replace('（', '(').replace('）', ')')
    if not response:
        return []
    matches = list(EMOTION_PATTERN.finditer(response))
    if not matches:
        return [("", response)]
    segments: List[tuple[str, str]] = []
    if matches[0].start() > 0:
        head = response[:matches[0].start()].strip()
        if head:
            segments.append(("", head))
    for i, m in enumerate(matches):
        emotion = m.group(0)[1:-1].strip().lower()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(response)
        text = response[start:end].strip()
        if text:
            segments.append((emotion, text))
    return segments


# 模块级"当前语音适配器"（单实例假设）：供 post_api_request 全局钩子
# 把中间轮文本投递到语音播放队列。
_lock = threading.Lock()
_active_instance: Optional["VoiceAdapter"] = None


def _set_active_instance(adapter: Optional["VoiceAdapter"]) -> None:
    global _active_instance
    with _lock:
        _active_instance = adapter


def _get_active_instance() -> Optional["VoiceAdapter"]:
    with _lock:
        return _active_instance


# ─── 语音组件装配（阶段 1/2 使用，connect 中初始化）─────────────────────


class VoiceAdapter(BasePlatformAdapter):
    """语音平台适配器。"""

    # 语音平台的"消息"就是扬声器播报，不渲染 markdown 代码块
    supports_code_blocks: bool = False
    # 播放是异步后台任务，支持 turn 结束后的异步递送
    supports_async_delivery: bool = True

    # 语音输入来自本地物理麦克风（sounddevice），不是网络暴露的账号消息——
    # 能物理接触到设备的人即视为已授权（等同 HOMEASSISTANT/WEBHOOK 语义）。
    # 因此授权由"本地物理访问"这一可信上游完成，无需平台用户白名单；
    # 如需严格白名单，可另行配置 VOICE_ALLOWED_USERS env（见 register()）。
    @property
    def authorization_is_upstream(self) -> bool:
        return True

    def __init__(self, config: PlatformConfig):
        platform = Platform("voice")
        super().__init__(config=config, platform=platform)

        extra = config.extra or {}
        self._extra = extra
        self._wakewords: Dict[str, dict] = extra.get("wakewords") or {}
        self._conversation_window_sec = float(
            extra.get("conversation_window_sec", 5.0)
        )
        # 身份前缀：多唤醒词共用扬声器时，播报前加 "我是{唤醒词}，"
        self._identity_prefix = bool(extra.get("identity_prefix", True))

        # gateway 事件循环（connect 时记录，供 SDK 回调线程投递）
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # 语音组件（connect 时装配）
        self._frontend = None       # voice_frontend（唤醒词/VAD/麦克风）
        self._asr = None            # TencentCloudASREngine
        self._tts = None            # TencentCloudTTSEngine
        self._player = None         # AudioPlayer
        # 当前命中的唤醒词（_on_wake_word 记录，on_complete 时使用）
        self._current_wake: Optional[str] = None
        # 播报串行队列（阶段 2）：所有会话回复 + 中间轮钩子共用，单消费者
        self._playback_queue: Optional[asyncio.Queue] = None
        self._playback_task: Optional[asyncio.Task] = None

    # ─── 配置解析 ────────────────────────────────────────

    def _resolve_voice_cfg(self) -> Dict[str, Any]:
        """合并插件配置与 voice_agent 默认配置。

        优先 platforms.voice.extra；未配置项回退到 voice_agent 的默认/
        项目 config.yaml（复用现有 ASR/VAD/唤醒词等默认值）。
        """
        try:
            from voice_agent.config import load_config
            base = (load_config() or {}).get("voice", {})
        except Exception:
            base = {}
        # extra 顶层键直接覆盖 voice 配置的对应段
        merged = dict(base)
        for key, value in self._extra.items():
            if key in ("wakewords", "identity_prefix", "conversation_window_sec",
                       "kws", "asr", "vad", "tts", "voiceprint", "mic"):
                if isinstance(value, dict) and isinstance(merged.get(key), dict):
                    merged[key] = {**merged[key], **value}
                else:
                    merged[key] = value
        return merged

    # ─── 生命周期 ────────────────────────────────────────

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        """初始化语音组件（唤醒词 + VAD + 麦克风 + 腾讯云 ASR）并启动监听。

        inbound 链路：sherpa 唤醒词命中 → VAD 确认人声 → 腾讯云 ASR →
        on_complete（last_text）→ MessageEvent → gateway 会话。
        """
        self._loop = asyncio.get_running_loop()
        try:
            if not check_requirements():
                logger.warning("[voice] 前置检查未通过，连接失败")
                return False

            vcfg = self._resolve_voice_cfg()
            self._init_asr(vcfg)
            self._init_frontend(vcfg)
            self._init_playback(vcfg)
            _set_active_instance(self)
            self._frontend.start()
            logger.info("[voice] 语音平台已连接（唤醒词: %s）",
                        ", ".join(self._wakewords) or "(未配置)")
            return True
        except Exception as exc:
            logger.error("[voice] connect 失败: %s", exc)
            return False

    def _init_playback(self, vcfg: Dict[str, Any]) -> None:
        """装配腾讯云 TTS + 音频播放器，启动播报串行队列。"""
        from voice_agent.audio_player import AudioPlayer, AudioPlayerConfig
        from voice_agent.tts_engine import TencentCloudTTSEngine, TencentTTSConfig

        tts_cfg = vcfg.get("tts", {})
        if tts_cfg.get("enabled", True):
            self._tts = TencentCloudTTSEngine(TencentTTSConfig(
                secret_id=os.getenv("VOICE_SecretId", ""),
                secret_key=os.getenv("VOICE_SecretKey", ""),
                app_id=os.getenv("VOICE_AppId", ""),
                voice_type=tts_cfg.get("voice_type", 101001),
                codec=tts_cfg.get("codec", "pcm"),
                sample_rate=tts_cfg.get("sample_rate", 16000),
                speed=tts_cfg.get("speed", 0.0),
                volume=tts_cfg.get("volume", 0.0),
            ))
            self._player = AudioPlayer(AudioPlayerConfig())
            self._player.start()
            self._tts.on_audio_chunk = self._player.feed

        # 播报串行队列：所有会话回复 + 中间轮钩子共用，单消费者
        self._playback_queue = asyncio.Queue()
        self._playback_task = asyncio.create_task(self._playback_consumer())

    def _init_asr(self, vcfg: Dict[str, Any]) -> None:
        """装配腾讯云 ASR 引擎（凭据来自 env VOICE_*，与现有 voice_agent 一致）。"""
        from voice_agent.asr_engine import TencentASRConfig, TencentCloudASREngine

        asr_cfg = vcfg.get("asr", {})
        self._asr = TencentCloudASREngine(TencentASRConfig(
            secret_id=os.getenv("VOICE_SecretId", ""),
            secret_key=os.getenv("VOICE_SecretKey", ""),
            app_id=os.getenv("VOICE_AppId", ""),
            engine_model=asr_cfg.get("engine_model", "16k_zh"),
            needvad=asr_cfg.get("needvad", False),
            voice_format=asr_cfg.get("voice_format", 1),
        ))
        # 服务端 VAD 可能在用户句间停顿时提前回调 on_final，这里只认
        # on_complete（整段说完）→ last_text 兜底，与 VoiceApp 一致
        self._asr.on_final = None
        self._asr.on_complete = self._on_asr_complete
        self._asr.on_error = self._on_asr_error

    def _init_frontend(self, vcfg: Dict[str, Any]) -> None:
        """装配语音前端（唤醒词 + VAD + 麦克风 → ASR 桥接）。"""
        from voice_agent.voice_frontend import VoiceFrontend, VoiceFrontendConfig

        wake = vcfg.get("wake_word", {})
        vad = vcfg.get("vad", {})
        kws = vcfg.get("kws", {}) or {}
        mic = vcfg.get("mic", {}) or {}
        front_cfg = VoiceFrontendConfig(
            wake_word_enabled=wake.get("enabled", True),
            wake_word_keyword=next(iter(self._wakewords), wake.get("keyword", "赫尔墨斯")),
            wake_word_threshold=float(kws.get("threshold", wake.get("sensitivity", 0.25))),
            kws_model_dir=kws.get("model_dir", "models/sherpa-kws"),
            kws_model_name=kws.get("model_name",
                                   "sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20"),
            kws_encoder=kws.get("encoder", "encoder-epoch-13-avg-2-chunk-8-left-64.int8.onnx"),
            kws_decoder=kws.get("decoder", "decoder-epoch-13-avg-2-chunk-8-left-64.onnx"),
            kws_joiner=kws.get("joiner", "joiner-epoch-13-avg-2-chunk-8-left-64.int8.onnx"),
            kws_tokens=kws.get("tokens", "tokens.txt"),
            vad_mode=int(vad.get("mode", 3)),
            min_speech_ms=int(vad.get("min_speech_ms", 200)),
            vad_silence_threshold_ms=int(vad.get("silence_threshold_ms", 600)),
            vad_speech_confirm_frames=int(vad.get("speech_confirm_frames", 3)),
            wake_guard_sec=float(vad.get("wake_guard_sec", 2.5)),
            mic_sample_rate=int(mic.get("sample_rate", 0)),
            mic_device=mic.get("device"),
            conversation_window_sec=self._conversation_window_sec,
            on_wake_word=self._on_wake_word,
            on_interrupt_request=self._on_interrupt_request,
        )
        self._frontend = VoiceFrontend(front_cfg, self._asr)

    async def disconnect(self) -> None:
        """停止语音组件与监听。"""
        _set_active_instance(None)
        if self._frontend:
            try:
                self._frontend.stop()
            except Exception as exc:
                logger.debug("[voice] frontend stop: %s", exc)
        if self._playback_task:
            self._playback_task.cancel()
            self._playback_task = None
        if self._player:
            self._player.stop()
        logger.info("[voice] 语音平台已断开")

    # ─── outbound：gateway 回复 → 播报 ───────────────────

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """把 agent 回复播报给用户。

        解析 (情绪)文字 分段 → 投入全局播报队列（串行）→ TTS 流式播放 →
        通知音 → 进入对话窗口期。播放由后台消费者执行，send 快速返回。
        [FINISH]（播放控制工具结束标记）不播报，直接关闭对话窗口。
        """
        text = (content or "").strip()
        if not text or self._playback_queue is None:
            return SendResult(success=False, error="播报队列未就绪")

        # voice 频道只播 LLM 回复：gateway 系统通知（home-channel 提示、
        # online/restart 广播等）静默，不进入播报队列
        if text.startswith(_SYSTEM_NOTICE_PREFIXES):
            logger.info("[voice] 系统通知已静默: %s", text[:60])
            return SendResult(success=True)

        if text.upper().strip("[]") == "FINISH":
            logger.info("[voice] [FINISH] 关闭对话窗口")
            self._close_conversation_window()
            return SendResult(success=True)

        segments = parse_emotion_segments(text)
        if not segments:
            return SendResult(success=True)

        wake = chat_id[len(WAKE_CHAT_PREFIX):] \
            if chat_id.startswith(WAKE_CHAT_PREFIX) else chat_id
        if self._identity_prefix and wake:
            emo, first = segments[0]
            segments[0] = (emo, f"我是{wake}，{first}")
        await self._playback_queue.put((wake, segments, True))
        logger.info("[voice] send → %s: %s", wake, text[:60])
        return SendResult(success=True)

    async def _playback_consumer(self) -> None:
        """播报队列消费者：串行播放所有会话的回复（含中间轮钩子投递）。"""
        try:
            while True:
                wake, segments, is_final = await self._playback_queue.get()
                try:
                    await self._play_segments(wake, segments, is_final)
                except Exception as exc:
                    logger.warning("[voice] 播报失败: %s", exc)
        except asyncio.CancelledError:
            logger.info("[voice] 播报队列已停止")

    async def _play_segments(
        self, wake: str, segments: List[tuple[str, str]], is_final: bool
    ) -> None:
        """TTS 流式播放一段或多段情绪文本。

        is_final=True（最终回复）：播完播通知音并进入对话窗口期；
        is_final=False（中间轮）：只播报，不进对话窗口。
        """
        if not segments or self._tts is None or self._player is None:
            return
        from voice_agent.music_control import player_pause, player_resume

        logger.info("[voice] 🔊 播报(%s): %d 段, 首段 %s...",
                    "最终" if is_final else "中间", len(segments), segments[0][1][:30])
        player_pause(force=True)
        try:
            if self._frontend:
                self._frontend.set_tts_playing(True)
            self._tts.on_audio_chunk = self._player.feed
            for emotion, text in segments:
                if not text or not text.strip():
                    continue
                self._tts.start()
                if emotion:
                    self._tts.set_emotion(emotion)
                self._tts.synthesize(text)
                self._tts.complete()
                self._tts.wait(timeout=15)
            self._player.wait_for_drain(timeout=15.0)
            time.sleep(0.15)

            if is_final:
                self._play_asset_notification()
                if self._player:
                    self._player.wait_for_drain(timeout=5.0)
                    time.sleep(0.05)
                # 播完进入连续对话窗口期：VAD 直接听，无需再喊唤醒词
                if self._frontend:
                    self._frontend.enter_conversation_window()
        finally:
            if self._tts:
                self._tts.on_audio_chunk = self._player.feed
            if self._frontend:
                self._frontend.set_tts_playing(False)
            player_resume()

    def _play_asset_notification(self) -> None:
        """播放通知音（最终回复播完的提示）。"""
        if not NOTIFICATION_PATH.exists() or self._player is None:
            return
        try:
            import wave
            with wave.open(str(NOTIFICATION_PATH), "rb") as wf:
                data = wf.readframes(wf.getnframes())
            self._player.feed(data)
        except Exception as exc:
            logger.warning("[voice] 通知音播放失败: %s", exc)

    def _close_conversation_window(self) -> None:
        """关闭对话窗口（[FINISH] / 打断时），回到唤醒词监听。"""
        if self._frontend is None:
            return
        try:
            self._frontend._cancel_conversation_timer()
            from voice_agent.voice_frontend import VoiceState
            self._frontend._set_state(VoiceState.IDLE)
        except Exception as exc:
            logger.debug("[voice] 关闭对话窗口: %s", exc)

    # ─── inbound：语音 → gateway 会话 ────────────────────

    def _on_wake_word(self, name: str) -> None:
        """唤醒词命中（VoiceFrontend 音频处理线程）：记录命中的唤醒词。"""
        self._current_wake = name or next(iter(self._wakewords), "")
        logger.info("[voice] 唤醒词命中: %s → chat_id=%s",
                    self._current_wake, self.wake_chat_id(self._current_wake))

    def _on_interrupt_request(self) -> None:
        """VoiceFrontend 在唤醒词命中/打断时调用：物理打断 TTS 播放并清播报队列。

        agent 层的打断由 gateway busy_input_mode=interrupt 处理（同会话新消息
        取消进行中 turn）；这里负责物理输出层（TTS + 扬声器 + 队列）。
        """
        try:
            if self._tts:
                self._tts.interrupt()
            if self._player:
                self._player.clear()
            if self._loop is not None and self._playback_queue is not None:
                async def _clear():
                    while not self._playback_queue.empty():
                        try:
                            self._playback_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                asyncio.run_coroutine_threadsafe(_clear(), self._loop)
            logger.info("[voice] 打断：TTS 停止 + 播放队列已清空")
        except Exception as exc:
            logger.warning("[voice] 打断处理异常: %s", exc)

    def _on_asr_complete(self) -> None:
        """ASR 整段识别完成（SDK 回调线程）：文本 → MessageEvent → gateway 会话。"""
        try:
            text = (self._asr.last_text or "").strip()
            if not text:
                logger.info("[voice] ASR 结果为空，跳过")
                return
            wake = self._current_wake or next(iter(self._wakewords), "")
            logger.info("[voice] 用户说了: %s", text[:60])

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
            # handle_message 是 async，需在 gateway 事件循环线程执行
            if self._loop is not None:
                asyncio.run_coroutine_threadsafe(
                    self.handle_message(event), self._loop
                )
        except Exception as exc:
            logger.warning("[voice] ASR 完成处理异常: %s", exc)

    def _on_asr_error(self, msg: str) -> None:
        logger.error("[voice] ASR 错误: %s", msg)

    # ─── 工具披露（条件披露 mpd_* 等）───────────────────

    def toolsets_for_source(self, source) -> Optional[List[str]]:
        """按来源动态披露工具集；返回 None 则回退 platform_toolsets.<platform>。

        默认 None（由 gateway 配置 `platform_toolsets.voice` 控制，
        例如 `[voice, mpd]` 只对语音平台开放 mpd 工具）。
        """
        return None

    # ─── 工具 ────────────────────────────────────────────

    def wake_chat_id(self, wake_name: str) -> str:
        return f"{WAKE_CHAT_PREFIX}{wake_name}"

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """返回会话信息：语音会话均为 dm 型，chat 名 = 唤醒词名。"""
        wake_name = (
            chat_id[len(WAKE_CHAT_PREFIX):]
            if chat_id.startswith(WAKE_CHAT_PREFIX) else chat_id
        )
        return {"name": wake_name or chat_id, "type": "dm"}


# ─── 前置检查 / 配置校验 ─────────────────────────────────


def check_requirements() -> bool:
    """前置检查（check_fn 契约：返回 bool）。

    校验腾讯云凭据与 sherpa 模型目录是否存在；缺失时记录日志并返回 False。
    """
    missing = [k for k in ("VOICE_SecretId", "VOICE_SecretKey", "VOICE_AppId")
               if not os.getenv(k)]
    if missing:
        logger.warning("[voice] 缺少腾讯云凭据: %s", ", ".join(missing))
        return False
    model_dir = PROJECT_ROOT / "models" / "sherpa-kws"
    if not model_dir.is_dir():
        logger.warning("[voice] sherpa 模型目录不存在: %s", model_dir)
        return False
    return True


def validate_config(config) -> bool:
    """校验语音平台配置（契约：接收 PlatformConfig 或 dict，返回 bool）。

    True = 可启动；False = 缺少必要配置（如未配置任何唤醒词）。
    """
    try:
        if isinstance(config, dict):
            extra = config.get("extra") or {}
        elif config is not None:
            extra = getattr(config, "extra", None) or {}
        else:
            extra = {}
        wakewords = extra.get("wakewords") or {}
        if not wakewords:
            logger.warning("[voice] platforms.voice.extra.wakewords 未配置任何唤醒词")
            return False
        return True
    except Exception as exc:
        logger.warning("[voice] validate_config 异常: %s", exc)
        return False


def is_connected() -> bool:
    return _get_active_instance() is not None


# ─── 中间轮播报钩子（阶段 2 实现）────────────────────────


def on_post_api_request(**kwargs):
    """post_api_request 钩子：中间轮文字回复直接播报（替代已废除的 speak 工具）。

    - finish_reason == "tool_calls" 且 assistant_message.content 非空
      → 阶段文本 → 语音播报队列（无通知音、不进对话窗口）
    - finish_reason == "stop" → 忽略（最终回复由 gateway send() 播放，天然去重）
    - content == "[FINISH]" → 关闭对话窗口（不播报）
    """
    adapter = _get_active_instance()
    if adapter is None:
        return
    try:
        finish_reason = kwargs.get("finish_reason") or ""
        assistant = kwargs.get("assistant_message")
        if isinstance(assistant, dict):
            content = assistant.get("content") or ""
        else:
            content = getattr(assistant, "content", None) or ""
        text = content.strip()
        if not text:
            return
        if text.upper().strip("[]") == "FINISH":
            adapter._close_conversation_window()
            return
        if finish_reason == "stop":
            return  # 最终回复由 send() 播放
        segments = parse_emotion_segments(text)
        if not segments:
            return
        wake = adapter._current_wake or ""
        loop = adapter._loop
        if loop is not None and adapter._playback_queue is not None:
            asyncio.run_coroutine_threadsafe(
                adapter._playback_queue.put((wake, segments, False)), loop
            )
            logger.info("[voice] 中间轮播报: %s", text[:60])
    except Exception as exc:
        logger.warning("[voice] post_api_request 钩子异常: %s", exc)


# ─── 插件入口 ────────────────────────────────────────────


def register(ctx) -> None:
    """插件入口：注册语音平台 + 中间轮播报钩子 + 语音相关工具。"""
    # 注册本项目工具（mpd 音乐播放控制等）到全局 registry；
    # 是否对某个平台披露由 platform_toolsets.<platform> 控制
    # （例如 platforms.voice 对应 toolset "voice_agent" 时才会被语音
    # 会话的 agent 调用 —— 条件披露，不影响其他平台）。
    try:
        from voice_agent.mpd_tool import register_all as register_mpd_tools
        register_mpd_tools()
    except Exception as exc:
        logger.warning("[voice] mpd 工具注册失败（可忽略）: %s", exc)

    ctx.register_platform(
        name="voice",
        label="语音",
        adapter_factory=lambda cfg: VoiceAdapter(cfg),
        check_fn=check_requirements,
        validate_config=validate_config,
        required_env=["VOICE_SecretId", "VOICE_SecretKey", "VOICE_AppId"],
        install_hint="pip install -e .  # hermes-voice-agent",
        is_connected=is_connected,
        # 语音是本地物理麦克风输入，无网络暴露——授权由这两个 env 控制：
        #   VOICE_ALLOW_ALL_USERS=true           信任所有说话人（推荐）
        #   VOICE_ALLOWED_USERS=voice-user,...   或显式白名单
        allowed_users_env="VOICE_ALLOWED_USERS",
        allow_all_env="VOICE_ALLOW_ALL_USERS",
        emoji="🎙️",
        platform_hint=(
            "你通过语音与用户交流：回复要简洁（3 句话以内），最终回答用 "
            "(情绪)文字 格式（情绪可选 neutral/sad/happy/angry/...）；"
            "工具调用过程中的阶段性说明直接作为文字回复输出，会被立即播报；"
            "使用播放控制工具后直接返回 [FINISH]。"
        ),
    )
    ctx.register_hook("post_api_request", on_post_api_request)
    logger.info("voice-platform 已注册（语音平台 + post_api_request 播报钩子）")
