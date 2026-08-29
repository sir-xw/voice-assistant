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

# 语音配置与路径约定（profile 目录，无 gateway 依赖的公共模块）
from ..profile_config import (
    load_voice_config,
    models_dir,
    resolve_kws_model_dir,
    resolve_voiceprint_lib_dir,
    voice_config_path,
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
PROMPT_PATH = PROJECT_ROOT / "assets" / "prompt.wav"
FAREWELL_PATH = PROJECT_ROOT / "assets" / "farewell.wav"
WAIT_TONE_PATH = PROJECT_ROOT / "assets" / "wait_cue_4_scale.wav"

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
        # 钩子最近播报的最终回复（chat_id → 文本）：send() 去重用
        self._last_final_by_chat: Dict[str, str] = {}

        # 等待回复提示音：ASR 结束后、LLM 回复/TTS 开始前循环播放
        # （提示用户当前不再收听）。_wait_tone_waiting 为等待计数。
        self._wait_tone_waiting = 0
        self._wait_tone_task: Optional[asyncio.Task] = None

        # 说话人识别（voiceprint 启用时非 None）：
        # _vp_round 收集本轮每个句子的 (显示名, 文本)，on_complete 时拼成
        # "[说话人] 内容" 消息发给 LLM；_vp_id_cache 缓存腾讯云 speaker_id →
        # (库id, 相似度)，同 speaker_id 只做一次声纹识别（低延迟）
        self.voiceprint = None
        self._vp_round: List[tuple[str, str]] = []
        self._vp_id_cache: dict = {}
        self._vp_auto_register = True
        self._vp_min_register_sec = 1.5
        self._vp_use_cache = True

    # ─── 配置解析 ────────────────────────────────────────

    def _resolve_voice_cfg(self) -> Dict[str, Any]:
        """合并语音配置：profile 的 voice-agent.yaml + platforms.voice.extra。

        优先级（高→低）：
          1. platforms.voice.extra（gateway 配置，按段覆盖）
          2. profile 语音配置（~/.hermes/voice-agent.yaml，与 voice_agent
             config.yaml 的 voice 段同构）
        不再读取插件源码目录/项目 config.yaml —— 配置归用户 profile 所有。
        """
        base = load_voice_config()
        # extra 顶层键直接覆盖语音配置的对应段
        merged = dict(base)
        for key, value in self._extra.items():
            if key in ("wakewords", "conversation_window_sec",
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

        硬件初始化（麦克风/播放器）为同步调用，放入线程池执行，避免阻塞
        gateway 事件循环（否则 Ctrl+C 优雅关闭无法推进）。
        """
        self._loop = asyncio.get_running_loop()
        try:
            if not check_requirements():
                logger.warning("[voice] 前置检查未通过，连接失败")
                return False

            vcfg = self._resolve_voice_cfg()
            self._init_voiceprint(vcfg)
            self._init_asr(vcfg)
            self._init_frontend(vcfg)
            self._init_playback(vcfg)
            _set_active_instance(self)
            # sounddevice 打开麦克风可能在无音频服务器时阻塞/耗时，
            # 放入线程池，避免卡住 gateway 事件循环
            await asyncio.to_thread(self._frontend.start)
            logger.info("[voice] 语音平台已连接（唤醒词: %s）",
                        ", ".join(self._wakewords) or "(未配置)")
            return True
        except Exception as exc:
            logger.error("[voice] connect 失败: %s", exc)
            await self._cleanup_partial()
            return False

    async def _cleanup_partial(self) -> None:
        """清理已创建的资源（connect 失败或 disconnect 时调用，幂等）。"""
        _set_active_instance(None)
        self._stop_wait_tone()
        if self._frontend:
            try:
                self._frontend.stop()
            except Exception:
                pass
        if self._playback_task:
            self._playback_task.cancel()
            try:
                await self._playback_task
            except (asyncio.CancelledError, Exception):
                pass
            self._playback_task = None
        if self._player:
            try:
                self._player.stop()
            except Exception:
                pass

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

    def _init_voiceprint(self, vcfg: Dict[str, Any]) -> None:
        """装配说话人识别（voiceprint.enabled 时）：CAM++ 声纹 + 特征库。

        lib_dir 走 profile 约定（<profile>/models/voiceprint_lib/）。
        CAM++ 模型缺失时降级（voiceprint=None，不阻断语音链路）。
        """
        vp_cfg = vcfg.get("voiceprint", {})
        if not vp_cfg.get("enabled", False):
            return
        from voice_agent.voiceprint import VoiceprintManager

        try:
            self.voiceprint = VoiceprintManager(
                lib_dir=resolve_voiceprint_lib_dir(vcfg),
                threshold=vp_cfg.get("threshold", 0.6),
                speaker_names=vp_cfg.get("speaker_names", {}),
            )
            self._vp_auto_register = vp_cfg.get("auto_register", True)
            self._vp_min_register_sec = vp_cfg.get("min_register_sec", 1.5)
            self._vp_use_cache = vp_cfg.get("speaker_id_cache", True)
            if self.voiceprint.extractor is None:
                logger.warning("[voice] CAM++ 声纹模型不可用，说话人识别降级")
                self.voiceprint = None
            else:
                logger.info("[voice] 说话人识别已启用（特征库: %s）",
                            resolve_voiceprint_lib_dir(vcfg))
        except Exception as exc:
            logger.warning("[voice] 说话人识别初始化失败（降级）: %s", exc)
            self.voiceprint = None

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
            # 说话人分离：需 speaker 引擎（16k_zh_en_speaker_2.0），供本地声纹识别
            enable_speaker_context=1 if self.voiceprint else 0,
        ))
        # 服务端 VAD 可能在用户句间停顿时提前回调 on_final，这里只认
        # on_complete（整段说完）→ last_text 兜底，与 VoiceApp 一致
        self._asr.on_final = None
        self._asr.on_complete = self._on_asr_complete
        self._asr.on_error = self._on_asr_error
        self._asr.on_start = self._on_asr_start
        if self.voiceprint:
            # 每完成一个句子 → 本地声纹识别说话人（并行旁路）
            self._asr.on_sentence = self._on_asr_sentence

    def _init_frontend(self, vcfg: Dict[str, Any]) -> None:
        """装配语音前端（唤醒词 + VAD + 麦克风 → ASR 桥接）。"""
        from voice_agent.voice_frontend import VoiceFrontend, VoiceFrontendConfig

        wake = vcfg.get("wake_word", {})
        vad = vcfg.get("vad", {})
        kws = vcfg.get("kws", {}) or {}
        mic = vcfg.get("mic", {}) or {}
        front_cfg = VoiceFrontendConfig(
            wake_word_enabled=wake.get("enabled", True),
            # keywords.txt 由 tools/gen_keywords.py 生成（触发词 @名字）；
            # wake_word_keyword 仅用于 keywords.txt 缺失时的兜底自动生成，
            # 取第一个触发词（而非名字），避免单/双字误唤醒
            wake_word_keyword=self._first_trigger_keyword()
            or wake.get("keyword", "赫尔墨斯"),
            wake_word_threshold=float(kws.get("threshold", wake.get("sensitivity", 0.25))),
            # 模型目录统一走 profile 约定：VoiceFrontend 内部会再拼
            # kws_model_name，因此这里传"类别目录"（.../models/sherpa-kws）
            kws_model_dir=str(resolve_kws_model_dir(vcfg).parent),
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
            # "auto"/空 → None（sounddevice 自动选择设备）；否则按设备名/索引
            mic_device=None
            if str(mic.get("device") or "").lower() in ("", "auto")
            else mic.get("device"),
            conversation_window_sec=self._conversation_window_sec,
            on_wake_word=self._on_wake_word,
            on_interrupt_request=self._on_interrupt_request,
            on_play_prompt=self._play_prompt,
            prompt_duration_sec=self._prompt_duration_sec(),
            # 连续对话窗口超时：播放告别语，回到唤醒词监听
            on_conversation_timeout=self._play_farewell,
        )
        self._frontend = VoiceFrontend(front_cfg, self._asr)

    def _prompt_duration_sec(self) -> float:
        """读取唤醒提示音 wav 的实际时长（秒），供唤醒后静音保护期使用。"""
        try:
            import wave
            with wave.open(str(PROMPT_PATH), "rb") as wf:
                return wf.getnframes() / wf.getframerate()
        except Exception:
            return 1.0

    def _play_asset(self, name: str) -> None:
        """播放 assets/ 下的 WAV 资产（prompt/farewell/notification）。"""
        path = {"prompt": PROMPT_PATH,
                "farewell": FAREWELL_PATH,
                "notification": NOTIFICATION_PATH}.get(name)
        if path is None or self._player is None or not path.exists():
            return
        try:
            import wave
            with wave.open(str(path), "rb") as wf:
                data = wf.readframes(wf.getnframes())
            self._player.feed(data)
            logger.info("[voice] 🔊 播放 %s", name)
        except Exception as exc:
            logger.warning("[voice] %s 播放失败: %s", name, exc)

    def _play_prompt(self) -> None:
        """唤醒词命中提示音（greeting）：VoiceFrontend 唤醒时回调播放。"""
        self._play_asset("prompt")

    def _play_farewell(self) -> None:
        """连续对话窗口超时的告别语。"""
        self._play_asset("farewell")

    async def disconnect(self) -> None:
        """停止语音组件与监听（幂等，快速返回）。"""
        try:
            await self._cleanup_partial()
            logger.info("[voice] 语音平台已断开")
        except Exception as exc:
            logger.warning("[voice] disconnect 异常: %s", exc)

    # ─── outbound：gateway 回复 → 播报 ───────────────────

    def _enqueue_playback(
        self,
        wake: str,
        segments: List[tuple[str, str]],
        is_final: bool,
    ) -> None:
        """线程安全地把一条播报投进串行播放队列（钩子/兜底路径共用）。"""
        if self._loop is None or self._playback_queue is None:
            return
        asyncio.run_coroutine_threadsafe(
            self._playback_queue.put((wake, list(segments), is_final)), self._loop
        )

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """gateway 转发入口。

        **LLM 回复的播报主通道是 post_api_request 钩子**（钩子只拿得到真实
        API 回复，系统通知不触发钩子，天然排除）。send() 只处理：
        1. [FINISH]（播放控制结束标记）→ 关闭对话窗口；
        2. 钩子已播报的最终回复 → 去重跳过；
        3. 系统通知（home-channel 提示 / 生命周期广播）→ 静默；
        4. 其余内容（钩子漏触发或回复被改写等异常路径）→ 兜底播报。
        """
        text = (content or "").strip()
        if not text:
            return SendResult(success=True)

        if text.upper().strip("[]") == "FINISH":
            logger.info("[voice] [FINISH] 关闭对话窗口")
            self._close_conversation_window()
            return SendResult(success=True)

        wake = chat_id[len(WAKE_CHAT_PREFIX):] \
            if chat_id.startswith(WAKE_CHAT_PREFIX) else chat_id

        # 钩子已播报过这个最终回复 → 去重，避免重复播放
        if self._last_final_by_chat.get(chat_id) == text:
            logger.info("[voice] send 去重：最终回复已由钩子播报，跳过")
            return SendResult(success=True)

        # 系统通知（home-channel 提示 / online-restart 广播）→ 静默
        if text.startswith(_SYSTEM_NOTICE_PREFIXES):
            logger.info("[voice] 系统通知已静默: %s", text[:60])
            return SendResult(success=True)

        # 兜底：钩子未播报的内容（gateway 工具轮文本 / 异常路径）。
        # 正常情况最终轮必由钩子播报（post_api_request stop），因此这里
        # 到达的内容大多是中间轮文本——按中间轮处理（is_final=False，
        # 只播报，不播通知音、不进对话窗口），也无身份前缀。
        if self._playback_queue is None:
            return SendResult(success=False, error="播报队列未就绪")
        segments = parse_emotion_segments(text)
        if not segments:
            return SendResult(success=True)
        self._enqueue_playback(wake, segments, False)
        logger.info("[voice] send 兜底播报 → %s: %s", wake, text[:60])
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

        # 播报开始：停止等待提示音
        self._stop_wait_tone()
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
                self._play_asset("notification")
                if self._player:
                    self._player.wait_for_drain(timeout=5.0)
                    time.sleep(0.05)
                # 播完进入连续对话窗口期：VAD 直接听，无需再喊唤醒词
                if self._frontend:
                    self._frontend.enter_conversation_window()
            else:
                # 中间轮播完：agent 可能仍在处理，恢复等待提示音
                self._start_wait_tone()
        finally:
            if self._tts:
                self._tts.on_audio_chunk = self._player.feed
            if self._frontend:
                self._frontend.set_tts_playing(False)
            player_resume()

    # ─── 等待回复提示音（思考中提示，不再收听）────────────

    def _start_wait_tone(self) -> None:
        """开始循环播放等待提示音（ASR 结束后、LLM 回复前）。

        幂等：计数 +1；计数从 0 变 1 时启动播放任务。
        可能从 ASR 回调线程调用，用 call_soon_threadsafe 调度到事件循环。
        """
        self._wait_tone_waiting += 1
        if self._wait_tone_task is not None:
            return
        if not WAIT_TONE_PATH.is_file():
            logger.warning("[voice] 等待提示音文件缺失: %s", WAIT_TONE_PATH)
            return
        if self._loop is None:
            return
        try:
            from voice_agent.music_control import player_pause
            player_pause(force=True)
        except Exception:
            pass
        if self._player:
            self._player.clear()
        self._loop.call_soon_threadsafe(self._spawn_wait_tone_task)

    def _spawn_wait_tone_task(self) -> None:
        """事件循环线程内创建提示音播放任务（供 call_soon_threadsafe 调用）。"""
        if self._wait_tone_task is None:
            self._wait_tone_task = asyncio.ensure_future(self._wait_tone_loop())

    def _stop_wait_tone(self) -> None:
        """停止等待提示音（TTS 开始 / 对话结束 / 打断时调用）。

        幂等：计数 -1；计数归 0 时取消播放任务并清空播放队列。
        """
        if self._wait_tone_waiting > 0:
            self._wait_tone_waiting -= 1
        if self._wait_tone_waiting == 0 and self._wait_tone_task is not None:
            self._wait_tone_task.cancel()
            self._wait_tone_task = None
            if self._player:
                self._player.clear()

    async def _wait_tone_loop(self) -> None:
        """循环把等待提示音 PCM 喂给播放器（按播放时长节流）。"""
        try:
            import wave
            with wave.open(str(WAIT_TONE_PATH), "rb") as wf:
                data = wf.readframes(wf.getnframes())
            if not data:
                return
            # 每段 2s，feed 后按播放时长 sleep，实现无缝循环
            while True:
                if self._player:
                    self._player.feed(data)
                await asyncio.sleep(len(data) / 32000.0)  # 16k 16bit = 32KB/s
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.warning("[voice] 等待提示音播放异常: %s", exc)

    def _close_conversation_window(self) -> None:
        """关闭对话窗口（[FINISH] / 打断时），回到唤醒词监听。"""
        self._stop_wait_tone()
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

    def _on_asr_start(self) -> None:
        """新一轮 ASR 会话：重置说话人识别上下文。"""
        self._vp_round.clear()
        self._vp_id_cache.clear()

    def _refresh_speaker_names(self) -> None:
        """从 profile 配置同步 speaker_names（内容比较，更新后自动生效）。

        修改 ~/.hermes/voice-agent.yaml 的 voiceprint.speaker_names 后，
        下一次句子识别时自动重载，无需重启 gateway。配置文件很小，
        每句读取一次开销可忽略；仅在实际变化时更新与记录日志。
        """
        if self.voiceprint is None:
            return
        cfg = load_voice_config()
        names = (cfg.get("voiceprint") or {}).get("speaker_names") or {}
        if dict(names) != self.voiceprint.speaker_names:
            self.voiceprint.speaker_names = dict(names)
            logger.info("[voice] speaker_names 已重载: %s", names)

    def _on_asr_sentence(self, info: dict) -> None:
        """
        腾讯云每完成一个句子 → 本地声纹识别说话人（并行旁路，不影响主链路）。

        识别结果收集到 _vp_round，on_complete 时拼成 "[说话人] 内容" 消息发给
        LLM；新说话人按配置自动分配 id 注册；同 speaker_id 缓存避免重复声纹
        计算。逻辑与 VoiceApp 模式一致。
        """
        if self.voiceprint is None:
            return
        # speaker_names 配置变更自动重载（mtime 缓存，无变化零开销）
        self._refresh_speaker_names()
        text = info.get("text", "").strip()
        if not text:
            return
        tx_spk = info.get("speaker_id", 0)
        try:
            # 1) speaker_id 缓存：同一会话同一人只做一次声纹识别
            if self._vp_use_cache and tx_spk in self._vp_id_cache:
                spk_id, score = self._vp_id_cache[tx_spk]
                name = self.voiceprint.display_name(spk_id)
                logger.info("🗣️ [%s] %s（相似度 %.2f，缓存）", name, text, score)
                self._vp_round.append((name, text))
                return

            # 2) 按腾讯云句子时段截取音频 → 声纹识别
            samples = self._asr.get_sentence_audio(info)
            if samples is None or len(samples) == 0:
                logger.warning("🗣️ 无法获取句子音频（缓冲不足）: %s", text[:20])
                return
            spk_id, score = self.voiceprint.identify(samples)

            if spk_id is None:
                # 新说话人：满足最短时长门槛才自动注册，避免劣质声纹入库
                if (self._vp_auto_register
                        and len(samples) / 16000 >= self._vp_min_register_sec):
                    new_id = self.voiceprint.register(samples)
                    if new_id:
                        spk_id, score = new_id, 1.0
                        name = self.voiceprint.display_name(spk_id)
                        logger.info("🗣️ [新说话人] %s → 已注册为 %s", text, name)
                    else:
                        name = "未知"
                        logger.info("🗣️ [未知] %s（注册失败，相似度 %.2f）", text, score)
                else:
                    name = "未知"
                    logger.info("🗣️ [未知] %s（相似度 %.2f）", text, score)
            else:
                name = self.voiceprint.display_name(spk_id)
                logger.info("🗣️ [%s] %s（相似度 %.2f）", name, text, score)

            self._vp_round.append((name, text))
            # 3) 缓存本次识别的 speaker_id（后续该人句子零声纹延迟）
            if self._vp_use_cache and spk_id:
                self._vp_id_cache[tx_spk] = (spk_id, score)
        except Exception as exc:
            logger.warning("说话人识别异常: %s", exc)

    def _on_asr_complete(self) -> None:
        """ASR 整段识别完成（SDK 回调线程）：文本 → MessageEvent → gateway 会话。"""
        try:
            text = (self._asr.last_text or "").strip()
            if not text:
                logger.info("[voice] ASR 结果为空，跳过")
                return
            wake = self._current_wake or next(iter(self._wakewords), "")
            # 说话人识别启用时：把本轮每句 "[说话人] 内容" 拼成消息发给 LLM
            msg = text
            if self.voiceprint is not None and self._vp_round:
                msg = "\n".join(f"[{spk}] {t}" for spk, t in self._vp_round)
                self._vp_round.clear()
            logger.info("[voice] 用户说了: %s", msg[:60])

            source = self.build_source(
                chat_id=self.wake_chat_id(wake),
                chat_name=wake,
                chat_type="dm",
                user_id="voice-user",
                user_name="用户",
            )
            event = MessageEvent(
                text=msg,
                message_type=MessageType.TEXT,
                source=source,
                message_id=f"voice-{int(time.time() * 1000)}",
            )
            # handle_message 是 async，需在 gateway 事件循环线程执行
            if self._loop is not None:
                asyncio.run_coroutine_threadsafe(
                    self.handle_message(event), self._loop
                )
            # 用户已说完，开始等待 LLM 回复：循环播放提示音（提示不再收听）
            self._start_wait_tone()
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

    def _first_trigger_keyword(self) -> str:
        """第一个触发词（wakewords 配置的 keywords 列表首项）。

        触发词与名字分离：名字是会话身份（chat_id=wake:<名字>），
        触发词是 sherpa KWS 实际检测的短语（建议较长，避免误唤醒）。
        """
        for conf in self._wakewords.values():
            if isinstance(conf, dict):
                kws = conf.get("keywords")
                if isinstance(kws, list):
                    for kw in kws:
                        if str(kw).strip():
                            return str(kw).strip()
        return ""

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


def _ensure_profile_env() -> None:
    """把 profile 目录的 .env（~/.hermes/.env）加载进进程环境（不覆盖已有值）。

    凭据（VOICE_SecretId/VOICE_SecretKey/VOICE_AppId 等）约定存 profile .env，
    不读取插件源码目录/项目 .env。
    """
    try:
        from dotenv import load_dotenv
        load_dotenv(voice_config_path().parent / ".env", override=False)
    except Exception:
        pass


def check_requirements() -> bool:
    """前置检查（check_fn 契约：返回 bool）。

    校验腾讯云凭据与 sherpa 模型目录是否存在；缺失时记录日志并返回 False。
    """
    _ensure_profile_env()
    missing = [k for k in ("VOICE_SecretId", "VOICE_SecretKey", "VOICE_AppId")
               if not os.getenv(k)]
    if missing:
        logger.warning("[voice] 缺少腾讯云凭据: %s", ", ".join(missing))
        return False
    model_dir = resolve_kws_model_dir(load_voice_config())
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
    """post_api_request 钩子：**LLM 回复的唯一播报主通道**。

    - finish_reason == "stop"：最终回复 → 播报（is_final=True，播完通知音
      + 进对话窗口），并记录去重（send() 收到同一文本时跳过）
    - finish_reason == "tool_calls" 且 assistant_message.content 非空：
      中间轮阶段文本 → 播报（is_final=False，无通知音、不进对话窗口）
    - content == "[FINISH]"：关闭对话窗口（不播报）

    系统通知（home-channel 提示、生命周期广播）不经过 agent API，不会触发
    本钩子，因此天然不会被播报——无需内容过滤即可保证"voice 只播 LLM 回复"。
    """
    adapter = _get_active_instance()
    if adapter is None:
        return
    logger.info(str(kwargs))
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

        wake = adapter._current_wake or ""
        is_final = finish_reason == "stop"
        segments = parse_emotion_segments(text)
        if not segments:
            return
        if is_final:
            # 记录去重：send() 收到同一最终回复文本时跳过
            adapter._last_final_by_chat[adapter.wake_chat_id(wake)] = text
        adapter._enqueue_playback(wake, segments, is_final)
        logger.info("[voice] 钩子播报(%s): %s",
                    "最终" if is_final else "中间", text[:60])
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
            '【语音交互说明】\n'
            '1. 用户的输入来自语音识别（ASR），可能存在同音字、漏字、多字等错误。\n'
            '   如果问题听起来不合逻辑，结合上下文做合理推断，而不是逐字照搬。\n'
            '2. 回答要简洁，控制在 3 句话以内。\n'
            '   需要列举时用「第一、第二、第三」代替长段落。\n'
            '3. 回答中自然融入确认——不是生硬复述，而是把确认编织在回答里。\n'
            '   例如用户说「今天天气怎么样」，不要说「你是问今天天气吗？」\n'
            '   直接说「今天晴天，25度，适合出门。」\n'
            '   如果确实听清了，不需要额外确认。\n'
            '4. 如果实在听不懂，直接表示没听清。\n'
            '\n'
            '【说话人标识说明】\n'
            '发送给你的每条用户消息会以 [说话人身份] 前缀标注这句话是谁说的，\n'
            '例如「[爸爸] 今天天气怎么样？」；多人连续说话时每句单独标注，如\n'
            '「[爸爸] 今天天气怎么样？\n[妈妈] 顺便查下明天的」。说话人身份用于\n'
            '帮助你理解对话上下文（例如区分不同家庭成员提出的问题），回答时\n'
            '不需要复述说话人。\n'
            '\n'
            '【语音播报规则】\n'
            '你的每条文字回复都会被系统实时语音播报给用户（无需调用任何播报工具）。\n'
            '- 工具调用过程中的阶段性说明（如「让我搜索网络」）请直接作为文字回复输出，\n'
            '  系统会立即播报；不需要文字说明的纯工具调用轮次可以只返回工具调用。\n'
            '- 文字格式回复可以带上情绪标识，格式为：(情绪)你要说的话\n'
            '  例如：(happy)你好，有什么可以帮助你的？\n'
            '  情绪可选值：neutral(中性) sad(悲伤) happy(高兴) angry(生气) fear(恐惧) '
            'story(故事) poetry(诗歌) sajiao(撒娇) disgusted(厌恶) amaze(震惊) exciting(兴奋) '
            'aojiao(傲娇) jieshuo(解说)\n'
            '\n'
            '【播放控制工具规则】\n'
            '你拥有 mpd_ 系列工具用于控制音乐播放（如 mpd_play、mpd_pause、mpd_stop、\n'
            'mpd_previous、mpd_next 等）。\n'
            '- 当你使用播放控制工具时，播放控制工具执行后直接返回 [FINISH] 作为文字回复，\n'
            '  该条文字回复不要使用 (情绪)格式。\n'
            '- [FINISH] 表示操作已完成，系统会自动关闭对话窗口，\n'
            '  用户可以通过再次说唤醒词来继续对话。'
        ),
    )
    ctx.register_hook("post_api_request", on_post_api_request)
    logger.info("voice-platform 已注册（语音平台 + post_api_request 播报钩子）")
