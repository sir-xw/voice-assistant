"""
腾讯云实时语音识别引擎 — 官方 SDK 封装。

使用 tencentcloud-speech-sdk-python 提供的 SpeechRecognizer
进行 WebSocket 握手、签名、音频发送、结果回调。
"""

import logging
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from tencentcloud_speech.common.credential import Credential
from tencentcloud_speech.asr.speech_recognizer import SpeechRecognizer, SpeechRecognitionListener

logger = logging.getLogger(__name__)


class ASRState(Enum):
    IDLE = "idle"
    CONNECTING = "connecting"
    RECORDING = "recording"
    PROCESSING = "processing"
    ERROR = "error"
    COMPLETED = "completed"


@dataclass
class TencentASRConfig:
    """腾讯云 ASR 配置（对接 SDK）。"""

    secret_id: str
    secret_key: str
    app_id: str
    engine_model: str = "16k_zh"
    needvad: bool = False
    voice_format: int = 1          # 1=PCM
    filter_dirty: int = 0
    filter_modal: int = 0
    filter_punc: int = 0
    convert_num_mode: int = 1
    word_info: int = 0
    vad_silence_time: int = 0      # 服务端 VAD 静音超时(ms)，0=不设置


class _SDKListener(SpeechRecognitionListener):
    """
    桥接 SDK 回调 → 项目回调。
    """

    def __init__(self, engine: "TencentCloudASREngine"):
        super().__init__()
        self.engine = engine

    def on_recognition_start(self, response):
        logger.info(f"ASR started (voice_id={response.get('voice_id','')[:8]}...)")
        # 注意：SDK 在 WS 连接建立前就同步触发此回调，
        # 此时设置 RECORDING 会导致 feed_audio 调 write() 阻塞。
        # 状态保持 CONNECTING，由 feed_audio 检测 WS 就绪后再切换。
        if self.engine.on_start:
            try:
                self.engine.on_start()
            except Exception:
                pass

    def on_sentence_begin(self, response):
        # slice_type=0：一段话开始
        pass

    def on_recognition_result_change(self, response):
        # slice_type=1：中间非稳态结果
        text = response.get("result", {}).get("voice_text_str", "")
        if text and self.engine.on_interim:
            try:
                self.engine.on_interim(text)
            except Exception:
                pass

    def on_sentence_end(self, response):
        # slice_type=2：最终稳态结果
        text = response.get("result", {}).get("voice_text_str", "")
        if text:
            self.engine._last_final_text = text
            if self.engine.on_final:
                try:
                    self.engine.on_final(text)
                except Exception:
                    pass

    def on_recognition_complete(self, response):
        logger.info("ASR recognition complete")

        # SDK 可能因 final 优先返回而跳过 slice_type=2，
        # 从 complete 消息补发文本
        text = response.get("result", {}).get("voice_text_str", "")
        if text:
            self.engine._last_final_text = text
            if self.engine.on_final:
                try:
                    self.engine.on_final(text)
                except Exception:
                    pass

        self.engine._set_state(ASRState.COMPLETED)
        self.engine._complete_event.set()
        if self.engine.on_complete:
            try:
                self.engine.on_complete()
            except Exception:
                pass

    def on_fail(self, response):
        code = response.get("code", -1)
        message = response.get("message", "Unknown error")

        # 404 时给出更明确的指引
        if "404" in str(message):
            hint = ("腾讯云 ASR 返回 404。常见原因：\n"
                    "  1. VOICE_AppId 不正确 → 检查 .env\n"
                    "  2. 未开通实时语音识别服务\n"
                    "  3. 地区 endpoint 不对")
            logger.error(f"ASR 404: {hint}")
            message = hint

        logger.error(f"ASR failed: code={code}, msg={message}")
        self.engine._set_state(ASRState.ERROR)
        if self.engine.on_error:
            try:
                self.engine.on_error(f"[{code}] {message}")
            except Exception:
                pass


class TencentCloudASREngine:
    """
    腾讯云实时语音识别引擎（SDK 封装）。

    用法:
        engine = TencentCloudASREngine(config)
        engine.on_final = lambda text: print(text)

        engine.start_recognition()
        engine.feed_audio(pcm_bytes)   # 可在任意线程调用
        engine.stop_recognition()
    """

    def __init__(self, config: TencentASRConfig):
        self.config = config

        # SDK 组件
        self._credential = Credential(config.secret_id, config.secret_key)
        self._listener = _SDKListener(self)
        self._recognizer: SpeechRecognizer | None = None

        # 状态
        self._state = ASRState.IDLE
        self._lock = threading.Lock()
        self._complete_event = threading.Event()
        self._last_final_text: str = ""

        # 连接期缓冲：WS 建立前的音频先存着，连上后补发
        self._pending_buffer: list[bytes] = []
        self._pending_buffer_max = 30  # 约 6s（200ms/frame × 30）

        # 回调（外部注册）
        self.on_start: Callable[[], None] | None = None
        self.on_interim: Callable[[str], None] | None = None
        self.on_final: Callable[[str], None] | None = None
        self.on_complete: Callable[[], None] | None = None
        self.on_error: Callable[[str], None] | None = None
        self.on_state_change: Callable[[ASRState], None] | None = None

    # ─── 公共 API ────────────────────────────────────────

    def start_recognition(self):
        """
        开始新的识别会话。
        创建 SpeechRecognizer、建立 WebSocket 连接。
        线程安全，可在任意线程调用。
        """
        self._complete_event.clear()
        self._last_final_text = ""
        self._pending_buffer.clear()
        self._set_state(ASRState.CONNECTING)

        # 创建 SDK Recognizer
        self._recognizer = SpeechRecognizer(
            appid=self.config.app_id,
            credential=self._credential,
            engine_model_type=self.config.engine_model,
            listener=self._listener,
        )

        # 配置参数
        r = self._recognizer
        r.set_voice_format(self.config.voice_format)
        r.set_filter_dirty(self.config.filter_dirty)
        r.set_filter_modal(self.config.filter_modal)
        r.set_filter_punc(self.config.filter_punc)
        r.set_convert_num_mode(self.config.convert_num_mode)
        r.set_word_info(self.config.word_info)
        r.set_need_vad(1 if self.config.needvad else 0)
        if self.config.vad_silence_time > 0:
            r.set_vad_silence_time(self.config.vad_silence_time)

        # 启动连接（非阻塞，内部开线程）
        r.start()

    def feed_audio(self, pcm_chunk: bytes):
        """
        馈送 PCM 音频数据。
        可在任意线程调用。

        关键：SDK 的 on_recognition_start 在 WS 连接前就触发，
        所以这里按 SDK 的实际状态判断，而非依赖 ASRState 的 RECORDING。
        """
        if self._recognizer is None:
            return

        # 检查 SDK 内部 WebSocket 实际状态
        sdk_status = getattr(self._recognizer, "status", 0)

        if sdk_status == 2:  # OPENED → WS 已就绪，正常发送
            # 补发连接期缓冲
            if self._pending_buffer:
                self._flush_pending()
                self._set_state(ASRState.RECORDING)
            try:
                self._recognizer.write(pcm_chunk)
            except Exception as e:
                logger.warning(f"feed_audio error: {e}")

        elif sdk_status == 1:  # STARTED → WS 连接中，缓冲
            self._pending_buffer.append(pcm_chunk)
            if len(self._pending_buffer) > self._pending_buffer_max:
                self._pending_buffer.pop(0)

        # 其他状态（0=NOTOPEN, 3=FINAL, 4=ERROR, 5=CLOSED）: 丢弃

    def stop_recognition(self):
        """
        结束当前识别会话。
        发送结束标志，等待 SDK 回调 on_recognition_complete。
        """
        if self._recognizer is None:
            return
        self._set_state(ASRState.PROCESSING)
        try:
            self._recognizer.stop()
        except Exception as e:
            logger.warning(f"stop_recognition error: {e}")

    # ─── 属性 ────────────────────────────────────────────

    @property
    def state(self) -> ASRState:
        with self._lock:
            return self._state

    @property
    def voice_id(self) -> str:
        if self._recognizer:
            return self._recognizer.voice_id or ""
        return ""

    @property
    def last_text(self) -> str:
        """本次识别最终文本。"""
        return self._last_final_text

    def wait_for_complete(self, timeout: float = 10.0) -> bool:
        """
        阻塞等待本次识别完成（on_recognition_complete 触发后返回）。

        Args:
            timeout: 超时秒数

        Returns:
            True=正常完成, False=超时
        """
        return self._complete_event.wait(timeout=timeout)

    def _flush_pending(self):
        """将连接期缓冲的音频补发给 SDK。"""
        if not self._pending_buffer or not self._recognizer:
            return
        chunks = list(self._pending_buffer)
        self._pending_buffer.clear()
        for chunk in chunks:
            try:
                self._recognizer.write(chunk)
            except Exception as e:
                logger.warning(f"flush_pending error: {e}")
                break

    # ─── 内部 ────────────────────────────────────────────

    def _set_state(self, state: ASRState):
        with self._lock:
            old = self._state
            self._state = state
        if state != old:
            logger.debug(f"ASR state: {old.value} → {state.value}")
            if self.on_state_change:
                try:
                    self.on_state_change(state)
                except Exception:
                    pass
