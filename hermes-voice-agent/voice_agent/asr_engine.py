"""
腾讯云实时语音识别引擎 — 官方 SDK 封装（realtime_recognizer_v2）。

使用 tencentcloud-speech-sdk-python 提供的 RealtimeRecognizerV2
进行 WebSocket 握手、签名、音频发送、结果回调（句子模式）。
"""

import logging
import sys
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from voice_agent.tencentcloud_speech.common.credential import Credential
from voice_agent.tencentcloud_speech.asr.realtime_recognizer_v2 import RealtimeRecognitionListenerV2, RealtimeRecognizerV2


logger = logging.getLogger(__name__)

# 会话音频缓冲上限（秒）：句子时间戳基于 feed 起点，缓冲需覆盖"句子可能跨越
# 的最长时段"。实时场景中一轮对话一般 < 30s，保留最近 30s 足够且内存仅 ~1MB
SESSION_AUDIO_MAX_SEC = 30.0
SAMPLE_RATE = 16000
SAMPLE_BYTES = 2  # int16


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
    enable_speaker_context: int = 0  # 是否开启说话人分离


class _SDKListener(RealtimeRecognitionListenerV2):
    """
    桥接 SDK 回调 → 项目回调。

    realtime_recognizer_v2 句子模式回调：
      - on_recognition_start      连接建立后触发
      - on_recognition_sentences  句子列表（sentence_type: 0=中间, 1=最终）
      - on_sentence_end           final==1，整个识别结束
      - on_fail                   失败
    """

    def __init__(self, engine: "TencentCloudASREngine"):
        super().__init__()
        self.engine = engine

    def on_recognition_start(self, response):
        logger.info(f"ASR started (voice_id={response.get('voice_id','')[:8]}...)")
        # v2 在 WS 连接建立后才触发此回调，此时可安全进入 RECORDING，
        # 由 feed_audio 检测到 OPENED 后切换。
        if self.engine.on_start:
            try:
                self.engine.on_start()
            except Exception:
                pass

    def on_recognition_sentences(self, response):
        # 句子模式：每条消息都是句子列表
        sentences = response.get("sentences", {}).get("sentence_list", [])
        interim_parts: list[str] = []
        for s in sentences:
            text = s.get("sentence", "")
            if s.get("sentence_type") == 0:
                # 中间非稳态结果
                if text:
                    interim_parts.append(text)
            else:
                # 最终稳态结果（sentence_type=1），保留完整句子信息
                self._collect_final_sentences([s])
        if interim_parts and self.engine.on_interim:
            try:
                self.engine.on_interim("".join(interim_parts))
            except Exception:
                pass

    def _collect_final_sentences(self, sentences):
        """
        收集最终稳态句子（sentence_type=1）的完整信息并回调。

        腾讯云实时识别（开启说话人分离后）每条最终句子自带：
          - speaker_id: 说话人编号（0, 1, 2, ...）
          - start_time / end_time: 句子在音频中的起止毫秒
        这些信息供上层做"按句子段落截取音频 → 声纹识别说话人身份"。

        收集到 engine._sentences（按到达顺序），同时向后兼容：
          - engine._last_final_text / on_final 保持原有行为
          - 新增可选回调 engine.on_sentence(info)
        """
        for s in sentences:
            if s.get("sentence_type") != 1:
                continue
            text = s.get("sentence", "")
            info = {
                "text": text,
                "speaker_id": s.get("speaker_id", 0),
                "start_ms": int(s.get("start_time", 0) or 0),
                "end_ms": int(s.get("end_time", 0) or 0),
                "sentence_id": s.get("sentence_id", 0),
            }
            logger.debug("final sentence: %s", info)
            if text:
                self.engine._last_final_text = text
            self.engine._sentences.append(info)
            if self.engine.on_sentence:
                try:
                    self.engine.on_sentence(info)
                except Exception:
                    pass
            if text and self.engine.on_final:
                try:
                    self.engine.on_final(text)
                except Exception:
                    pass

    def on_sentence_end(self, response):
        logger.info("ASR recognition complete")
        # final==1：整个识别结束。从最终句子兜底补发文本/句子信息
        sentences = response.get("sentences", {}).get("sentence_list", [])
        self._collect_final_sentences(sentences)
        final_parts = [s.get("sentence", "") for s in sentences
                       if s.get("sentence_type") == 1]
        if final_parts:
            text = "".join(final_parts)
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
        self._recognizer: RealtimeRecognizerV2 | None = None

        # 状态
        self._state = ASRState.IDLE
        self._lock = threading.Lock()
        self._complete_event = threading.Event()
        self._last_final_text: str = ""
        # 本次识别的最终句子列表（含 speaker_id / 起止毫秒），按到达顺序
        self._sentences: list[dict] = []

        # 会话音频缓冲：feed_audio 送入的 16k PCM 累积（供按句子时间戳截取音频
        # 做本地声纹识别）。deque 存块，保留最近 SESSION_AUDIO_MAX_SEC 秒
        self._session_audio: deque[bytes] = deque()
        self._session_audio_sec: float = 0.0

        # 连接期缓冲：WS 建立前的音频先存着，连上后补发
        self._pending_buffer: list[bytes] = []
        self._pending_buffer_max = 30  # 约 6s（200ms/frame × 30）

        # 回调（外部注册）
        self.on_start: Callable[[], None] | None = None
        self.on_interim: Callable[[str], None] | None = None
        self.on_final: Callable[[str], None] | None = None
        self.on_sentence: Callable[[dict], None] | None = None
        self.on_complete: Callable[[], None] | None = None
        self.on_error: Callable[[str], None] | None = None
        self.on_state_change: Callable[[ASRState], None] | None = None

    # ─── 公共 API ────────────────────────────────────────

    def start_recognition(self):
        """
        开始新的识别会话。
        创建 RealtimeRecognizerV2、建立 WebSocket 连接。
        线程安全，可在任意线程调用；非阻塞（连接在后台线程进行）。
        """
        self._complete_event.clear()
        self._last_final_text = ""
        self._sentences.clear()
        self._session_audio.clear()
        self._session_audio_sec = 0.0
        self._pending_buffer.clear()
        self._set_state(ASRState.CONNECTING)

        # 创建 SDK Recognizer（v2 句子模式）
        self._recognizer = RealtimeRecognizerV2(
            appid=self.config.app_id,
            credential=self._credential,
            engine_model_type=self.config.engine_model,
            listener=self._listener,
        )

        # 配置参数
        r = self._recognizer
        r.set_voice_format(self.config.voice_format)
        r.set_convert_num_mode(self.config.convert_num_mode)
        r.set_need_vad(1 if self.config.needvad else 0)
        if self.config.vad_silence_time > 0:
            r.set_vad_silence_time(self.config.vad_silence_time)
        if self.config.enable_speaker_context:
            # 说话人分离（可选）：需同时开启 diarization 与 speaker context
            r.set_speaker_diarization(1)
            r.set_enable_speaker_context(1)

        # v2 的 start() 是同步连接（阻塞直到 WS 就绪 + 收首包），
        # 放到后台线程执行，保持 start_recognition 非阻塞；
        # 连接失败通过 on_fail 上报，与旧版行为一致。
        def _connect():
            try:
                r.start()
            except Exception as e:
                logger.error(f"ASR connect failed: {e}")
                fail_resp = {"code": -1, "message": str(e),
                             "voice_id": r.voice_id}
                self._listener.on_fail(fail_resp)

        threading.Thread(target=_connect, daemon=True).start()

    def feed_audio(self, pcm_chunk: bytes):
        """
        馈送 PCM 音频数据。
        可在任意线程调用。

        v2 的 start() 在后台线程同步连接，连接建立后触发
        on_recognition_start；此处按 SDK 内部 _status 判断 WS 是否就绪。
        """
        if self._recognizer is None:
            return

        # 会话音频缓冲：无条件累积（与 ASR 收到的音频流顺序一致——连接期
        # pending 也是按 feed 顺序缓冲、连接后按序补发，故时间戳天然对齐）
        self._append_session_audio(pcm_chunk)

        # 检查 SDK 内部 WebSocket 实际状态（0=NOTOPEN, 1=STARTED,
        # 2=OPENED, 3=FINAL, 4=ERROR, 5=CLOSED）
        sdk_status = getattr(self._recognizer, "_status", 0)

        if sdk_status == 2:  # OPENED → WS 已就绪，正常发送
            # 补发连接期缓冲
            if self._pending_buffer:
                self._flush_pending()
                self._set_state(ASRState.RECORDING)
            try:
                self._recognizer.write(pcm_chunk)
            except Exception as e:
                logger.warning(f"feed_audio error: {e}")

        elif sdk_status in (0, 1):  # NOTOPEN/STARTED → WS 连接中，缓冲
            self._pending_buffer.append(pcm_chunk)
            if len(self._pending_buffer) > self._pending_buffer_max:
                self._pending_buffer.pop(0)

        # 其他状态（3=FINAL, 4=ERROR, 5=CLOSED）: 丢弃

    def stop_recognition(self):
        """
        结束当前识别会话。
        v2 的 stop() 发送结束标志并等待接收线程结束，期间服务端
        返回 final 消息会触发 on_sentence_end → COMPLETED。
        """
        if self._recognizer is None:
            return
        self._set_state(ASRState.PROCESSING)
        try:
            self._recognizer.stop()
        except Exception as e:
            logger.warning(f"stop_recognition error: {e}")

    # ─── 会话音频缓冲（供本地声纹识别按句子时间戳截取音频）──────

    def _append_session_audio(self, pcm: bytes):
        """累积会话音频，超出 SESSION_AUDIO_MAX_SEC 时丢弃最旧数据。"""
        if not pcm:
            return
        self._session_audio.append(pcm)
        self._session_audio_sec += len(pcm) / (SAMPLE_RATE * SAMPLE_BYTES)
        while (self._session_audio_sec > SESSION_AUDIO_MAX_SEC
               and len(self._session_audio) > 1):
            dropped = self._session_audio.popleft()
            self._session_audio_sec -= len(dropped) / (SAMPLE_RATE * SAMPLE_BYTES)

    def get_sentence_audio(self, sentence: dict):
        """
        按句子的起止毫秒（相对本次 feed 音频流起点）从会话缓冲截取音频，
        返回 16k float32 样本（[-1, 1]）；缓冲不足或区间非法时返回 None。

        供上层在 on_sentence 回调中取该句音频做本地声纹识别。
        """
        import numpy as np

        if not self._session_audio:
            return None
        start_byte = int(sentence.get("start_ms", 0) / 1000.0
                         * SAMPLE_RATE * SAMPLE_BYTES)
        end_byte = int(sentence.get("end_ms", 0) / 1000.0
                       * SAMPLE_RATE * SAMPLE_BYTES)
        if start_byte < 0 or end_byte <= start_byte:
            return None
        buf = b"".join(self._session_audio)
        if start_byte >= len(buf):
            return None
        end_byte = min(end_byte, len(buf))
        chunk = buf[start_byte:end_byte]
        # 不足 50ms 的片段无识别意义
        if len(chunk) < int(0.05 * SAMPLE_RATE) * SAMPLE_BYTES:
            return None
        return np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768

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

    @property
    def sentences(self) -> list[dict]:
        """
        本次识别的最终句子列表（按到达顺序），每项含:
          text / speaker_id / start_ms / end_ms / sentence_id
        开启说话人分离（enable_speaker_context=1）后 speaker_id 才有意义。
        """
        return list(self._sentences)

    def wait_for_complete(self, timeout: float = 10.0) -> bool:
        """
        阻塞等待本次识别完成（on_sentence_end 触发后返回）。

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
