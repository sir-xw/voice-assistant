"""
语音前端核心模块。

管理麦克风采集、唤醒词检测（sherpa-onnx）、
语音活动检测（WebRTC VAD）、ASR 引擎桥接、
连续对话窗口。

跨平台兼容（Windows / Linux）。
"""

import collections
import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHANNELS = 1
FRAME_DURATION_MS = 30
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)
RING_BUFFER_SECONDS = 1.5
RING_BUFFER_SIZE = int(SAMPLE_RATE * RING_BUFFER_SECONDS)

# sherpa-onnx 模型路径（由 VoiceFrontendConfig.kws_* 字段控制，通过 config.yaml 配置）


class VoiceState(Enum):
    IDLE = "idle"
    CONVERSATION = "conversation"  # 连续对话窗口期
    LISTENING = "listening"
    RECORDING = "recording"
    PROCESSING = "processing"
    ERROR = "error"


@dataclass
class VoiceFrontendConfig:
    wake_word_enabled: bool = True
    wake_word_keyword: str = "赫尔墨斯"
    wake_word_threshold: float = 0.25
    wake_word_score: float = 1.0

    # sherpa-onnx KWS 模型路径和文件名（相对项目根目录）
    kws_model_dir: str = "models/sherpa-kws"
    kws_model_name: str = "sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20"
    kws_encoder: str = "encoder-epoch-13-avg-2-chunk-8-left-64.int8.onnx"
    kws_decoder: str = "decoder-epoch-13-avg-2-chunk-8-left-64.onnx"
    kws_joiner: str = "joiner-epoch-13-avg-2-chunk-8-left-64.int8.onnx"
    kws_tokens: str = "tokens.txt"

    vad_mode: int = 3
    min_speech_ms: int = 200
    # VAD 静音超时：语音结束后继续维持 RECORDING 状态的时长（ms）
    vad_silence_threshold_ms: int = 600
    # VAD 语音开始确认帧数：连续多少帧检测到语音才认为真正开始
    vad_speech_confirm_frames: int = 3

    sample_rate: int = SAMPLE_RATE
    frame_duration_ms: int = FRAME_DURATION_MS

    # 连续对话
    conversation_window_sec: float = 8.0

    on_interim: Callable[[str], None] | None = None
    on_final: Callable[[str], None] | None = None
    on_error: Callable[[str], None] | None = None
    on_state_change: Callable[[VoiceState], None] | None = None
    on_wake_word: Callable[[str], None] | None = None  # 参数: 唤醒词原文
    on_interrupt_request: Callable[[], None] | None = None

    on_play_prompt: Callable[[], None] | None = None
    prompt_duration_sec: float = 1.0  # 提示音实际时长秒，由 main.py 传入


def _split_pinyin_syllable(syllable: str) -> str:
    """
    将带声调的拼音音节拆分为声母+韵母（声调跟在韵母后），
    符合 sherpa-onnx keywords.txt 格式。

    例如:
        nǐ  → n ǐ
        hǎo → h ǎo
        xiǎo → x iǎo
        nà  → n à
        a   → a
        é   → é
    """
    initials = [
        "zh", "ch", "sh",
        "b", "p", "m", "f", "d", "t", "n", "l",
        "g", "k", "h", "j", "q", "x",
        "r", "z", "c", "s", "y", "w",
    ]
    matched_init = ""
    for init in sorted(initials, key=len, reverse=True):
        if syllable.startswith(init):
            matched_init = init
            break
    if matched_init:
        final = syllable[len(matched_init):]
        if final:
            return f"{matched_init} {final}"
        else:
            return matched_init
    else:
        return syllable


def _auto_format_keyword(text: str) -> str:
    """
    自动生成 sherpa-onnx 关键词格式：
    - 纯英文：直接保留（alexa → alexa）
    - 含中文：使用 pypinyin 转换为拼音，每个音节拆成声母+韵母
    """
    has_cjk = any('\u4e00' <= c <= '\u9fff' for c in text)
    if has_cjk:
        import pypinyin
        syllables = pypinyin.pinyin([text])
        parts = []
        for syl in syllables:
            for syl1 in syl:
                parts.append(_split_pinyin_syllable(syl1))
        return "  ".join(parts)
    return text.strip().lower()


class VoiceFrontend:
    """
    语音前端：麦克风采集 → 唤醒词 → VAD → ASR 桥接。

    状态机:
      IDLE → (唤醒词) → RECORDING → PROCESSING → (agent回答+TTS)
        → CONVERSATION(8s) → VAD检测到说话 → RECORDING → ...
        → CONVERSATION(超时) → IDLE
      任何状态下唤醒词命中 → 打断 + 新 RECORDING

    VAD 使用静音超时机制：语音结束后保持一段时间 RECORDING，
    避免语音中短暂停顿被误判为结束。同时使用环形缓冲保存最近
    音频，VAD 确认语音开始时先将缓冲中的音频送入 ASR，避免
    丢失语音前段。
    """

    def __init__(self, config: VoiceFrontendConfig, asr_engine):
        self.config = config
        self.asr_engine = asr_engine

        self._state = VoiceState.IDLE
        self._lock = threading.Lock()
        self._running = False

        # 音频环形缓冲区（保存最近 ~1.5 秒 PCM int16 数据）
        self._ring_buffer = collections.deque(maxlen=RING_BUFFER_SIZE)

        # VAD
        self._vad = None
        self._speech_start_time: float | None = None
        self._speech_end_time: float | None = None
        self._min_speech_frames = config.min_speech_ms // config.frame_duration_ms
        # 静音超时相关
        self._silence_threshold_frames = config.vad_silence_threshold_ms // config.frame_duration_ms
        self._silence_counter = 0
        # 语音开始确认
        self._speech_confirm_counter = 0
        self._speech_confirm_threshold = config.vad_speech_confirm_frames
        # 标记是否已进入语音状态（用于静音超时）
        self._was_speech = False

        # 音频流
        self._audio_stream = None
        # 音频处理队列（回调 → 工作线程，避免阻塞音频回调导致 overflow）
        self._audio_queue: queue.Queue = queue.Queue(maxsize=64)
        self._audio_consumer: threading.Thread | None = None

        # sherpa-onnx 唤醒词
        self._spotter = None
        self._kw_stream = None

        # ASR 桥接
        self._asr_active = False
        self._recording_start_time: float = 0.0
        self._prompt_end_time: float = 0.0  # 提示音播完时间点

        # TTS 播放门控：播放期间不送 ASR，只监听唤醒词
        self._tts_playing = False
        # 对话窗口冷却：播放结束后短暂屏蔽回声
        self._conversation_cooldown_until: float = 0.0

        # 连续对话
        self._conversation_timer: threading.Timer | None = None
        self._conversation_window_sec = config.conversation_window_sec

    def start(self):
        if self._running:
            return
        self._running = True
        self._set_state(VoiceState.IDLE)
        self._init_vad()
        if self.config.wake_word_enabled:
            self._init_wake_word()
        self._start_mic()
        logger.info("Voice frontend started")

    def stop(self):
        self._running = False
        self._cancel_conversation_timer()
        if self._audio_stream:
            try:
                self._audio_stream.stop()
                self._audio_stream.close()
            except Exception:
                pass
            self._audio_stream = None
        self._set_state(VoiceState.IDLE)

    def interrupt(self):
        if self.config.on_interrupt_request:
            self.config.on_interrupt_request()

    def enter_conversation_window(self):
        """TTS 播放完毕后调用，进入连续对话窗口期。"""
        if self._state != VoiceState.IDLE:
            return
        self._cancel_conversation_timer()
        # 设置短冷却，让 sounddevice 硬件缓冲排空
        self._conversation_cooldown_until = time.time()
        self._set_state(VoiceState.CONVERSATION)
        self._conversation_timer = threading.Timer(
            self._conversation_window_sec, self._on_conversation_timeout
        )
        self._conversation_timer.daemon = True
        self._conversation_timer.start()
        logger.info(f"Conversation window started ({self._conversation_window_sec}s)")

    def _on_conversation_timeout(self):
        """连续对话窗口超时，回到待机。"""
        self._cancel_conversation_timer()
        self._set_state(VoiceState.IDLE)
        logger.info("Conversation window expired, need wake word")

    def _cancel_conversation_timer(self):
        if self._conversation_timer:
            self._conversation_timer.cancel()
            self._conversation_timer = None

    @property
    def state(self) -> VoiceState:
        return self._state

    def set_tts_playing(self, playing: bool):
        """
        设置 TTS 播放状态。
        True = TTS 正在播放 → 不送 ASR，只监听唤醒词
        False = TTS 播放完毕 → 恢复正常
        """
        self._tts_playing = playing

    # VAD
    def _init_vad(self):
        try:
            import webrtcvad
            self._vad = webrtcvad.Vad(self.config.vad_mode)
            logger.info(f"WebRTC VAD initialized (mode={self.config.vad_mode})")
        except ImportError:
            logger.warning("webrtcvad not installed")
            self._vad = None

    def _is_speech(self, pcm_frame: bytes) -> bool:
        if self._vad is None:
            return True
        try:
            return self._vad.is_speech(pcm_frame, SAMPLE_RATE)
        except Exception:
            return False

    # 唤醒词（sherpa-onnx）
    def _init_wake_word(self):
        import sherpa_onnx

        keyword = self.config.wake_word_keyword
        threshold = self.config.wake_word_threshold
        score = self.config.wake_word_score

        # 构建模型路径（相对项目根目录）
        project_root = Path(__file__).resolve().parent.parent
        model_dir = project_root / self.config.kws_model_dir
        model_path = model_dir / self.config.kws_model_name

        # keywords.txt 由 tools/gen_keywords.py 从 raw_keywords.txt 生成
        # 如果不存在则从第一个 agent 的 name 回退生成一个
        kw_file = model_path / "keywords.txt"
        if not kw_file.exists():
            final_kw = _auto_format_keyword(keyword)
            kw_file.write_text(f"{final_kw} @{keyword}\n", encoding="utf-8")
            logger.info("keywords.txt 不存在，从 config 自动生成: %s → %s", keyword, final_kw)

        if not model_path.exists():
            logger.error(f"sherpa-onnx model not found at {model_path}")
            self._spotter = None
            return

        # 从 config 读取文件名
        enc = model_path / self.config.kws_encoder
        dec = model_path / self.config.kws_decoder
        joi = model_path / self.config.kws_joiner
        tok = model_path / self.config.kws_tokens

        try:
            self._spotter = sherpa_onnx.KeywordSpotter(
                tokens=str(tok), encoder=str(enc), decoder=str(dec), joiner=str(joi),
                num_threads=1, max_active_paths=4,
                keywords_file=str(kw_file),
                keywords_score=score, keywords_threshold=threshold,
                num_trailing_blanks=1, provider="cpu",
            )
            self._kw_stream = self._spotter.create_stream()
            logger.info(f"sherpa-onnx KWS initialized")
        except Exception as e:
            logger.error(f"sherpa-onnx KWS init failed: {e}")
            self._spotter = None
            self._kw_stream = None

    def _check_wake_word(self, audio_samples: np.ndarray) -> str | None:
        """
        检测唤醒词。
        返回 keywords.txt 中 @ 后面的原始文本（如 "赫尔墨斯"），
        未命中则返回 None。
        """
        if self._spotter is None or self._kw_stream is None:
            return None
        try:
            audio_float = audio_samples.astype(np.float32) / 32768.0
            self._kw_stream.accept_waveform(SAMPLE_RATE, audio_float)
            while self._spotter.is_ready(self._kw_stream):
                self._spotter.decode_stream(self._kw_stream)
                result = self._spotter.get_result(self._kw_stream)
                if result:
                    self._spotter.reset_stream(self._kw_stream)
                    # result 是 str，如 "赫尔墨斯"
                    keyword = result.strip()
                    return keyword
        except Exception:
            pass
        return None

    # 麦克风
    def _start_mic(self):
        import sounddevice as sd

        def audio_callback(indata, frames, time_info, status):
            if not self._running:
                return
            if status:
                logger.debug(f"Audio status: {status}")
            pcm_data = indata.tobytes()
            try:
                self._audio_queue.put_nowait((pcm_data, indata))
            except queue.Full:
                pass  # 丢弃最旧帧，防止阻塞回调

        # 启动消费者线程（处理 VAD、唤醒词等耗时操作）
        self._audio_consumer = threading.Thread(
            target=self._audio_consumer_loop, daemon=True, name="audio-proc"
        )
        self._audio_consumer.start()

        try:
            self._audio_stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype="int16",
                blocksize=FRAME_SIZE,
                latency="high",
                callback=audio_callback,
            )
            self._audio_stream.start()
            logger.info("Mic stream started")
        except Exception as e:
            logger.error(f"Failed to start mic: {e}")
            self._set_state(VoiceState.ERROR)
            if self.config.on_error:
                self.config.on_error(f"麦克风启动失败: {e}")

    def _audio_consumer_loop(self):
        """音频队列消费者：在独立线程中处理 VAD、唤醒词等耗时操作。"""
        logger.debug("Audio consumer started")
        while self._running:
            try:
                pcm_bytes, audio_array = self._audio_queue.get(timeout=1.0)
                self._process_audio_frame(pcm_bytes, audio_array)
            except queue.Empty:
                continue
            except Exception:
                logger.exception("Audio consumer error")
        logger.debug("Audio consumer stopped")

    # 核心状态机
    def _process_audio_frame(self, pcm_bytes: bytes, audio_array: np.ndarray):
        samples = np.frombuffer(pcm_bytes, dtype=np.int16)
        # 写入环形缓冲区（始终保存最近 ~1.5 秒音频）
        self._ring_buffer.extend(samples.tolist())

        state = self._state

        if state == VoiceState.IDLE:
            if self.config.wake_word_enabled:
                kw = self._check_wake_word(samples)
                if kw:
                    logger.info("Wake word detected: %s", kw)
                    self._on_wake_word_detected(kw)

        elif state == VoiceState.CONVERSATION:
            # TTS 播放期间不送 ASR
            if self._tts_playing:
                return
            # 播放结束后冷却期内跳过 VAD（防回声）
            if time.time() < self._conversation_cooldown_until:
                return
            # 连续对话窗口：VAD 检测到说话就启动 ASR
            self._vad_process(pcm_bytes)

        elif state == VoiceState.RECORDING:
            # 提示音播放期间：不送 ASR
            if time.time() < self._prompt_end_time:
                return
            self._vad_process(pcm_bytes)

        elif state == VoiceState.PROCESSING:
            pass

    def _vad_process(self, pcm_bytes: bytes):
        """
        带静音超时和语音开始确认的 VAD 处理。

        语音开始：需要连续 _speech_confirm_threshold 帧检测到语音才确认。
        语音结束：检测到静音后进入静音超时计数，超过 _silence_threshold_frames
                 才认为语音真正结束。
        """
        is_speech = self._is_speech(pcm_bytes)

        if is_speech:
            self._speech_confirm_counter += 1
            self._silence_counter = 0

            if self._speech_confirm_counter >= self._speech_confirm_threshold:
                if not self._was_speech:
                    # 首次确认语音开始：将环形缓冲中的音频送入 ASR
                    logger.info("VAD: speech confirmed, flushing ring buffer")
                    self._flush_ring_buffer_to_asr()
                    if self._state != VoiceState.RECORDING:
                        self._start_recording()
                self._was_speech = True
                self._feed_to_asr(pcm_bytes)
        else:
            self._speech_confirm_counter = 0
            if self._was_speech:
                # 语音中遇到静音：进入静音超时计数
                self._silence_counter += 1
                self._feed_to_asr(pcm_bytes)
                if self._silence_counter >= self._silence_threshold_frames:
                    logger.info("VAD: silence threshold expired, speech ended")
                    self._was_speech = False
                    self._silence_counter = 0
                    if self._state == VoiceState.RECORDING:
                        self._stop_recording()
            else:
                # 非语音状态，静音帧不做处理
                pass

    def _flush_ring_buffer_to_asr(self):
        """
        将环形缓冲中的音频数据送入 ASR。
        确保 VAD 确认语音开始时，语音前段不会丢失。
        """
        if not self._asr_active:
            return
        # 环形缓冲是 deque，转为 bytes
        buf_samples = list(self._ring_buffer)
        if buf_samples:
            buf_bytes = np.array(buf_samples, dtype=np.int16).tobytes()
            self.asr_engine.feed_audio(buf_bytes)

    def _on_wake_word_detected(self, keyword: str):
        """唤醒词命中：打断 + 播提示音 + 立即 ASR。"""
        self._cancel_conversation_timer()
        self._speech_start_time = None
        self._consecutive_silence_frames = 0
        self._was_speech = False
        self._silence_counter = 0
        self._speech_confirm_counter = 0

        self.interrupt()

        # 通知上层哪个唤醒词触发了
        if self.config.on_wake_word:
            self.config.on_wake_word(keyword)

        # TTS 播放期间的打断不播提示音，避免回声再次触发
        if not self._tts_playing:
            if self.config.on_play_prompt:
                self.config.on_play_prompt()

        if self._tts_playing:
            # TTS 播放中唤醒：不立即 ASR，等 VAD 确认人声
            self._set_state(VoiceState.LISTENING)
            logger.info("Wake word during TTS → LISTENING (VAD will start ASR)")
        else:
            self._set_state(VoiceState.RECORDING)
            self._asr_active = True
            self._recording_start_time = time.time()
            self._prompt_end_time = time.time() + self.config.prompt_duration_sec
            self.asr_engine.start_recognition()

    def _start_recording(self):
        """启动 ASR 识别。"""
        self._cancel_conversation_timer()
        self._set_state(VoiceState.RECORDING)
        self._consecutive_silence_frames = 0
        self._asr_active = True
        self._recording_start_time = time.time()
        self.asr_engine.start_recognition()

    def _feed_to_asr(self, pcm_bytes: bytes):
        if not self._asr_active:
            return
        self.asr_engine.feed_audio(pcm_bytes)

    def _stop_recording(self):
        self._set_state(VoiceState.PROCESSING)
        self._asr_active = False
        self.asr_engine.stop_recognition()
        threading.Thread(target=self._wait_asr_result, daemon=True).start()

    def _wait_asr_result(self):
        completed = self.asr_engine.wait_for_complete(timeout=10.0)
        if not completed:
            logger.warning("ASR wait_for_complete timeout")
            self._set_state(VoiceState.IDLE)
            return
        final_text = self.asr_engine.last_text
        logger.info(f"📝 ASR final: {final_text}")
        self._set_state(VoiceState.IDLE)

    # 状态管理
    def _set_state(self, state: VoiceState):
        with self._lock:
            old = self._state
            self._state = state
        if state != old:
            logger.debug(f"State: {old.value} → {state.value}")
            if self.config.on_state_change:
                try:
                    self.config.on_state_change(state)
                except Exception:
                    pass
            if state in (VoiceState.IDLE, VoiceState.CONVERSATION):
                self._speech_start_time = None
                self._consecutive_silence_frames = 0
                self._was_speech = False
                self._silence_counter = 0
                self._speech_confirm_counter = 0