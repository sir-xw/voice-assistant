"""
腾讯云流式语音合成引擎 — 官方 SDK 封装。

使用 tencentcloud-speech-sdk-python 提供的 FlowingSpeechSynthesizer
进行流式文本→语音合成，输出 PCM 音频直接播放。
"""

import logging
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from voice_service.tencentcloud_speech.common.credential import Credential
from voice_service.tencentcloud_speech.tts.flowing_speech_synthesizer import (
    FlowingSpeechSynthesizer,
    FlowingSpeechSynthesisListener,
    FlowingSpeechSynthesizer_ACTION_SYNTHESIS,
    FlowingSpeechSynthesizer_ACTION_COMPLETE,
    FlowingSpeechSynthesizer_ACTION_RESET,
)

logger = logging.getLogger(__name__)


@dataclass
class TencentTTSConfig:
    """腾讯云流式语音合成配置。"""

    secret_id: str
    secret_key: str
    app_id: str
    voice_type: int = 101001        # 中文女生 晓晓
    codec: str = "pcm"              # pcm | mp3
    sample_rate: int = 16000
    speed: float = 0.0              # -2~6
    volume: float = 10.0            # -10~10，为了和音乐音量匹配，设置为最大，靠全局音量来共同调节
    emotion_category: str = ""      # 情绪/风格：neutral(中性) sad(悲伤) happy(高兴) angry(生气) fear(恐惧) story(故事) poetry(诗歌) sajiao(撒娇) disgusted(厌恶) amaze(震惊) exciting(兴奋) aojiao(傲娇) jieshuo(解说)
    emotion_intensity: int = 100    # 情绪强度 0-100，默认100


class _TTSListener(FlowingSpeechSynthesisListener):
    """桥接 SDK TTS 回调 → 项目回调。"""

    def __init__(self, engine: "TencentCloudTTSEngine"):
        super().__init__()
        self.engine = engine

    def on_synthesis_start(self, session_id):
        logger.info(f"TTS started (session_id={session_id[:8]}...)")
        if self.engine.on_start:
            try:
                self.engine.on_start()
            except Exception:
                pass

    def on_audio_result(self, audio_bytes: bytes):
        """收到 PCM 音频块 → 转发播放器。"""
        if self.engine.on_audio_chunk:
            try:
                self.engine.on_audio_chunk(audio_bytes)
            except Exception:
                pass

    def on_text_result(self, response):
        """时间戳等信息，可忽略。"""
        pass

    def on_synthesis_end(self):
        logger.info("TTS synthesis end")
        if self.engine.on_end:
            try:
                self.engine.on_end()
            except Exception:
                pass

    def on_synthesis_fail(self, response):
        code = response.get("code", -1)
        message = response.get("message", "Unknown error")
        logger.error(f"TTS failed: code={code}, msg={message}")
        if self.engine.on_error:
            try:
                self.engine.on_error(f"[{code}] {message}")
            except Exception:
                pass


class TencentCloudTTSEngine:
    """
    腾讯云流式语音合成引擎（SDK 封装）。

    用法:
        engine = TencentCloudTTSEngine(config)
        engine.on_audio_chunk = lambda pcm: player.feed(pcm)

        engine.synthesize("你好，请问有什么可以帮你的？")
        engine.synthesize("这是第二句。")
        engine.complete()
        engine.wait()                         # 等待合成结束
    """

    def __init__(self, config: TencentTTSConfig):
        self.config = config
        self._credential = Credential(config.secret_id, config.secret_key)
        self._listener = _TTSListener(self)
        self._synthesizer: FlowingSpeechSynthesizer | None = None
        self._lock = threading.Lock()

        # 回调
        self.on_audio_chunk: Callable[[bytes], None] | None = None
        self.on_start: Callable[[], None] | None = None
        self.on_end: Callable[[], None] | None = None
        self.on_error: Callable[[str], None] | None = None

    def set_emotion(self, category: str, intensity: int = 100):
        """
        动态设置本次合成的情绪/风格（需在 start() 之后、synthesize() 之前调用）。

        Args:
            category: 情绪类别
            intensity: 情绪强度 0-200
        """
        if self._synthesizer and category:
            self._synthesizer.set_emotion_category(category)
            self._synthesizer.set_emotion_intensity(intensity)
            logger.info("TTS emotion set: %s (intensity=%d)", category, intensity)

    # ─── 公共 API ────────────────────────────────────────

    def start(self):
        """
        启动 TTS 会话。
        建立 WebSocket 连接，等待 READY 事件。
        调用后可通过 synthesize() 发送文本。
        """
        self._synthesizer = FlowingSpeechSynthesizer(
            appid=self.config.app_id,
            credential=self._credential,
            listener=self._listener,
        )

        s = self._synthesizer
        s.set_voice_type(self.config.voice_type)
        s.set_codec(self.config.codec)
        s.set_sample_rate(self.config.sample_rate)
        s.set_speed(self.config.speed)
        s.set_volume(self.config.volume)
        s.set_enable_subtitle(0)  # 不需要时间戳

        # 设置情绪/风格
        if self.config.emotion_category:
            s.set_emotion_category(self.config.emotion_category)
            s.set_emotion_intensity(self.config.emotion_intensity)
            logger.info("TTS emotion: %s (intensity=%d)",
                        self.config.emotion_category, self.config.emotion_intensity)

        s.start()

        # 等待 READY 事件
        if not s.wait_ready(5000):
            logger.warning("TTS READY timeout")
            if self.on_error:
                self.on_error("TTS 连接超时")

    def synthesize(self, text: str):
        """
        发送文本进行合成。

        Args:
            text: 待合成的文本
        """
        if self._synthesizer is None:
            logger.warning("TTS not started")
            return
        if not text.strip():
            return
        logger.debug(f"TTS synthesize: {text[:30]}...")
        self._synthesizer.process(text, FlowingSpeechSynthesizer_ACTION_SYNTHESIS)

    def complete(self):
        """通知服务端文本已全部发送。"""
        if self._synthesizer is None:
            return
        logger.info("TTS sending complete signal")
        self._synthesizer.complete(FlowingSpeechSynthesizer_ACTION_COMPLETE)

    def wait(self, timeout: float = 30.0):
        """
        等待合成结束（阻塞）。

        Args:
            timeout: 超时秒数
        """
        if self._synthesizer is None:
            return
        t = threading.Thread(target=self._synthesizer.wait, daemon=True)
        t.start()
        t.join(timeout=timeout)

    def close(self):
        """关闭 TTS 连接。"""
        try:
            self.complete()
            self.wait(5)
        except Exception:
            pass
        self._synthesizer = None

    def interrupt(self):
        """
        打断当前合成。
        清空缓存并关闭连接。
        """
        logger.info("TTS interrupt")
        if self._synthesizer:
            try:
                self._synthesizer.reset(FlowingSpeechSynthesizer_ACTION_RESET)
                self._synthesizer.complete(FlowingSpeechSynthesizer_ACTION_COMPLETE)
            except Exception:
                pass
            self._synthesizer = None
