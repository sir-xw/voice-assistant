"""
PCM 音频播放器。

从队列消费 PCM 音频数据，通过 sounddevice 实时播放。
支持暂停、恢复、停止（打断）。

跨平台兼容（Windows / Linux），自动选择可用音频设备。
"""

import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import os

logger = logging.getLogger(__name__)

# set fixed pulse client name
os.environ['PULSE_CLIENTNAME'] = 'hermes-voice-agent'


class PlayerState(Enum):
    IDLE = "idle"
    PLAYING = "playing"
    PAUSED = "paused"
    STOPPED = "stopped"


@dataclass
class AudioPlayerConfig:
    """音频播放器配置。"""

    sample_rate: int = 16000
    channels: int = 1
    dtype: str = "int16"
    blocksize: int = 3200       # 100ms 的音频块
    device: str | None = None    # None = 默认输出设备


class AudioPlayer:
    """
    PCM 音频播放器。

    用法:
        player = AudioPlayer()
        player.start()

        # 在任意线程馈送 PCM 数据
        player.feed(pcm_bytes)

        # 控制
        player.pause()
        player.resume()
        player.stop()        # 清空队列 + 停止播放
        player.clear()       # 仅清空队列（不停止）
    """

    def __init__(self, config: AudioPlayerConfig | None = None):
        self.config = config or AudioPlayerConfig()
        self._audio_queue: queue.Queue[bytes | None] = queue.Queue()
        self._state = PlayerState.IDLE
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stream = None
        self._stop_event = threading.Event()
        self._drain_event = threading.Event()  # set when all audio has played

        # 回调
        self.on_state_change: Callable[[PlayerState], None] | None = None

    # ─── 公共 API ────────────────────────────────────────

    def start(self):
        """启动播放器线程。"""
        if self._thread and self._thread.is_alive():
            logger.warning("Audio player already running")
            return

        self._stop_event.clear()
        self._set_state(PlayerState.IDLE)
        self._thread = threading.Thread(
            target=self._run,
            name="audio-player",
            daemon=True,
        )
        self._thread.start()
        logger.info("Audio player started")

    def stop(self):
        """停止播放器，清空队列，关闭流。"""
        self._stop_event.set()
        self.clear()
        self._set_state(PlayerState.STOPPED)

        self._close_stream()

        if self._thread:
            self._thread.join(timeout=3)
            self._thread = None

        logger.info("Audio player stopped")

    def feed(self, pcm_data: bytes):
        """馈送 PCM 数据到播放队列（线程安全）。"""
        if self._state == PlayerState.STOPPED:
            return
        self._drain_event.clear()
        self._audio_queue.put(pcm_data)

    def pause(self):
        """暂停播放。"""
        with self._lock:
            if self._state == PlayerState.PLAYING:
                self._set_state(PlayerState.PAUSED)

    def resume(self):
        """恢复播放。"""
        with self._lock:
            if self._state == PlayerState.PAUSED:
                self._set_state(PlayerState.PLAYING)

    def clear(self):
        """清空播放队列。"""
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

    def wait_for_drain(self, timeout: float = 10.0) -> bool:
        """
        阻塞直到所有音频已写入声卡并播完。
        用于判断 TTS 是否真正播完。

        原理：播放线程在队列空且无待写数据时设置 _drain_event，
        feed() 时清除该事件。

        Args:
            timeout: 超时秒数

        Returns:
            True=已播完, False=超时
        """
        return self._drain_event.wait(timeout=timeout)

    def is_playing(self) -> bool:
        return self._state == PlayerState.PLAYING

    @property
    def state(self) -> PlayerState:
        return self._state

    # ─── 内部 ────────────────────────────────────────────

    def _set_state(self, state: PlayerState):
        self._state = state
        if self.on_state_change:
            try:
                self.on_state_change(state)
            except Exception:
                pass

    def _open_stream(self, sd):
        """打开/重建输出流（已存在则先关闭）。失败抛异常，由调用方决定重试。"""
        self._close_stream()
        stream = sd.OutputStream(
            samplerate=self.config.sample_rate,
            channels=self.config.channels,
            dtype=self.config.dtype,
            blocksize=self.config.blocksize,
            latency="low",
            device="pulse",
        )
        stream.start()
        self._stream = stream
        self._set_state(PlayerState.PLAYING)

    def _close_stream(self):
        """关闭输出流（幂等）。断流/自愈重建时释放底层 PortAudio 资源。"""
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass

    def _run(self):
        """播放线程主循环：消费队列写音频。

        流缺失或写失败（PulseAudio/PipeWire 重启、USB 设备重建等导致连接
        断开）时自动关闭并重建输出流，最多重试 3 次；仍失败则丢弃该数据块
        并等待后续数据再次重建 —— 不再因一次写失败就退出播放线程。
        """
        import sounddevice as sd

        try:
            # 初始打开失败不致命：后续有数据时会按需重建
            try:
                self._open_stream(sd)
            except Exception as e:
                logger.error(f"Audio output stream open failed (will retry on data): {e}")
                self._set_state(PlayerState.IDLE)

            while not self._stop_event.is_set():
                # 从队列取数据，超时以便检查 stop 事件
                try:
                    data = self._audio_queue.get(timeout=0.05)
                    self._drain_event.clear()
                except queue.Empty:
                    # 队列空且已无待写数据 → 播放完毕
                    self._drain_event.set()
                    continue

                if data is None:
                    break

                # 检查暂停
                while self._state == PlayerState.PAUSED and not self._stop_event.is_set():
                    time.sleep(0.1)

                if self._stop_event.is_set():
                    break

                # bytes → numpy int16（sounddevice 需要数组）
                if isinstance(data, bytes):
                    try:
                        import numpy as np
                        data = np.frombuffer(data, dtype=np.int16)
                    except Exception:
                        pass

                # 写入播放流：流缺失或写失败 → 关闭重建，最多重试 3 次
                written = False
                for attempt in range(3):
                    try:
                        if self._stream is None:
                            self._open_stream(sd)
                        self._stream.write(data)
                        written = True
                        break
                    except Exception as e:
                        logger.warning(f"Audio write error (attempt {attempt + 1}): {e}")
                        self._close_stream()
                        time.sleep(0.5)
                if not written:
                    logger.error("Audio write failed after retries, dropping chunk")
                    self._set_state(PlayerState.IDLE)

        except Exception as e:
            logger.error(f"Audio player error: {e}")
            self._set_state(PlayerState.STOPPED)
        finally:
            self._close_stream()
            self._set_state(PlayerState.IDLE)
