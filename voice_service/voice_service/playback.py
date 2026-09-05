"""
播报队列消费者（outbound，M3）。

消费 speak/asset 两类任务，串行出声：
- speak（interim/final/raw）：逐段按情绪经腾讯云 TTS 流式合成 → AudioPlayer 播放；
  final 播完 → 通知音 → 进入连续对话窗口；interim 播完若仍在等待回复 → 恢复等待音；
- asset：提示音/通知音/告别语 wav 播放（不入 TTS）；
- 等待音：ASR 结果上行后循环播放（提示"不再收听"），首个 speak 到达即停；
  无回复超时（wait_reply_timeout_sec）→ 告别语收尾；
- 旧轮兜底：speak 带 turn_seq，小于已消费回合的迟到帧被丢弃（防被打断的旧回复播出来）。

音乐避让：每次 TTS/等待音出声前经 MusicHoldClient 通知协调器 hold，
静默后 release —— 恢复/保持播放由协调器的 intent 决定（架构文档 §10）。

线程模型：consumer 等在 asyncio 事件循环；TTS/播放阻塞操作放 to_thread
（保持 WS/心跳响应）；SDK/音频线程经 run_coroutine_threadsafe 调 async 方法。
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import protocol as P
from . import voice_assets as VA
from .service_config import VoiceServiceConfig

logger = logging.getLogger("voice_service.playback")

# 等待音按 16k/16bit 播放速率计算 sleep（32KB/s）
_PCM_BYTES_PER_SEC = 16000 * 2


class Playback:
    def __init__(self, cfg: VoiceServiceConfig, server,
                 frontend_provider: Callable[[], Optional[Any]],
                 *, silent: bool = False):
        self.cfg = cfg
        self.server = server
        self.get_frontend = frontend_provider
        self.silent = silent

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._queue: Optional[asyncio.Queue] = None
        self._task: Optional[asyncio.Task] = None

        self._player = None          # AudioPlayer（silent 时不建）
        self._tts = None             # TencentCloudTTSEngine
        self._music: Optional[Any] = None   # MusicHoldClient

        # 回合与状态
        # 每个会话（wake）最新入队的 speak turn_seq。丢帧规则：只有「同一会话、
        # 严格更旧」的 speak 是"打断后迟到的旧回复"才丢弃（R2 兜底，正常几乎不触发）；
        # 同轮多段（interim 与 final 同 turn_seq）与不同会话的并行回复都放行。
        self._last_turn_by_wake: Dict[str, int] = {}
        self._interrupted = False    # 本地打断标记（to_thread 内检查）
        self._playing = False        # 正在播放 TTS/资产（打断判定用）
        # 等待回复
        self._waiting = 0
        self._wait_task: Optional[asyncio.Task] = None
        self._wait_timeout_task: Optional[asyncio.Task] = None

    # ─── 生命周期 ────────────────────────────────────────

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue()
        raw = self.cfg.raw
        mc_cfg = raw.get("music_coordinator") or {}
        if mc_cfg.get("enabled", True) and mc_cfg.get("socket"):
            from .music_client import MusicHoldClient
            self._music = MusicHoldClient(str(mc_cfg["socket"]))
        if not self.silent:
            self._build_output(raw.get("tts") or {})
            if self._player is None:
                # 明确要求出声但无法构造播放器 → 提示并静默降级
                logger.warning("[playback] 输出不可用，静默降级（silent）")
                self.silent = True
        self._task = asyncio.create_task(self._consumer())
        logger.info("[playback] 播报队列已启动（silent=%s）", self.silent)

    async def stop(self) -> None:
        await self._stop_wait_tone()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        if self._player is not None:
            try:
                self._player.stop()
            except Exception as exc:
                logger.warning("[playback] player stop: %s", exc)
        if self._tts is not None:
            try:
                self._tts.close()
            except Exception:
                pass
        if self._music is not None:
            self._music.close()

    def _build_output(self, tts_cfg: dict) -> None:
        from .audio_player import AudioPlayer, AudioPlayerConfig
        from .tts_engine import TencentCloudTTSEngine, TencentTTSConfig

        self._player = AudioPlayer(AudioPlayerConfig())
        self._player.start()
        t = self.cfg.tencent
        self._tts = TencentCloudTTSEngine(TencentTTSConfig(
            secret_id=t.secret_id, secret_key=t.secret_key, app_id=t.app_id,
            voice_type=tts_cfg.get("voice_type", 101001),
            codec=tts_cfg.get("codec", "pcm"),
            sample_rate=int(tts_cfg.get("sample_rate", 16000)),
            speed=float(tts_cfg.get("speed", 0.0)),
            volume=float(tts_cfg.get("volume", 10.0)),
        ))
        self._tts.on_audio_chunk = self._player.feed

    # ─── 入队（asyncio 内调用）───────────────────────────

    async def enqueue_speak(self, *, id: str, wake: str, kind: str,
                            segments: List[P.EmotionSegment],
                            turn_seq: int = 0) -> bool:
        """speak 入队。仅丢弃「同一会话更旧轮」的迟到回复，返回 False。"""
        if turn_seq:
            last = self._last_turn_by_wake.get(wake, 0)
            if turn_seq < last:
                logger.info("[playback] 丢弃迟到旧轮 speak turn=%d（%s 已到 %d）",
                            turn_seq, wake or "?", last)
                return False
            if turn_seq > last:
                self._last_turn_by_wake[wake] = turn_seq
        await self._queue.put({"type": "speak", "id": id, "wake": wake,
                               "kind": kind, "segments": list(segments),
                               "turn_seq": turn_seq})
        return True

    async def enqueue_asset(self, name: str) -> None:
        await self._queue.put({"type": "asset", "name": name})

    # ─── 线程安全入口（音频/SDK/frontend 线程调用）────────

    def play_asset(self, name: str) -> None:
        """播放资产（prompt/farewell 等）——线程安全。"""
        self._ts(self.enqueue_asset(name), "play_asset")

    def on_utterance_done(self) -> None:
        """ASR 整段结果已上行：进入等待回复（循环等待音 + 超时兜底）。"""
        self._ts(self._utterance_done(), "on_utterance_done")

    def on_interrupt(self) -> None:
        """本地打断（唤醒命中/抢话）：停 TTS、清队列、停等待音。"""
        self._ts(self._interrupt(), "on_interrupt")

    def on_conversation_timeout(self) -> None:
        """连续对话窗口超时（frontend Timer 线程）：告别语收尾。"""
        self._ts(self._conversation_timeout(), "on_conversation_timeout")

    # ─── 控制（dispatch 调用，async）──────────────────────

    async def close_window(self) -> None:
        """关对话窗口（[FINISH]/control close_window）：回唤醒词监听。"""
        await self._stop_wait_tone()
        frontend = self.get_frontend()
        if frontend is not None:
            try:
                from .voice_frontend import VoiceState
                frontend._cancel_conversation_timer()
                frontend._set_state(VoiceState.IDLE)
            except Exception as exc:
                logger.debug("[playback] close_window: %s", exc)

    async def interrupt(self) -> None:
        await self._interrupt()

    # ─── 内部实现 ────────────────────────────────────────

    def _ts(self, coro, what: str) -> None:
        if self._loop is None or self._loop.is_closed():
            return
        try:
            asyncio.run_coroutine_threadsafe(coro, self._loop)
        except Exception as exc:
            logger.warning("[playback] %s 调度失败: %s", what, exc)

    # -- 等待音 --

    async def _utterance_done(self) -> None:
        self._waiting += 1
        if self._waiting == 1:
            await self._start_wait_tone()

    async def _start_wait_tone(self) -> None:
        if self._wait_task is not None:
            return
        if self.silent or self._player is None:
            # 无声环境：只跑超时兜底，不实际放等待音
            pass
        else:
            if self._music is not None:
                self._music.hold()
            self._wait_task = asyncio.create_task(self._wait_tone_loop())
        if self._wait_timeout_task is None:
            self._wait_timeout_task = asyncio.create_task(self._wait_timeout())

    async def _wait_tone_loop(self) -> None:
        data = VA.asset_pcm16("wait_cue")
        if not data:
            logger.warning("[playback] 等待音缺失，跳过")
            return
        try:
            while True:
                if self._player is not None:
                    self._player.feed(data)
                await asyncio.sleep(len(data) / _PCM_BYTES_PER_SEC)
        except asyncio.CancelledError:
            pass

    async def _wait_timeout(self) -> None:
        timeout = self.cfg.service.wait_reply_timeout_sec
        await asyncio.sleep(timeout)
        if self._waiting <= 0 or self._playing:
            return  # 已有 speak 在播或不再等待 → 无需兜底
        logger.info("[playback] 等待回复超时（%ss），告别语收尾", timeout)
        self._waiting = 0
        await self._stop_wait_tone()
        await self._play_asset("farewell")

    async def _stop_wait_tone(self) -> None:
        self._waiting = 0
        for attr in ("_wait_task", "_wait_timeout_task"):
            task = getattr(self, attr)
            if task is not None:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                setattr(self, attr, None)
        if self._music is not None:
            self._music.release()

    # -- 打断 --

    async def _interrupt(self) -> None:
        logger.info("[playback] 打断：清播报队列 + 停 TTS/等待音")
        self._interrupted = True
        await self._stop_wait_tone()
        if self._tts is not None:
            try:
                self._tts.interrupt()
            except Exception as exc:
                logger.debug("[playback] tts.interrupt: %s", exc)
        if self._player is not None:
            self._player.clear()
        if self._queue is not None:
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

    # -- 消费者 --

    async def _consumer(self) -> None:
        assert self._queue is not None
        try:
            while True:
                job = await self._queue.get()
                try:
                    await self._handle(job)
                except Exception as exc:
                    logger.exception("[playback] 任务处理异常: %s", exc)
        except asyncio.CancelledError:
            logger.info("[playback] 播报队列已停止")

    async def _handle(self, job: Dict[str, Any]) -> None:
        if job["type"] == "asset":
            await self._play_asset(job["name"])
            return
        # speak：统一按「回合回复文本」处理 —— 不区分 final/interim。
        # 播报方式（通知音/进对话窗口 vs 保持等待）由本服务的会话状态决定
        # （有内容可念 = 用户当前这轮的回答，播完即通知音 + 连续对话窗口）。
        self._interrupted = False
        await self._stop_wait_tone()      # 开始播报：停等待音

        kind = job.get("kind")
        segments = job.get("segments") or []
        texts = [t for _, t in segments if t and t.strip()]
        ok = True
        if texts:
            self._playing = True
            try:
                if self.silent or self._tts is None or self._player is None:
                    logger.info("[playback] 🔊 播报(silent, %s): %d 段",
                                job.get("wake"), len(texts))
                else:
                    logger.info("[playback] 🔊 播报(%s): %d 段",
                                job.get("wake"), len(texts))
                    ok = await asyncio.to_thread(
                        self._play_tts_sync, job.get("id"), segments)
            finally:
                self._playing = False

        # speak_done 上行
        await self.server.broadcast_event(P.EVT_SPEAK_DONE, {
            "id": job.get("id", ""), "kind": kind, "ok": ok})

        if not ok or self._interrupted or not texts:
            return

        # 播完本轮回答：通知音 + 进入连续对话窗口（用户可直接接着说，无需再唤醒）
        await self._play_asset("notification")
        frontend = self.get_frontend()
        if frontend is not None:
            frontend.enter_conversation_window()

    # -- 播放（同步阻塞部分，to_thread 执行）--

    def _play_tts_sync(self, speak_id: str,
                       segments: List[P.EmotionSegment]) -> bool:
        """逐段按情绪流式合成播放（同步，to_thread 中执行）。"""
        assert self._tts is not None and self._player is not None
        frontend = self.get_frontend()
        if self._music is not None:
            self._music.hold()
        if frontend is not None:
            frontend.set_tts_playing(True)
        try:
            for emotion, text in segments:
                if not text or not text.strip():
                    continue
                if self._interrupted:
                    break
                self._tts.on_audio_chunk = self._player.feed
                self._tts.start()
                if emotion:
                    self._tts.set_emotion(emotion)
                self._tts.synthesize(text)
                self._tts.complete()
                self._tts.wait(timeout=15)
            self._player.wait_for_drain(timeout=15.0)
            time.sleep(0.15)
            return True
        except Exception as exc:
            logger.warning("[playback] TTS 播报异常(speak %s): %s", speak_id, exc)
            return False
        finally:
            try:
                self._player.clear()
            except Exception:
                pass
            if self._music is not None:
                self._music.release()
            if frontend is not None:
                frontend.set_tts_playing(False)

    async def _play_asset(self, name: str) -> None:
        if self.silent or self._player is None:
            logger.info("[playback] 🔊 资产(silent): %s", name)
            return
        data = VA.asset_pcm16(name)
        if not data:
            return
        logger.info("[playback] 🔊 资产: %s", name)
        self._player.feed(data)
        await asyncio.to_thread(self._player.wait_for_drain, timeout=5.0)

    # -- 会话收尾 --

    async def _conversation_timeout(self) -> None:
        await self._stop_wait_tone()
        await self._play_asset("farewell")
