#!/usr/bin/env python3
"""
Hermes Agent 语音输入前端 — 入口点。

整合：
  - Sherpa-onnx唤醒词 + WebRTC VAD
  - 腾讯云实时语音识别（ASR SDK）
  - Hermes Agent（Python 库，嵌入运行）
  - 腾讯云流式语音合成（TTS SDK）
  - 唤醒词打断 TTS + 代理对话

用法:
    python -u src/main.py
"""

import asyncio
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# 确保 src 和 hermes-agent 可导入
sys.path.insert(0, str(Path(__file__).resolve().parent))

from asr_engine import TencentCloudASREngine, TencentASRConfig
from audio_player import AudioPlayer, AudioPlayerConfig
from config import load_config
from tts_engine import TencentCloudTTSEngine, TencentTTSConfig
from voice_frontend import VoiceFrontend, VoiceFrontendConfig

# Hermes Agent（嵌入运行）
from run_agent import AIAgent

logger = logging.getLogger("main")


class VoiceApp:
    """
    语音应用主控。
    唤醒词 → VAD → ASR → Hermes Agent → TTS 全链路。
    """

    def __init__(self, config_path: str | None = None):
        self._full_cfg = load_config(config_path)
        self.config = self._full_cfg["voice"]
        self._running = False

        # 组件
        self.asr_engine: TencentCloudASREngine | None = None
        self.tts_engine: TencentCloudTTSEngine | None = None
        self.audio_player: AudioPlayer | None = None
        self.frontend: VoiceFrontend | None = None

        # 多 AI Agent 实例：{name: (agent, executor, history, generation)}
        # 每个 agent 有独享的 ThreadPoolExecutor、会话历史和代际计数器
        self._agents: dict[str, tuple] = {}
        self._active_name: str | None = None  # 当前由哪个唤醒词触发
        self._loop: asyncio.AbstractEventLoop | None = None

        # 待汇报队列：agent 完成但当前有其他 agent 活跃时排队
        # 元素: (keyword, response_text, gen)
        self._pending: list[tuple[str, str, int]] = []
        self._user_title = self.config.get("user_title", "主人")

    # ─── 生命周期 ────────────────────────────────────────

    async def start(self):
        self._loop = asyncio.get_running_loop()

        logger.info("=" * 50)
        logger.info("Hermes Agent Voice Frontend")
        logger.info("=" * 50)

        tencent_cfg = self._full_cfg.get("tencent", {})

        # 1. ASR 引擎
        asr_config = TencentASRConfig(
            secret_id=tencent_cfg.get("secret_id", ""),
            secret_key=tencent_cfg.get("secret_key", ""),
            app_id=tencent_cfg.get("app_id", ""),
            engine_model=self.config["asr"]["engine_model"],
            needvad=self.config["asr"]["needvad"],
            voice_format=self.config["asr"]["voice_format"],
        )
        self.asr_engine = TencentCloudASREngine(asr_config)
        self.asr_engine.on_start = self._on_asr_start
        self.asr_engine.on_interim = self._on_asr_interim
        # 不绑定 on_final：服务端 VAD 可能在用户句间停顿时就触发 on_sentence_end，
        # 导致 agent 在用户未说完时就开始回答。仅在 on_complete 后由本地 VAD 控制时机。
        self.asr_engine.on_final = None
        self.asr_engine.on_complete = self._on_asr_complete
        self.asr_engine.on_error = self._on_asr_error

        # 2. TTS 引擎
        tts_cfg = self.config.get("tts", {})
        if tts_cfg.get("enabled", True):
            tts_config = TencentTTSConfig(
                secret_id=tencent_cfg.get("secret_id", ""),
                secret_key=tencent_cfg.get("secret_key", ""),
                app_id=tencent_cfg.get("app_id", ""),
                voice_type=tts_cfg.get("voice_type", 101001),
                codec=tts_cfg.get("codec", "pcm"),
                sample_rate=tts_cfg.get("sample_rate", 16000),
                speed=tts_cfg.get("speed", 0.0),
                volume=tts_cfg.get("volume", 0.0),
            )
            self.tts_engine = TencentCloudTTSEngine(tts_config)

        # 3. 音频播放器（on_audio_write 在 frontend 初始化后绑定）
        self.audio_player = AudioPlayer(AudioPlayerConfig())
        self.audio_player.start()
        if self.tts_engine:
            # TTS 音频 → 播放器
            self.tts_engine.on_audio_chunk = self.audio_player.feed

        # 4. 初始化 AI Agent 实例（每个唤醒词一个，每个有独享线程池）
        agents_config = self.config.get("agents", [])

        for ac in agents_config:
            name = ac["name"]
            agent = AIAgent(
                model=ac.get("model", "openai/gpt-4o-mini"),
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
                max_iterations=ac.get("max_iterations", 10),
                session_id=ac.get("session_id", f"hermes-{name}"),
                ephemeral_system_prompt=ac.get("system_prompt", ""),
                reasoning_config={'enabled': False},
            )
            executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=name)
            self._agents[name] = (agent, executor, None, 0)  # (agent, executor, history, generation)
            logger.info("Agent [%s]: model=%s, session=%s",
                        name, ac.get("model"), ac.get("session_id"))

        logger.info("Total agents: %d", len(self._agents))

        # 5. 语音前端
        first_agent_name = agents_config[0]["name"] if agents_config else "赫尔墨斯"
        frontend_config = VoiceFrontendConfig(
            wake_word_enabled=self.config["wake_word"]["enabled"],
            wake_word_keyword=first_agent_name,
            wake_word_threshold=self.config["wake_word"].get("threshold", 0.25),
            wake_word_score=self.config["wake_word"].get("score", 1.0),
            # KWS 模型路径和文件名（从 config.yaml 读取）
            kws_model_dir=self.config["wake_word"].get("model", {}).get("dir", "models/sherpa-kws"),
            kws_model_name=self.config["wake_word"].get("model", {}).get("name", "sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20"),
            kws_encoder=self.config["wake_word"].get("model", {}).get("encoder", "encoder-epoch-13-avg-2-chunk-8-left-64.int8.onnx"),
            kws_decoder=self.config["wake_word"].get("model", {}).get("decoder", "decoder-epoch-13-avg-2-chunk-8-left-64.onnx"),
            kws_joiner=self.config["wake_word"].get("model", {}).get("joiner", "joiner-epoch-13-avg-2-chunk-8-left-64.int8.onnx"),
            kws_tokens=self.config["wake_word"].get("model", {}).get("tokens", "tokens.txt"),
            vad_mode=self.config["vad"]["mode"],
            min_speech_ms=self.config["vad"]["min_speech_ms"],
            vad_silence_threshold_ms=self.config["vad"].get("silence_threshold_ms", 600),
            vad_speech_confirm_frames=self.config["vad"].get("speech_confirm_frames", 3),
            conversation_window_sec=self.config.get("conversation_window", {}).get("timeout_sec", 8.0),
            on_interim=self._on_frontend_interim,
            on_final=self._on_frontend_final,
            on_error=self._on_frontend_error,
            on_state_change=self._on_frontend_state_change,
            on_wake_word=self._on_wake_word,
            on_interrupt_request=self._on_interrupt_request,
            prompt_duration_sec=self._get_prompt_duration_sec(),
            on_play_prompt=self._on_play_prompt,
        )
        self.frontend = VoiceFrontend(frontend_config, self.asr_engine)
        self.frontend.start()

        self._running = True
        logger.info("=" * 50)
        logger.info("Ready — say the wake word to start")
        logger.info(f"  Wake words: {list(self._agents.keys())}")
        logger.info(f"  ASR model : {asr_config.engine_model}")
        logger.info(f"  Agents    : {len(self._agents)} ({', '.join(self._agents.keys())})")
        logger.info(f"  TTS voice : {tts_config.voice_type}")
        logger.info("=" * 50)

    async def stop(self):
        self._running = False
        if self.frontend:
            self.frontend.stop()
        if self.audio_player:
            self.audio_player.stop()
        for name, (agent, executor, history, gen) in self._agents.items():
            executor.shutdown(wait=False)
        logger.info("Voice frontend stopped")

    # ─── 核心流程 ────────────────────────────────────────

    def _ask_agent(self, text: str, name: str):
        """
        在后台线程中用对应唤醒词的 Hermes Agent 问答。

        如果 agent 仍是当前活跃的，完成时立即 TTS；
        如果已被其他唤醒词取代，结果进入待汇报队列排队，
        等待当前连续对话窗口过期后递送。
        """
        entry = self._agents.get(name)
        if entry is None:
            logger.error("No agent for name: %s", name)
            return

        agent, executor, history, gen = entry

        def _chat():
            nonlocal history
            try:
                agent._interrupt_requested = False

                logger.info(f"(gen={gen}) [{name}] Asking: {text[:60]}...")
                result = agent.run_conversation(
                    user_message=text,
                    conversation_history=history,
                )
                history = result["messages"]
                self._agents[name] = (agent, executor, history, gen)
                response = result.get("final_response", "")
                logger.info(f"(gen={gen}) [{name}] Response: {response[:80]}...")

                # 取当前代际，判断结果是否仍有效
                _, _, _, current_gen = self._agents.get(name, (None, None, None, -1))
                if gen != current_gen:
                    logger.info(f"🚫 [{name}] gen {gen} != current {current_gen}, stale")
                    return

                if name == self._active_name:
                    asyncio.run_coroutine_threadsafe(
                        self._speak(response), self._loop
                    )
                else:
                    self._pending.append((name, response, gen))
                    logger.info(f"📥 [{name}] queued (pending={len(self._pending)})")
            except Exception as e:
                logger.error(f"Agent [{name}] chat error: {e}")

        executor.submit(_chat)

    async def _speak(self, text: str):
        """TTS 合成并播放。代际检查由调用方负责。"""
        if not text.strip() or not self.tts_engine:
            return

        logger.info(f"🔊 TTS: {text[:60]}...")
        try:
            if self.frontend:
                self.frontend.set_tts_playing(True)
            self.tts_engine.start()
            self.tts_engine.synthesize(text)
            self.tts_engine.complete()
            self.tts_engine.wait(timeout=15)
            if self.audio_player:
                self.audio_player.wait_for_drain(timeout=15.0)
                import time as _t
                _t.sleep(0.15)
            self._play_asset("notification")
            if self.audio_player:
                self.audio_player.wait_for_drain(timeout=5.0)
                _t.sleep(0.05)
            if self.frontend:
                self.frontend.enter_conversation_window()
        except Exception as e:
            logger.warning(f"TTS error: {e}")
        finally:
            if self.frontend:
                self.frontend.set_tts_playing(False)
        # TTS 播放完毕后尝试递送队列中的待汇报结果
        self._deliver_pending()

    def _deliver_pending(self):
        """
        递送队列中的待汇报结果。

        条件：frontend 处于 IDLE 状态（无 ASR、TTS、连续对话），
        且队列非空。
        格式："{user_title}，我是{keyword}，{text}"
        """
        if not self._pending:
            return
        if self.frontend and self.frontend.state.value != "idle":
            return

        kw, text, gen = self._pending.pop(0)
        # 检查该 agent 的代际是否仍匹配
        _, _, _, current_gen = self._agents.get(kw, (None, None, None, -1))
        if gen != current_gen:
            logger.info(f"🚫 Pending [{kw}] gen {gen} != current {current_gen}, discard")
            self._deliver_pending()  # 尝试下一个
            return

        greeting = f"{self._user_title}，我是{kw}，{text}"
        logger.info("📤 Delivering pending result from [%s]: %s...", kw, text[:60])
        asyncio.run_coroutine_threadsafe(
            self._speak(greeting), self._loop
        )

    def _cancel_current_output(self):
        """打断当前输出（TTS + 当前活跃 agent 的工具调用）。"""
        logger.info("⏹ Interrupting current output")

        # 打断当前活跃 agent
        if self._active_name:
            entry = self._agents.get(self._active_name)
            if entry:
                agent, _, _, _ = entry
                try:
                    agent.interrupt()
                except Exception as e:
                    logger.debug(f"Agent interrupt: {e}")

        if self.tts_engine:
            self.tts_engine.interrupt()
        if self.audio_player:
            self.audio_player.clear()

    # ─── 回调 ────────────────────────────────────────────

    PROMPT_PATH = Path(__file__).resolve().parent.parent / "assets" / "prompt.wav"
    NOTIFICATION_PATH = Path(__file__).resolve().parent.parent / "assets" / "notification.wav"
    FAREWELL_PATH = Path(__file__).resolve().parent.parent / "assets" / "farewell.wav"

    def _get_prompt_duration_sec(self) -> float:
        """读取提示音 wav 的实际时长（秒）。"""
        try:
            import wave
            with wave.open(str(self.PROMPT_PATH), "rb") as wf:
                frames = wf.getnframes()
                sr = wf.getframerate()
                return frames / sr
        except Exception:
            return 1.0

    def _play_asset(self, name: str):
        """播放 assets 目录下的 WAV 文件。"""
        path = {"prompt": self.PROMPT_PATH,
                "notification": self.NOTIFICATION_PATH,
                "farewell": self.FAREWELL_PATH}.get(name)
        if not path or not path.exists():
            return
        try:
            import wave
            with wave.open(str(path), "rb") as wf:
                data = wf.readframes(wf.getnframes())
            if self.audio_player:
                self.audio_player.feed(data)
        except Exception as e:
            logger.warning(f"Play {name} error: {e}")

    def _on_play_prompt(self):
        """播唤醒提示音。"""
        self._play_asset("prompt")

    def _on_wake_word(self, name: str):
        """唤醒词命中：记录活跃 name，递增该 agent 代际。"""
        self._active_name = name
        # 递增该 agent 的代际（旧结果自动失效）
        entry = self._agents.get(name)
        if entry:
            agent, executor, history, gen = entry
            self._agents[name] = (agent, executor, history, gen + 1)
            logger.info("🔊 Wake word: '%s' (gen=%d)", name, gen + 1)
        else:
            logger.info("🔊 Wake word: '%s' (no agent)", name)

    def _on_interrupt_request(self):
        """VoiceFrontend 在唤醒词命中时首先调用此回调。"""
        self._cancel_current_output()

    def _on_frontend_state_change(self, state):
        logger.debug(f"Frontend: {state.value}")
        # 连续对话窗口过期 → IDLE → 递送待汇报结果
        if state.value == "idle":
            self._deliver_pending()

    def _on_frontend_interim(self, text: str):
        pass

    def _on_frontend_final(self, text: str):
        if not text.strip():
            return
        name = self._active_name
        if name is None:
            logger.warning("No active agent for ASR result")
            return
        logger.info(f"📝 [{name}] User said: {text}")
        self._ask_agent(text, name)

    def _on_frontend_error(self, msg: str):
        logger.error(f"❌ Frontend: {msg}")

    # ASR 回调

    def _on_asr_start(self):
        logger.debug("ASR started")

    def _on_asr_complete(self):
        logger.debug("ASR complete")
        # 兜底：如果 on_final 未被 SDK 回调，从 last_text 补发
        if self.asr_engine and self.asr_engine.last_text.strip():
            if self.frontend and self.frontend.config.on_final:
                self.frontend.config.on_final(self.asr_engine.last_text)
        else:
            # ASR 返回空文本 → 播放告别语
            self._play_asset("farewell")

    def _on_asr_interim(self, text: str):
        if self.frontend and self.frontend.config.on_interim:
            self.frontend.config.on_interim(text)

    def _on_asr_final(self, text: str):
        if self.frontend and self.frontend.config.on_final:
            self.frontend.config.on_final(text)

    def _on_asr_error(self, msg: str):
        logger.error(f"ASR: {msg}")
        if self.frontend and self.frontend.config.on_error:
            self.frontend.config.on_error(msg)


# ─── 入口 ────────────────────────────────────────────────

async def main():
    app = VoiceApp()
    try:
        await app.start()
        while app._running:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("\nShutting down...")
    finally:
        await app.stop()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    asyncio.run(main())
