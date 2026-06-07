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
import threading
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
        self.hermes_agent: AIAgent | None = None

        # 线程池（agent.chat 是同步阻塞的）
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hermes")
        self._loop: asyncio.AbstractEventLoop | None = None

        # 代际计数器：每次唤醒词递增，旧结果的 speak 自动丢弃
        self._generation = 0
        self._gen_lock = threading.Lock()
        # 会话历史
        self.history = None

    # ─── 生命周期 ────────────────────────────────────────

    async def start(self):
        self._loop = asyncio.get_running_loop()

        logger.info("=" * 50)
        logger.info("Hermes Agent Voice Frontend")
        logger.info("=" * 50)

        tencent_cfg = self._full_cfg.get("tencent", {})
        hermes_cfg = self.config.get("hermes_agent", {})

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
        self.asr_engine.on_final = self._on_asr_final
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

        # 4. Hermes Agent（单实例，保持 session + 记忆）
        model = hermes_cfg.get("model", "openai/gpt-4o-mini")
        self._agent_session_id = hermes_cfg.get("session_id", "hermes-voice-session")
        self._agent_system_prompt = hermes_cfg.get(
            "system_prompt",
            "你是一个语音助手。\n\n"
            "重要规则：\n"
            "1. 用户的输入来自语音识别（ASR），可能存在同音字、漏字、多字等错误。\n"
            "   如果问题听起来不合逻辑，结合上下文做合理推断。\n"
            "2. 回答要简洁，控制在 3 句话以内。\n"
            "3. 回答中自然融入确认，不需要生硬复述。\n"
            "4. 如果实在听不懂，直接说「不好意思没听清，能再说一遍吗？」"
        )
        self.hermes_agent = AIAgent(
            model=model,
            quiet_mode=True,
            skip_context_files=True,   # 启用 AGENTS.md
            skip_memory=True,           # 启用长期记忆
            max_iterations=hermes_cfg.get("max_iterations", 10),
            session_id=self._agent_session_id,  # 固定会话
            ephemeral_system_prompt=self._agent_system_prompt,
            reasoning_config={'enabled': False},
        )
        logger.info(f"Hermes Agent initialized: model={model}, session={self._agent_session_id}")

        # 5. 语音前端
        frontend_config = VoiceFrontendConfig(
            wake_word_enabled=self.config["wake_word"]["enabled"],
            wake_word_keyword=self.config["wake_word"]["keyword"],
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
        logger.info(f"  Wake word : '{self.config['wake_word']['keyword']}'")
        logger.info(f"  ASR model : {asr_config.engine_model}")
        logger.info(f"  Agent model: {model}")
        logger.info(f"  TTS voice : {tts_config.voice_type}")
        logger.info("=" * 50)

    async def stop(self):
        self._running = False
        if self.frontend:
            self.frontend.stop()
        if self.audio_player:
            self.audio_player.stop()
        self._executor.shutdown(wait=False)
        logger.info("Voice frontend stopped")

    # ─── 核心流程 ────────────────────────────────────────

    def _ask_hermes(self, text: str):
        """
        在后台线程中用 Hermes Agent 问答（单实例，保持 session + 记忆）。

        使用固定 session_id 保持对话连续性。
        系统提示词说明输入来自语音识别，要求复述确认 + 简洁回答。
        代际计数器防止旧结果在打断后播报。
        """
        with self._gen_lock:
            gen = self._generation

        if self.hermes_agent is None:
            logger.error("Hermes Agent not initialized")
            return

        def _chat():
            try:
                # 清除上一轮 interrupt 残留
                self.hermes_agent._interrupt_requested = False

                logger.info(f"🤔 (gen={gen}) Asking Hermes: {text[:60]}...")
                result = self.hermes_agent.run_conversation(
                    user_message=text,
                    conversation_history=self.history,
                )
                self.history = result["messages"]
                response = result.get("final_response", "")
                logger.info(f"🤖 (gen={gen}) Hermes: {response[:80]}...")
                asyncio.run_coroutine_threadsafe(
                    self._speak(response, gen), self._loop
                )
            except Exception as e:
                logger.error(f"Hermes chat error: {e}")

        self._executor.submit(_chat)

    async def _speak(self, text: str, gen: int):
        """TTS 合成并播放。若代际已过时则静默丢弃。"""
        if not text.strip() or not self.tts_engine:
            return

        # 检查代际是否仍有效
        with self._gen_lock:
            if gen != self._generation:
                logger.info(f"🚫 (gen={gen}) discarding stale TTS (current={self._generation})")
                return

        logger.info(f"🔊 (gen={gen}) TTS: {text[:60]}...")
        try:
            # 标记 TTS 播放中 → 前段不送 ASR
            if self.frontend:
                self.frontend.set_tts_playing(True)
            self.tts_engine.start()
            self.tts_engine.synthesize(text)
            self.tts_engine.complete()
            self.tts_engine.wait(timeout=15)
            # 等 AudioPlayer 真正播完
            if self.audio_player:
                self.audio_player.wait_for_drain(timeout=15.0)
                # sounddevice 硬件缓冲还有 ~blocksize 的延迟
                import time as _t
                _t.sleep(0.15)
            # 提示音（非人声），表示可以继续说话
            # 等提示音播完再开对话窗口，防止麦克风采集到提示音触发 VAD
            self._play_asset("notification")
            if self.audio_player:
                self.audio_player.wait_for_drain(timeout=5.0)
                _t.sleep(0.05)
            if self.frontend:
                self.frontend.enter_conversation_window()
        except Exception as e:
            logger.warning(f"TTS error: {e}")
        finally:
            # 无论 TTS 完成还是被打断，都恢复门控
            if self.frontend:
                self.frontend.set_tts_playing(False)

    def _cancel_current_output(self):
        """
        打断当前输出（仅停 TTS + 停工具，不丢弃 agent 结果）。
        唤醒词命中时调用。
        """
        logger.info(f"⏹ Interrupting current output")

        # 打断正在进行的 agent 工具调用
        if self.hermes_agent:
            try:
                self.hermes_agent.interrupt()
            except Exception as e:
                logger.debug(f"Agent interrupt: {e}")

        # 打断 TTS
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

    def _on_wake_word(self):
        logger.info("🔊 Wake word! (interrupt handled by on_interrupt_request)")

    def _on_interrupt_request(self):
        """VoiceFrontend 在唤醒词命中时首先调用此回调。"""
        self._cancel_current_output()

    def _on_frontend_state_change(self, state):
        logger.debug(f"Frontend: {state.value}")

    def _on_frontend_interim(self, text: str):
        pass

    def _on_frontend_final(self, text: str):
        if not text.strip():
            return
        logger.info(f"📝 User said: {text}")
        # 直接调用嵌入的 Hermes Agent
        self._ask_hermes(text)

    def _on_frontend_error(self, msg: str):
        logger.error(f"❌ Frontend: {msg}")

    # ASR 回调

    def _on_asr_start(self):
        logger.debug("ASR started")

    def _on_asr_complete(self):
        logger.debug("ASR complete")
        # ASR 返回空文本 → 播放告别语
        if self.asr_engine and not self.asr_engine.last_text.strip():
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
