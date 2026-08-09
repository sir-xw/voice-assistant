#!/usr/bin/env python3
"""
Hermes Agent 语音输入前端 — VoiceApp 应用逻辑。

整合：
  - Sherpa-onnx唤醒词 + WebRTC VAD
  - 腾讯云实时语音识别（ASR SDK）
  - Hermes Agent（Python 库，嵌入运行）
  - 腾讯云流式语音合成（TTS SDK）
  - 唤醒词打断 TTS + 代理对话
  - AI 自主 speak 工具（阶段性汇报 + 最终回答）

本模块只定义 VoiceApp，不包含启动入口。
启动方式见 __main__.py（python -m voice_agent）或项目根启动脚本。
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .asr_engine import TencentCloudASREngine, TencentASRConfig
from .audio_player import AudioPlayer, AudioPlayerConfig
from .config import load_config
from .tts_engine import TencentCloudTTSEngine, TencentTTSConfig
from .music_control import player_pause, player_resume
from .voice_frontend import VoiceFrontend, VoiceFrontendConfig, VoiceState

# Hermes Agent（嵌入运行）
from run_agent import AIAgent

# AI 工具注册
from tools.registry import registry
from .mpd_tool import register_all as register_mpd_tools

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

        # speak 工具队列：AI 通过工具发起的语音播报请求
        # 元素: (text, emotion)
        self._speak_queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue()
        self._speak_consumer_task: asyncio.Task | None = None

        # 待汇报队列：agent 完成但当前有其他 agent 活跃时排队
        # 元素: (keyword, response_text, gen, emotion)
        self._pending: list[tuple[str, str, int, str]] = []
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

        # 3. 音频播放器
        self.audio_player = AudioPlayer(AudioPlayerConfig())
        self.audio_player.start()
        if self.tts_engine:
            # TTS 音频 → 播放器
            self.tts_engine.on_audio_chunk = self.audio_player.feed

        # 4. 注册所有工具（在 agent 创建之前，确保工具可用）
        self._register_speak_tool()
        register_mpd_tools()

        # ─── 系统提示词（语音助手通用规则 + speak 工具 + 播放控制规则）─
        speak_prompt = (
            '\n\n【语音助手通用规则】\n'
            '你是一个语音助手。\n'
            '1. 用户的输入来自语音识别（ASR），可能存在同音字、漏字、多字等错误。\n'
            '   如果问题听起来不合逻辑，结合上下文做合理推断，而不是逐字照搬。\n'
            '2. 回答要简洁，控制在 3 句话以内。\n'
            '   需要列举时用「第一、第二、第三」代替长段落。\n'
            '3. 回答中自然融入确认——不是生硬复述，而是把确认编织在回答里。\n'
            '   例如用户说「今天天气怎么样」，不要说「你是问今天天气吗？」\n'
            '   直接说「今天晴天，25度，适合出门。」\n'
            '   如果确实听清了，不需要额外确认。\n'
            '4. 如果实在听不懂，直接说「不好意思没听清，能再说一遍吗？」\n'
            '\n'
            '【语音播报规则】\n'
            '你拥有 speak 工具，可以通过语音播报与用户实时交流。\n'
            '- 阶段性计划或过程信息（如「正在搜索网络」）→ 并行调用 speak(text="...") 来播报\n'
            '- 不要在 speak 工具中输出最终回答。最终回答请直接以文字格式回复，格式为：(情绪)你要说的话\n'
            '  例如：(happy)你好，有什么可以帮助你的？\n'
            '  情绪可选值：neutral(中性) sad(悲伤) happy(高兴) angry(生气) fear(恐惧) '
            'story(故事) poetry(诗歌) sajiao(撒娇) disgusted(厌恶) amaze(震惊) exciting(兴奋) '
            'aojiao(傲娇) jieshuo(解说)\n'
            '- 如果你没有用到情绪，可以用 neutral，例如：(neutral)这是回答\n'
            '\n'
            '【播放控制工具规则】\n'
            '你拥有 mpd_ 系列工具用于控制音乐播放（如 mpd_play、mpd_pause、mpd_stop、\n'
            'mpd_previous、mpd_next 等）。\n'
            '- 当你使用播放控制工具时，执行的同时允许并行调用 speak 工具，播放控制工具执行后直接返回 [FINISH] 作为文字回复，\n'
            '  该条文字回复不要使用 (情绪)格式。\n'
            '- [FINISH] 表示操作已完成，系统会自动关闭对话窗口，\n'
            '  用户可以通过再次说唤醒词来继续对话。'
        )

        # 5. 初始化 AI Agent 实例（每个唤醒词一个，每个有独享线程池）
        agents_config = self.config.get("agents", [])

        for ac in agents_config:
            name = ac["name"]
            merged_prompt = (ac.get("system_prompt", "") + speak_prompt)
            agent = AIAgent(
                model=ac.get("model", "openai/gpt-4o-mini"),
                quiet_mode=False,
                skip_context_files=True,
                skip_memory=True,
                max_iterations=ac.get("max_iterations", 10),
                session_id=ac.get("session_id", f"hermes-{name}"),
                ephemeral_system_prompt=merged_prompt,
                reasoning_config={'enabled': False},
            )
            executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=name)
            self._agents[name] = (agent, executor, None, 0)  # (agent, executor, history, generation)
            logger.info("Agent [%s]: model=%s, session=%s",
                        name, ac.get("model"), ac.get("session_id"))

        logger.info("Total agents: %d", len(self._agents))

        # 6. 语音前端
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
            on_conversation_timeout=lambda: self._play_asset("farewell"),
        )
        self.frontend = VoiceFrontend(frontend_config, self.asr_engine)
        self.frontend.start()

        # 7. 启动 speak 队列消费者
        self._speak_consumer_task = asyncio.create_task(self._speak_queue_consumer())

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

        # 取消 speak 队列消费者
        if self._speak_consumer_task:
            self._speak_consumer_task.cancel()
            try:
                await self._speak_consumer_task
            except asyncio.CancelledError:
                pass
            self._speak_consumer_task = None

        if self.frontend:
            self.frontend.stop()
        if self.audio_player:
            self.audio_player.stop()
        for name, (agent, executor, history, gen) in self._agents.items():
            executor.shutdown(wait=False)
        logger.info("Voice frontend stopped")

    # ─── speak 工具注册 ──────────────────────────────────

    def _register_speak_tool(self):
        """注册 speak 工具，供 AI Agent 调用进行语音播报。"""
        SPEAK_SCHEMA = {
            "name": "speak",
            "description": "用语音播报阶段性进展/过程信息给用户听。"
                           "注意：最终回答不要使用此工具，请直接以 (情绪)文字 格式返回文字回复。",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "要语音播报的文本内容"
                    },
                    "emotion": {
                        "type": "string",
                        "enum": [
                            "neutral", "sad", "happy", "angry", "fear",
                            "story", "poetry", "sajiao", "disgusted", "amaze",
                            "exciting", "aojiao", "jieshuo"
                        ],
                        "description": "语音情绪/风格。"
                                       "neutral(中性) sad(悲伤) happy(高兴) angry(生气) fear(恐惧) "
                                       "story(故事) poetry(诗歌) sajiao(撒娇) disgusted(厌恶) "
                                       "amaze(震惊) exciting(兴奋) aojiao(傲娇) jieshuo(解说)"
                    }
                },
                "required": ["text"]
            }
        }

        def _speak_handler(args, **kw):
            text = args.get("text", "")
            emotion = args.get("emotion", "")
            if not text.strip():
                return "empty text"
            # speak 工具仅用于阶段性汇报
            asyncio.run_coroutine_threadsafe(
                self._speak_queue.put((text, emotion)),
                self._loop,
            )
            return f"speak queued: {text[:40]}..."

        registry.register(
            name="speak",
            toolset="voice_agent",
            schema=SPEAK_SCHEMA,
            handler=_speak_handler,
        )
        logger.info("speak tool registered")

    # ─── speak 队列消费者 ────────────────────────────────

    async def _speak_queue_consumer(self):
        """
        后台消费 speak 队列，逐条进行阶段性汇报的 TTS 播报。
        """
        try:
            while True:
                text, emotion = await self._speak_queue.get()
                await self._speak_raw(text, emotion=emotion)
        except asyncio.CancelledError:
            logger.info("Speak consumer cancelled")

    async def _speak_raw(self, text: str, emotion: str = ""):
        """
        纯 TTS 合成播放（阶段性汇报使用，不播通知音，不进对话窗口）。

        Args:
            text: 要播报的文本
            emotion: 情绪/风格，如 "happy", "sad", "angry" 等，空字符串表示不指定
        """
        if not text.strip() or not self.tts_engine:
            return

        logger.info(f"🔊 Speak: {text[:60]}...")
        player_pause(force=True)
        try:
            if self.frontend:
                self.frontend.set_tts_playing(True)
            self.tts_engine.start()
            if emotion:
                self.tts_engine.set_emotion(emotion)
            self.tts_engine.synthesize(text)
            self.tts_engine.complete()
            self.tts_engine.wait(timeout=15)
            if self.audio_player:
                self.audio_player.wait_for_drain(timeout=15.0)
                import time as _t
                _t.sleep(0.15)
        except Exception as e:
            logger.warning(f"Speak error: {e}")
        finally:
            if self.frontend:
                self.frontend.set_tts_playing(False)
            player_resume()

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

                # 事先读取当前代际作为基线，run_conversation 执行期间若被打断会递增
                _, _, _, start_gen = self._agents.get(name, (None, None, None, -1))

                result = agent.run_conversation(
                    user_message=text,
                    conversation_history=history,
                )
                history = result["messages"]
                response = result.get("final_response", "").strip()
                logger.info(f"(gen={gen}) [{name}] Response: {response[:80]}...")

                # 读取当前代际
                _, _, _, current_gen = self._agents.get(name, (None, None, None, -1))

                # 基线变了说明 run_conversation 期间被 interrupt，结果已失效
                if start_gen != current_gen:
                    logger.info(f"🚫 [{name}] gen {start_gen} → {current_gen}, interrupted, skip")
                    return

                # 代际未变，安全写回 history
                self._agents[name] = (agent, executor, history, current_gen)

                # agent.interrupt() 后原线程收尾返回的回复以 "Operation interrupted:" 开头，忽略
                if response.startswith("Operation interrupted:"):
                    logger.info(f"⏹ [{name}] Interrupted response, skip")
                    return

                # 播放控制类工具执行后的回复标记 [FINISH]，跳过 TTS 并关闭对话窗口
                if response.upper().strip("[]") == "FINISH":
                    logger.info(f"✅ [{name}] FINISH signal, close conversation window")
                    if name == self._active_name and self.frontend:
                        asyncio.run_coroutine_threadsafe(
                            self._close_conversation(), self._loop
                        )
                    return

                # 解析 (情绪)文字内容 格式
                emotion, speak_text = self._parse_emotion_response(response)
                logger.info(f"(gen={gen}) [{name}] Parsed: emotion={emotion}, text={speak_text[:60]}")
                if not speak_text:
                    logger.info(f"✅ [{name}] Empty response after parse, skip")
                    return

                if name == self._active_name:
                    asyncio.run_coroutine_threadsafe(
                        self._speak_response(speak_text, emotion), self._loop
                    )
                else:
                    self._pending.append((name, speak_text, gen, emotion))
                    logger.info(f"📥 [{name}] queued (pending={len(self._pending)})")
            except Exception as e:
                logger.error(f"Agent [{name}] chat error: {e}")

        executor.submit(_chat)

    async def _speak(self, text: str):
        """TTS 合成并播放（完整流程，含通知音和对话窗口）。代际检查由调用方负责。"""
        if not text.strip() or not self.tts_engine:
            return

        logger.info(f"🔊 TTS: {text[:60]}...")
        # TTS 播放前强制暂停音乐（即使 agent 执行期间用户恢复了播放）
        player_pause(force=True)
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
            # TTS 播放完毕（含异常情况）恢复音乐
            player_resume()
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

        kw, text, gen, emotion = self._pending.pop(0)
        # 检查该 agent 的代际是否仍匹配
        _, _, _, current_gen = self._agents.get(kw, (None, None, None, -1))
        if gen != current_gen:
            logger.info(f"🚫 Pending [{kw}] gen {gen} != current {current_gen}, discard")
            self._deliver_pending()  # 尝试下一个
            return

        greeting = f"{self._user_title}，我是{kw}，{text}"
        logger.info("📤 Delivering pending result from [%s]: %s...", kw, text[:60])
        asyncio.run_coroutine_threadsafe(
            self._speak_response(greeting, emotion), self._loop
        )

    # ─── 情绪解析工具 ────────────────────────────────────

    def _parse_emotion_response(self, response: str) -> tuple[str, str]:
        """
        解析 AI 最终回复中的 (情绪)文字内容 格式。

        Args:
            response: AI 的文字回复

        Returns:
            (emotion, text) 元组。
            emotion 为空表示未指定情绪。
            如果格式不匹配，emotion="" 且 text=response。
        """
        response = response.strip().replace('（', '(').replace('）',')')
        if response.startswith("("):
            close_idx = response.find(")")
            if close_idx > 0:
                emotion = response[1:close_idx].strip().lower()
                text = response[close_idx+1:].strip()
                # 验证情绪是否有效
                valid_emotions = {
                    "neutral", "sad", "happy", "angry", "fear",
                    "story", "poetry", "sajiao", "disgusted", "amaze",
                    "exciting", "aojiao", "jieshuo"
                }
                if emotion not in valid_emotions:
                    emotion = ""  # 无效情绪视为未指定
                return emotion, text
        return "", response

    async def _speak_response(self, text: str, emotion: str = ""):
        """
        播放 AI 最终回答（带通知音和对话窗口）。

        Args:
            text: 要播报的文本
            emotion: 情绪/风格
        """
        if not text.strip() or not self.tts_engine:
            return

        logger.info(f"🔊 Response: {text[:60]}...")
        player_pause(force=True)
        try:
            if self.frontend:
                self.frontend.set_tts_playing(True)
            self.tts_engine.start()
            if emotion:
                self.tts_engine.set_emotion(emotion)
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
            logger.warning(f"Response TTS error: {e}")
        finally:
            if self.frontend:
                self.frontend.set_tts_playing(False)
            player_resume()
        self._deliver_pending()

    async def _close_conversation(self):
        """关闭对话窗口，回到 IDLE 状态（处理后恢复音乐）。"""
        if self.frontend:
            self.frontend._cancel_conversation_timer()
            # 强制切换到 IDLE，触发 _on_frontend_state_change → player_resume + _deliver_pending
            self.frontend._set_state(VoiceState.IDLE)
            await asyncio.sleep(0)

    def _cancel_current_output(self):
        """打断当前物理输出（TTS + 音频播放），不涉及 agent 逻辑层。"""
        logger.info("⏹ Interrupting current output")
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
        """
        唤醒词命中：区分两种情况——
        1. 唤醒的是当前活跃 agent（同人）→ 完全打断（agent + TTS + speak队列）
        2. 唤醒的是其他 agent（切换）→ 仅打断 TTS 播放，保留原 agent 继续处理
        """
        old_active = self._active_name
        self._active_name = name

        # 递增该 agent 的代际（旧结果自动失效）
        entry = self._agents.get(name)
        if entry:
            agent, executor, history, gen = entry
            self._agents[name] = (agent, executor, history, gen + 1)
            logger.info("🔊 Wake word: '%s' (gen=%d)", name, gen + 1)
        else:
            logger.info("🔊 Wake word: '%s' (no agent)", name)
        # 唤醒后暂停音乐播放
        player_pause()

        if name == old_active:
            # 同一 agent 被再次唤醒：完全打断该 agent 的所有操作
            logger.info("🔄 Same agent [%s], full interrupt", name)
            # 打断 agent 的工具调用
            if entry:
                agent, _, _, _ = entry
                try:
                    agent.interrupt()
                except Exception as e:
                    logger.debug(f"Agent interrupt: {e}")
            # 清空 speak 队列中尚未播放的消息
            while not self._speak_queue.empty():
                try:
                    self._speak_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            logger.info("🗑️ Cleared speak queue for same-agent interrupt")
        else:
            # 切换 agent：仅打断 TTS/音频，不干涉原 agent 逻辑
            if old_active:
                logger.info("🔄 Switch agent [%s] → [%s], TTS only interrupt", old_active, name)

    def _on_interrupt_request(self):
        """
        VoiceFrontend 在唤醒词命中时首先调用此回调。
        此时还不知道是哪个唤醒词，只打断物理输出（TTS + 音频）。
        agent 层的打断决策在 _on_wake_word 中根据是否同 agent 决定。
        """
        self._cancel_current_output()

    def _on_frontend_state_change(self, state):
        logger.debug(f"Frontend: {state.value}")
        # 连续对话窗口过期 → IDLE → 递送待汇报结果
        if state.value == "idle":
            # 对话完全结束（无 ASR、TTS、连续对话窗口），恢复音乐播放
            player_resume()
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
