#!/usr/bin/env python3
"""
Hermes Agent 语音输入前端 — VoiceApp 应用逻辑。

整合：
  - Sherpa-onnx唤醒词 + WebRTC VAD
  - 腾讯云实时语音识别（ASR SDK）
  - Hermes Agent（Python 库，嵌入运行）
  - 腾讯云流式语音合成（TTS SDK）
  - 唤醒词打断 TTS + 代理对话
  - post_api_request 插件回调驱动语音播报（替代 speak 工具）：
    每次 LLM 回复经 speech-relay 插件转发到播放队列实时播报，
    最终回复（finish_reason=stop）播完后进入对话窗口

本模块只定义 VoiceApp，不包含启动入口。
启动方式见 __main__.py（python -m voice_agent）或项目根启动脚本。
"""

import asyncio
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .asr_engine import TencentCloudASREngine, TencentASRConfig
from .audio_player import AudioPlayer, AudioPlayerConfig
from .config import load_config
from .tts_engine import TencentCloudTTSEngine, TencentTTSConfig
from .music_control import player_pause, player_resume
from .voice_frontend import VoiceFrontend, VoiceFrontendConfig, VoiceState

# ─── 项目级插件引导 ──────────────────────────────────────
# 必须在 AIAgent 构造之前调用（agent/agent_init.py 内部会触发插件发现并缓存状态）：
# 让 ./.hermes/plugins 下的项目插件（speech-relay 等）在本进程加载，
# 且不修改全局 ~/.hermes/config.yaml（进程内白名单替换）。
from .plugin_hooks import ensure_project_plugins_loaded

ensure_project_plugins_loaded()

# Hermes Agent（嵌入运行）
from run_agent import AIAgent

# AI 工具注册
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

        # LLM 回复播放队列：post_api_request 回调（speech-relay 插件 → 播报桥）
        # 把每次 LLM 回复投递到这里。
        # 元素: (name, gen, segments, is_final)
        #   segments 为 [(emotion, text), ...]（已解析的情绪分段）
        #   is_final=True 表示 finish_reason=stop 的最终回复（播完进对话窗口）
        self._reply_queue: asyncio.Queue[tuple[str, int, list, bool]] = asyncio.Queue()
        self._reply_consumer_task: asyncio.Task | None = None

        # 会话 id → 唤醒词名：post_api_request 载荷里只有 session_id，靠它定位 agent
        self._session_to_name: dict[str, str] = {}

        # 待汇报队列：agent 完成但当前有其他 agent 活跃时排队
        # 元素: (keyword, segments, gen)；segments 为 [(emotion, text), ...]
        self._pending: list[tuple[str, list, int]] = []
        self._user_title = self.config.get("user_title", "主人")

        # 说话人识别（voiceprint 启用时非 None）：
        # _vp_round 收集本轮对话每个句子的 (显示名, 文本)，on_final 时拼成
        # "[说话人] 内容" 消息发给 LLM；_vp_id_cache 缓存腾讯云 speaker_id →
        # (库id, 相似度)，同 speaker_id 只做一次声纹识别（低延迟）
        self.voiceprint = None
        self._vp_round: list[tuple[str, str]] = []
        self._vp_id_cache: dict[int, tuple[str, float]] = {}
        self._vp_auto_register = True
        self._vp_min_register_sec = 1.5
        self._vp_use_cache = True

        # 等待回复提示音：ASR 结束后、LLM 回复/TTS 开始前循环播放，
        # 提示用户当前不再收听语音。_wait_tone_waiting 为等待计数（多 agent
        # 并发时任一在等待即播放）
        self._wait_tone_task: asyncio.Task | None = None
        self._wait_tone_waiting = 0

    # ─── 生命周期 ────────────────────────────────────────

    async def start(self):
        self._loop = asyncio.get_running_loop()

        logger.info("=" * 50)
        logger.info("Hermes Agent Voice Frontend")
        logger.info("=" * 50)

        tencent_cfg = self._full_cfg.get("tencent", {})

        # 说话人识别（可选，并行旁路）：每完成一个腾讯云句子，本地 CAM++
        # 提取声纹与特征库对照，标注说话人；启用时 ASR 需开启说话人分离
        vp_cfg = self.config.get("voiceprint", {})
        if vp_cfg.get("enabled", False):
            from .voiceprint import VoiceprintManager
            lib_dir = Path(vp_cfg.get("lib_dir", "models/voiceprint_lib"))
            if not lib_dir.is_absolute():
                lib_dir = Path(__file__).resolve().parent.parent / lib_dir
            self.voiceprint = VoiceprintManager(
                lib_dir=lib_dir,
                threshold=vp_cfg.get("threshold", 0.6),
                speaker_names=vp_cfg.get("speaker_names", {}),
            )
            self._vp_auto_register = vp_cfg.get("auto_register", True)
            self._vp_min_register_sec = vp_cfg.get("min_register_sec", 1.5)
            self._vp_use_cache = vp_cfg.get("speaker_id_cache", True)
            if self.voiceprint.extractor is None:
                logger.warning("CAM++ 声纹模型不可用，说话人识别降级（不标注说话人）")
                self.voiceprint = None
        else:
            logger.info("说话人识别未启用（voice.voiceprint.enabled=false）")

        # 1. ASR 引擎
        asr_config = TencentASRConfig(
            secret_id=tencent_cfg.get("secret_id", ""),
            secret_key=tencent_cfg.get("secret_key", ""),
            app_id=tencent_cfg.get("app_id", ""),
            engine_model=self.config["asr"]["engine_model"],
            needvad=self.config["asr"]["needvad"],
            voice_format=self.config["asr"]["voice_format"],
            # 开启说话人分离：句子带 speaker_id + 起止毫秒（需 speaker 引擎）
            enable_speaker_context=1 if self.voiceprint else 0,
        )
        self.asr_engine = TencentCloudASREngine(asr_config)
        self.asr_engine.on_start = self._on_asr_start
        self.asr_engine.on_interim = self._on_asr_interim
        # 不绑定 on_final：服务端 VAD 可能在用户句间停顿时就触发 on_sentence_end，
        # 导致 agent 在用户未说完时就开始回答。仅在 on_complete 后由本地 VAD 控制时机。
        self.asr_engine.on_final = None
        self.asr_engine.on_complete = self._on_asr_complete
        self.asr_engine.on_error = self._on_asr_error
        if self.voiceprint:
            # 每完成一个句子 → 本地声纹识别说话人（并行旁路）
            self.asr_engine.on_sentence = self._on_asr_sentence

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
        register_mpd_tools()

        # ─── 系统提示词（语音助手通用规则 + 自动播报 + 播放控制规则）─
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
            '【说话人标识说明】\n'
            '发送给你的每条用户消息会以 [说话人身份] 前缀标注这句话是谁说的，\n'
            '例如「[爸爸] 今天天气怎么样？」；多人连续说话时每句单独标注，如\n'
            '「[爸爸] 今天天气怎么样？\n[妈妈] 顺便查下明天的」。说话人身份用于\n'
            '帮助你理解对话上下文（例如区分不同家庭成员提出的问题），回答时\n'
            '不需要复述说话人。\n'
            '\n'
            '【语音播报规则】\n'
            '你的每条文字回复都会被系统实时语音播报给用户（无需调用任何播报工具）。\n'
            '- 工具调用过程中的阶段性说明（如「正在搜索网络」）请直接作为文字回复输出，\n'
            '  系统会立即播报；不需要文字说明的纯工具调用轮次可以只返回工具调用。\n'
            '- 最终回答请直接以文字格式回复，格式为：(情绪)你要说的话\n'
            '  例如：(happy)你好，有什么可以帮助你的？\n'
            '  情绪可选值：neutral(中性) sad(悲伤) happy(高兴) angry(生气) fear(恐惧) '
            'story(故事) poetry(诗歌) sajiao(撒娇) disgusted(厌恶) amaze(震惊) exciting(兴奋) '
            'aojiao(傲娇) jieshuo(解说)\n'
            '- 如果你没有用到情绪，可以用 neutral，例如：(neutral)这是回答\n'
            '\n'
            '【播放控制工具规则】\n'
            '你拥有 mpd_ 系列工具用于控制音乐播放（如 mpd_play、mpd_pause、mpd_stop、\n'
            'mpd_previous、mpd_next 等）。\n'
            '- 当你使用播放控制工具时，播放控制工具执行后直接返回 [FINISH] 作为文字回复，\n'
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
                provider=ac.get("provider"),
                api_mode=ac.get("api_mode"),
                base_url=ac.get("base_url"),
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
            self._session_to_name[agent.session_id] = name
            logger.info("Agent [%s]: model=%s, provider=%s, api_mode=%s, session=%s",
                        name, ac.get("model"), agent.provider,
                        agent.api_mode, ac.get("session_id"))

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
            wake_guard_sec=self.config["vad"].get("wake_guard_sec", 2.5),
            mic_sample_rate=self.config.get("mic_sample_rate", 0),
            mic_device=None if str(self.config.get("mic_device") or "").lower()
            in ("", "auto") else self.config["mic_device"],
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

        # 7. 启动 LLM 回复播放队列消费者
        self._reply_consumer_task = asyncio.create_task(self._reply_queue_consumer())

        # 8. 注册播报桥接收方：speech-relay 插件回调把 LLM 回复转交到这里
        from .speech_bridge import set_sink
        set_sink(self._on_llm_reply_hook)

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

        # 注销播报桥接收方
        from .speech_bridge import clear_sink
        clear_sink()

        # 取消 LLM 回复播放队列消费者
        if self._reply_consumer_task:
            self._reply_consumer_task.cancel()
            try:
                await self._reply_consumer_task
            except asyncio.CancelledError:
                pass
            self._reply_consumer_task = None

        if self.frontend:
            self.frontend.stop()
        if self.audio_player:
            self.audio_player.stop()
        for name, (agent, executor, history, gen) in self._agents.items():
            executor.shutdown(wait=False)
        logger.info("Voice frontend stopped")

    # ─── LLM 回复播放队列消费者 ──────────────────────────

    async def _reply_queue_consumer(self):
        """
        后台消费 LLM 回复播放队列：
        - 中间轮（is_final=False）：逐段阶段性播报（不播通知音，不进对话窗口）
        - 最终轮（is_final=True）：整段流式播放，播完播通知音并进入对话窗口
        播放前做代际校验，期间被打断（gen 已递增）的回复直接丢弃。
        """
        try:
            while True:
                name, gen, segments, is_final = await self._reply_queue.get()
                _, _, _, current_gen = self._agents.get(name, (None, None, None, -1))
                if gen != current_gen:
                    logger.info(f"🚫 Reply [{name}] gen {gen} != current {current_gen}, discard")
                    continue
                if is_final:
                    await self._speak_response(segments)
                else:
                    for emotion, text in segments:
                        if text and text.strip():
                            await self._speak_raw(text, emotion=emotion)
        except asyncio.CancelledError:
            logger.info("Reply consumer cancelled")

    async def _speak_raw(self, text: str, emotion: str = ""):
        """
        纯 TTS 合成播放（阶段性汇报 / 中间轮回复使用，不播通知音，不进对话窗口）。

        Args:
            text: 要播报的文本
            emotion: 情绪/风格，如 "happy", "sad", "angry" 等，空字符串表示不指定
        """
        if not text.strip() or not self.tts_engine:
            return

        logger.info(f"🔊 Speak: {text[:60]}...")
        self._stop_wait_tone()  # 播报前停止提示音
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
            # 阶段性汇报播完：agent 可能仍在等待最终回复 → 恢复提示音
            if self._wait_tone_waiting > 0:
                self._start_wait_tone()

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

                # 播放控制类工具执行后的回复标记 [FINISH]：回调路径（_on_llm_reply_hook）
                # 已处理并关闭对话窗口，这里兜底一次（钩子异常未触发时）
                if response.upper().strip("[]") == "FINISH":
                    logger.info(f"✅ [{name}] FINISH signal, close conversation window")
                    if name == self._active_name and self.frontend:
                        asyncio.run_coroutine_threadsafe(
                            self._close_conversation(), self._loop
                        )
                    return

                # 语音播报（含情绪分段解析、通知音、对话窗口）由 post_api_request
                # 回调完成：speech-relay 插件 → 播报桥 → _on_llm_reply_hook →
                # 播放队列。这里只保留日志与空回复兜底，不再重复播放。
                segments = self._parse_emotion_segments(response)
                logger.info(f"(gen={gen}) [{name}] Final response: {len(segments)} 段, "
                            f"{segments[0][1][:40] if segments else '(空)'}...")
            except Exception as e:
                logger.error(f"Agent [{name}] chat error: {e}")
            finally:
                # agent 处理结束（无论结果）：停止等待提示音。
                # _stop_wait_tone 是同步函数，跨线程用 call_soon_threadsafe
                # （run_coroutine_threadsafe 需要协程对象，传同步调用会抛 TypeError）
                self._loop.call_soon_threadsafe(self._stop_wait_tone)

        # 开始等待 LLM 回复：循环播放提示音（提示不再收听）
        self._start_wait_tone()
        executor.submit(_chat)

    # ─── LLM 回复钩子（post_api_request 回调）────────────

    def _on_llm_reply_hook(self, payload: dict):
        """
        播报桥接收方：speech-relay 插件把每次 API 调用后的回复载荷转交到这里。

        运行在 hermes agent 的调用线程（同步）。职责：
        - 通过 session_id 定位唤醒词名；
        - 解析 (情绪)文字 分段，投递到播放队列；
        - finish_reason=stop 视为最终回复（is_final=True，播完进对话窗口）；
        - [FINISH]（播放控制工具结束标记）不播报，直接关闭对话窗口；
        - 当前已被其他唤醒词取代时，最终回复进入待汇报队列排队递送。
        """
        try:
            session_id = payload.get("session_id") or ""
            name = self._session_to_name.get(session_id)
            if not name or name not in self._agents:
                return
            finish_reason = payload.get("finish_reason") or ""
            assistant = payload.get("assistant_message")
            if isinstance(assistant, dict):
                content = assistant.get("content") or ""
            else:
                content = getattr(assistant, "content", None) or ""
            text = content.strip()
            if not text:
                return

            # 播放控制类工具结束标记 [FINISH]：不播报，关闭对话窗口
            if text.upper().strip("[]") == "FINISH":
                logger.info(f"✅ [{name}] FINISH signal (hook), close conversation window")
                if name == self._active_name and self.frontend:
                    asyncio.run_coroutine_threadsafe(
                        self._close_conversation(), self._loop
                    )
                return

            # 代际：回调时刻的当前 gen，播放前消费者还会再校验一次
            _, _, _, gen = self._agents.get(name, (None, None, None, 0))
            segments = self._parse_emotion_segments(text)
            if not segments:
                return
            is_final = finish_reason == "stop"
            if name == self._active_name:
                asyncio.run_coroutine_threadsafe(
                    self._reply_queue.put((name, gen, segments, is_final)),
                    self._loop,
                )
            elif is_final:
                # 已被其他唤醒词取代：最终回复进入待汇报队列，等对话窗口过期后递送
                self._pending.append((name, segments, gen))
                logger.info(f"📥 [{name}] queued (pending={len(self._pending)})")
            # 非最终轮且已不活跃：对话被打断，丢弃
        except Exception as e:
            logger.error(f"LLM reply hook error: {e}")

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

        kw, segments, gen = self._pending.pop(0)
        # 检查该 agent 的代际是否仍匹配
        _, _, _, current_gen = self._agents.get(kw, (None, None, None, -1))
        if gen != current_gen:
            logger.info(f"🚫 Pending [{kw}] gen {gen} != current {current_gen}, discard")
            self._deliver_pending()  # 尝试下一个
            return

        # 问候前缀加到第一段，然后统一走 _speak_response（内部支持多段拼接播放）
        first_emotion, first_text = segments[0]
        greeting = f"{self._user_title}，我是{kw}，{first_text}"
        segs = [(first_emotion, greeting)] + segments[1:]
        logger.info("📤 Delivering pending result from [%s]: %s...",
                    kw, greeting[:60])
        asyncio.run_coroutine_threadsafe(
            self._speak_response(segs), self._loop
        )

    # ─── 情绪解析工具 ────────────────────────────────────

    # 合法情绪集合（与 TTS 引擎保持一致）
    VALID_EMOTIONS = {
        "neutral", "sad", "happy", "angry", "fear",
        "story", "poetry", "sajiao", "disgusted", "amaze",
        "exciting", "aojiao", "jieshuo",
    }
    EMOTION_PATTERN = re.compile(
        r"\((?:%s)\)" % "|".join(sorted(VALID_EMOTIONS)), re.IGNORECASE)

    def _parse_emotion_segments(self, response: str) -> list[tuple[str, str]]:
        """
        解析 AI 最终回复中的 (情绪)文字内容 格式，支持多个情绪标记分段。

        大模型可能在一段回复里用多个情绪标记分别表达不同语气，例如：
            (neutral)好嘞，再来一个！……(happy)哈哈，好笑吗？
        这里把每个 "(情绪)" 标记作为一段的起点，逐段切出
        [(emotion, text), ...]；段首无标记的文本归入 emotion="" 段。

        Args:
            response: AI 的文字回复

        Returns:
            [(emotion, text), ...] 非空段列表；emotion 为空表示该段未指定情绪。
            整段无任何情绪标记时返回 [("", response)]。
        """
        response = response.strip().replace('（', '(').replace('）', ')')
        if not response:
            return []
        matches = list(self.EMOTION_PATTERN.finditer(response))
        if not matches:
            return [("", response)]

        segments: list[tuple[str, str]] = []
        # 第一个标记之前的文本（无情绪前缀）
        if matches[0].start() > 0:
            head = response[:matches[0].start()].strip()
            if head:
                segments.append(("", head))
        # 每个标记领起一段，到下一个标记前结束
        for i, m in enumerate(matches):
            emotion = m.group(0)[1:-1].strip().lower()
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(response)
            text = response[start:end].strip()
            if text:
                segments.append((emotion, text))
        return segments

    async def _speak_response(self, segments):
        """
        播放 AI 最终回答（支持多段情绪，带通知音和对话窗口）。

        Args:
            segments: [(emotion, text), ...] 段列表；emotion 为空表示不指定情绪。
                      单段回复传 [("", text)] 或 [("neutral", text)] 即可。

        播放策略：**全程流式实时**。腾讯云 TTS 边合成边推音频块，on_audio_chunk
        直接接播放器，音频随到随播（低首字延迟）；多段时逐段按各自情绪合成，
        前一段的音频在播放队列里自然续上后一段的内容（段间换情绪需重连 WS，
        会有极短的衔接停顿）。全部内容播完后播通知音并进入对话窗口。
        """
        import time as _t

        if not segments or not self.tts_engine:
            return
        texts = [t for _, t in segments if t and t.strip()]
        if not texts:
            return

        logger.info(f"🔊 Response: {len(segments)} 段, 首段 {texts[0][:40]}...")
        self._stop_wait_tone()  # TTS 播放前停止提示音
        player_pause(force=True)
        try:
            if self.frontend:
                self.frontend.set_tts_playing(True)

            # 全程流式实时：音频块直接喂播放器，逐段边合成边播
            self.tts_engine.on_audio_chunk = (
                self.audio_player.feed if self.audio_player else None)
            for emotion, text in segments:
                if not text or not text.strip():
                    continue
                self.tts_engine.start()
                if emotion:
                    self.tts_engine.set_emotion(emotion)
                self.tts_engine.synthesize(text)
                self.tts_engine.complete()
                self.tts_engine.wait(timeout=15)

            # 全部合成完，等待播放队列播完
            if self.audio_player:
                self.audio_player.wait_for_drain(timeout=15.0)
                _t.sleep(0.15)

            # 全部播完后：通知音 + 进入对话窗口
            self._play_asset("notification")
            if self.audio_player:
                self.audio_player.wait_for_drain(timeout=5.0)
                _t.sleep(0.05)
            if self.frontend:
                self.frontend.enter_conversation_window()
        except Exception as e:
            logger.warning(f"Response TTS error: {e}")
        finally:
            # 恢复音频回调到播放器
            if self.tts_engine:
                self.tts_engine.on_audio_chunk = (
                    self.audio_player.feed if self.audio_player else None)
            if self.frontend:
                self.frontend.set_tts_playing(False)
            player_resume()
        self._deliver_pending()

    async def _close_conversation(self):
        """关闭对话窗口，回到 IDLE 状态（处理后恢复音乐）。"""
        self._stop_wait_tone()  # 对话结束：停止等待提示音
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
    # 等待回复提示音（2s 低幅提示音，循环播放提示"不再收听"）
    WAIT_TONE_PATH = Path(__file__).resolve().parent.parent / "assets" / "wait_cue_4_scale.wav"

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

    # ─── 等待回复提示音 ──────────────────────────────

    def _start_wait_tone(self):
        """
        开始循环播放提示音（等待 LLM 回复期间，提示用户当前不再收听）。

        幂等：计数 +1；计数从 0 变 1 时启动播放任务。
        播放前暂停音乐，避免与背景音混播。

        注意：本方法可能在非 asyncio 线程被调用（_ask_agent 由 ASR SDK
        接收线程触发），因此用 call_soon_threadsafe 把任务创建调度到事件
        循环线程，避免 ensure_future 因"无运行中的事件循环"抛异常。
        """
        self._wait_tone_waiting += 1
        if self._wait_tone_task is None:
            if not self.WAIT_TONE_PATH.is_file():
                logger.warning("等待提示音文件缺失: %s", self.WAIT_TONE_PATH)
                return
            if self._loop is None:
                logger.warning("事件循环未就绪，跳过等待提示音")
                return
            player_pause(force=True)
            if self.audio_player:
                self.audio_player.clear()
            self._loop.call_soon_threadsafe(self._spawn_wait_tone_task)

    def _spawn_wait_tone_task(self):
        """在事件循环线程内创建提示音播放任务（供 call_soon_threadsafe 调用）。"""
        if self._wait_tone_task is None:
            self._wait_tone_task = asyncio.ensure_future(self._wait_tone_loop())

    def _stop_wait_tone(self):
        """
        停止提示音（TTS 开始 / 对话结束 / 打断时调用）。

        幂等：计数 -1；计数归 0 时取消播放任务并清空队列中未播的提示音。
        """
        if self._wait_tone_waiting > 0:
            self._wait_tone_waiting -= 1
        if self._wait_tone_waiting == 0 and self._wait_tone_task is not None:
            self._wait_tone_task.cancel()
            self._wait_tone_task = None
            if self.audio_player:
                self.audio_player.clear()

    async def _wait_tone_loop(self):
        """循环把提示音 PCM 喂给播放器（按播放时长节流）。"""
        try:
            import wave

            with wave.open(str(self.WAIT_TONE_PATH), "rb") as wf:
                data = wf.readframes(wf.getnframes())
            if not data:
                return
            # 每段 2s，feed 后按播放时长 sleep，实现无缝循环
            while True:
                if self.audio_player:
                    self.audio_player.feed(data)
                await asyncio.sleep(len(data) / 32000.0)  # 16k 16bit = 32KB/s
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning("提示音播放异常: %s", e)

    def _on_play_prompt(self):
        """播唤醒提示音。"""
        self._play_asset("prompt")

    def _on_wake_word(self, name: str):
        """
        唤醒词命中：区分两种情况——
        1. 唤醒的是当前活跃 agent（同人）→ 完全打断（agent + TTS + 回复播放队列）
        2. 唤醒的是其他 agent（切换）→ 仅打断 TTS 播放，保留原 agent 继续处理
        """
        old_active = self._active_name
        self._active_name = name

        # 唤醒即打断：停止等待提示音（用户要开始说话/新对话）
        self._stop_wait_tone()

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
            # 清空 LLM 回复播放队列中尚未播放的消息
            while not self._reply_queue.empty():
                try:
                    self._reply_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            logger.info("🗑️ Cleared reply queue for same-agent interrupt")
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
            # 兜底：对话结束（agent 出错/无回复等未走 TTS 路径时）停止等待提示音
            self._stop_wait_tone()
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
        # 说话人识别启用时：把本轮每句 "[说话人] 内容" 拼成消息发给 LLM
        if self.voiceprint is not None and self._vp_round:
            msg = "\n".join(f"[{spk}] {t}" for spk, t in self._vp_round)
            self._vp_round.clear()
        else:
            msg = text
        logger.info(f"📝 [{name}] User said: {msg}")
        self._ask_agent(msg, name)

    def _on_frontend_error(self, msg: str):
        logger.error(f"❌ Frontend: {msg}")

    # ASR 回调

    def _on_asr_start(self):
        logger.debug("ASR started")
        # 新一轮 ASR 会话：重置说话人识别上下文（本轮句子列表 + speaker_id 缓存）
        self._vp_round.clear()
        self._vp_id_cache.clear()

    def _on_asr_sentence(self, info: dict):
        """
        腾讯云每完成一个句子 → 本地声纹识别说话人（并行旁路，不影响 agent 流程）。

        识别结果收集到 _vp_round，on_final 时拼成 "[说话人] 内容" 消息发给 LLM；
        新说话人按配置自动分配 id 注册；同 speaker_id 缓存避免重复声纹计算。
        """
        if self.voiceprint is None:
            return
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
            samples = self.asr_engine.get_sentence_audio(info)
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
        except Exception as e:
            logger.warning("说话人识别异常: %s", e)

    def _on_asr_complete(self):
        logger.debug("ASR complete")
        # 会话结束：说话人 speaker_id 缓存失效
        self._vp_id_cache.clear()
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
