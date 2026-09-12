"""
语音输入装配（inbound，M2）：VoiceFrontend + ASR(+voiceprint) → WS 上行事件。

把语音侧「麦克风 → 唤醒词 → VAD → 腾讯云 ASR」装配进 Voice Service，并把事件
上行到已连接的 hermes voice gateway（client）：

- KWS 命中 → 上行 `wake_word`（本地提示音/打断行为在 M3 playback 接入前占位）
- ASR 整段完成 → 上行 `asr_result`（voiceprint 启用时文本按句带
  `[名字 (ID: 编号)]` 前缀，格式见 PROTOCOL.md §5）
- 本地回合号 `turn_seq` 随 asr_result 上行，客户端用它标记 speak 归属轮次

硬件与模型路径约定：模型/关键词默认相对 **voice_service 子项目根**
（= 本模块的上两级目录；与复制来的 voice_frontend.py 默认解析一致），
KWS 模型目录可在 `voice_service:` 配置中覆盖。

注意：本模块装配组件但不代表启动设备；`start()` 才会打开麦克风 ——
在部署机器上由 systemd 以 --audio 启动，开发/无头环境不调用。
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from . import protocol as P
from . import voice_assets as VA
from .server import VoiceServer
from .service_config import VoiceServiceConfig
from . import kws_words

logger = logging.getLogger("voice_service.inbound")

DEFAULT_KWS_MODEL_NAME = "sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20"


class Inbound:
    """语音输入装配：构造 ASR/VoiceFrontend/voiceprint 并把事件上行到 server。"""

    def __init__(self, cfg: VoiceServiceConfig, server: VoiceServer,
                 playback=None):
        self.cfg = cfg
        self.server = server
        self.playback = playback      # Playback | None（M3：回调接播报）
        self.asr = None            # TencentCloudASREngine
        self.frontend = None       # VoiceFrontend
        self.voiceprint = None     # VoiceprintManager | None
        self._current_wake: str = ""
        self._turn_seq: int = 0
        # voiceprint 辅助（与旧 adapter 一致）
        self._vp_round: list = []
        self._vp_id_cache: dict = {}
        self._vp_auto_register = True
        self._vp_min_register_sec = 1.5
        self._vp_use_cache = True

    # ─── 组装 ────────────────────────────────────────────

    def enable(self) -> None:
        """构造全部输入组件（不启动硬件）。失败抛异常由调用方降级。"""
        raw = self.cfg.raw
        vp_cfg = raw.get("voiceprint") or {}
        if vp_cfg.get("enabled", False):
            self._init_voiceprint(vp_cfg)
        self._init_asr(raw.get("asr") or {})
        self._init_frontend(
            raw.get("wake_word") or {}, raw.get("vad") or {},
            raw.get("kws") or {}, raw.get("mic") or {})

    def start(self) -> None:
        """打开麦克风开始监听（真实硬件；需 --audio 显式启用）。"""
        if self.frontend is None:
            raise RuntimeError("inbound 未 enable()")
        self.frontend.start()

    def stop(self) -> None:
        if self.frontend is not None:
            try:
                self.frontend.stop()
            except Exception as exc:
                logger.warning("[inbound] frontend stop: %s", exc)

    # ─── 组件构造 ────────────────────────────────────────

    def _tencent_creds(self) -> dict:
        t = self.cfg.tencent
        return {"secret_id": t.secret_id, "secret_key": t.secret_key,
                "app_id": t.app_id}

    def _init_voiceprint(self, vp_cfg: dict) -> None:
        from .voiceprint import VoiceprintManager
        try:
            lib_dir = vp_cfg.get("lib_dir") or "models/voiceprint_lib"
            p = Path(lib_dir)
            if not p.is_absolute():
                p = Path(__file__).resolve().parent.parent / lib_dir
            self.voiceprint = VoiceprintManager(
                lib_dir=str(p),
                threshold=float(vp_cfg.get("threshold", 0.6)),
                speaker_names=vp_cfg.get("speaker_names") or {},
            )
            self._vp_auto_register = bool(vp_cfg.get("auto_register", True))
            self._vp_min_register_sec = float(vp_cfg.get("min_register_sec", 1.5))
            self._vp_use_cache = bool(vp_cfg.get("speaker_id_cache", True))
            if self.voiceprint.extractor is None:
                logger.warning("[inbound] 声纹模型不可用，说话人识别降级")
                self.voiceprint = None
            else:
                logger.info("[inbound] 说话人识别已启用（特征库: %s）", p)
        except Exception as exc:
            logger.warning("[inbound] voiceprint 初始化失败（降级）: %s", exc)
            self.voiceprint = None

    def _init_asr(self, asr_cfg: dict) -> None:
        from .asr_engine import TencentASRConfig, TencentCloudASREngine
        creds = self._tencent_creds()
        self.asr = TencentCloudASREngine(TencentASRConfig(
            secret_id=creds["secret_id"],
            secret_key=creds["secret_key"],
            app_id=creds["app_id"],
            # 显式配置优先；未配置时 voiceprint 开启默认走说话人分离引擎
            engine_model=asr_cfg.get("engine_model")
            or ("16k_zh_en_speaker_2.0" if self.voiceprint else "16k_zh"),
            needvad=asr_cfg.get("needvad", False),
            voice_format=asr_cfg.get("voice_format", 1),
            enable_speaker_context=1 if self.voiceprint else 0,
        ))
        self.asr.on_start = self._on_asr_start
        self.asr.on_final = None   # 只听 on_complete（与 VoiceApp 一致）
        self.asr.on_complete = self._on_asr_complete
        self.asr.on_error = self._on_asr_error
        if self.voiceprint:
            self.asr.on_sentence = self._on_asr_sentence

    def _init_frontend(self, wake_cfg: dict, vad_cfg: dict,
                       kws_cfg: dict, mic_cfg: dict) -> None:
        from .voice_frontend import VoiceFrontend, VoiceFrontendConfig

        # 助手表：config 是唯一源（服务启动时已同步 keywords.txt 到模型目录，
        # 这里只取首个助手名作为 keywords.txt 缺失时的兜底生成词）
        assistants = kws_words.parse_assistants(wake_cfg)
        fallback_name = (assistants[0]["name"] if assistants
                         else (wake_cfg.get("keyword") or "小布"))

        # 模型默认目录：voice_service 子项目根下的 models/sherpa-kws
        project_root = Path(__file__).resolve().parent.parent
        kws_dir = str(project_root / "models" / "sherpa-kws")

        self.frontend = VoiceFrontend(VoiceFrontendConfig(
            wake_word_enabled=bool(wake_cfg.get("enabled", True)),
            # 仅当 keywords.txt 缺失时才用该词兜底自动生成（正常由 config
            # assistants 生成词表；此字段仅为极端缺失场景兜底）
            wake_word_keyword=fallback_name,
            wake_word_threshold=float(kws_cfg.get("threshold",
                                                  wake_cfg.get("threshold", 0.25))),
            # KWS 命中加分（sherpa keywords_score，越高越容易被该词触发）
            wake_word_score=float(kws_cfg.get("score", wake_cfg.get("score", 1.0))),
            kws_model_dir=kws_dir,
            kws_model_name=kws_cfg.get("model_name", DEFAULT_KWS_MODEL_NAME),
            kws_encoder=kws_cfg.get("encoder",
                                    "encoder-epoch-13-avg-2-chunk-8-left-64.int8.onnx"),
            kws_decoder=kws_cfg.get("decoder",
                                    "decoder-epoch-13-avg-2-chunk-8-left-64.onnx"),
            kws_joiner=kws_cfg.get("joiner",
                                   "joiner-epoch-13-avg-2-chunk-8-left-64.int8.onnx"),
            kws_tokens=kws_cfg.get("tokens", "tokens.txt"),
            vad_mode=int(vad_cfg.get("mode", 3)),
            min_speech_ms=int(vad_cfg.get("min_speech_ms", 200)),
            vad_silence_threshold_ms=int(vad_cfg.get("silence_threshold_ms", 600)),
            vad_speech_confirm_frames=int(vad_cfg.get("speech_confirm_frames", 3)),
            wake_guard_sec=float(vad_cfg.get("wake_guard_sec", 2.5)),
            mic_sample_rate=int(mic_cfg.get("sample_rate", 0)),
            mic_device=None if str(mic_cfg.get("device") or "").lower()
            in ("", "auto") else mic_cfg.get("device"),
            mic_no_data_timeout_sec=float(mic_cfg.get("no_data_timeout_sec", 3.0)),
            mic_fail_cooldown_sec=float(mic_cfg.get("fail_cooldown_sec", 60.0)),
            conversation_window_sec=float(
                (self.cfg.raw.get("conversation_window") or {})
                .get("timeout_sec", 5.0)),
            on_wake_word=self._on_wake_word,
            on_interrupt_request=self._on_interrupt_request,
            on_play_prompt=self._on_play_prompt,
            prompt_duration_sec=VA.asset_duration_sec("prompt"),
            on_conversation_timeout=self._on_conversation_timeout,
        ), self.asr)

    # ─── VoiceFrontend 回调（音频线程）────────────────────

    def _on_wake_word(self, name: str) -> None:
        """唤醒命中：记录当前活跃助手并上行事件。"""
        self._current_wake = name or self._current_wake
        logger.info("[inbound] 唤醒词: %s", name)
        # 通知 playback 当前活跃助手（用于识别被其他对话打断的迟到回复）
        if self.playback is not None:
            self.playback.set_active_wake(self._current_wake)
        self._fire(P.EVT_WAKE_WORD, {"keyword": name, "wake": self._current_wake})

    def _on_interrupt_request(self) -> None:
        """本地物理打断（唤醒命中时）：停 TTS/清队列/停等待音（→ playback）。"""
        logger.info("[inbound] 本地打断请求")
        if self.playback is not None:
            self.playback.on_interrupt()

    def _on_play_prompt(self) -> None:
        """播唤醒提示音（→ playback 资产队列）。"""
        if self.playback is not None:
            self.playback.play_asset("prompt")
        else:
            logger.debug("[inbound] 播提示音（无 playback，忽略）")

    def _on_conversation_timeout(self) -> None:
        """连续对话窗口超时：告别语收尾（→ playback）。"""
        logger.info("[inbound] 连续对话窗口超时")
        if self.playback is not None:
            self.playback.on_conversation_timeout()

    # ─── ASR 回调（SDK 线程）──────────────────────────────

    def _on_asr_start(self) -> None:
        self._vp_round.clear()
        self._vp_id_cache.clear()

    def _on_asr_sentence(self, info: dict) -> None:
        """腾讯云每完成一个句子 → 本地声纹识别（voiceprint 旁路）。"""
        if self.voiceprint is None or self.asr is None:
            return
        text = (info.get("text") or "").strip()
        if not text:
            return
        tx_spk = info.get("speaker_id", 0)
        try:
            if self._vp_use_cache and tx_spk in self._vp_id_cache:
                spk_id, score = self._vp_id_cache[tx_spk]
                self._vp_round.append(
                    (self.voiceprint.speaker_label(spk_id), text))
                return
            samples = self.asr.get_sentence_audio(info)
            if samples is None or len(samples) == 0:
                return
            spk_id, score = self.voiceprint.identify(samples)
            if spk_id is None:
                if (self._vp_auto_register
                        and len(samples) / 16000 >= self._vp_min_register_sec):
                    new_id = self.voiceprint.register(samples)
                    spk_id = new_id if new_id else None
            # 未识别（含注册失败）→「未知」；自动注册但未映射 →「未知 (ID: 1xx)」，
            # 编号稳定，agent 仍能区分不同陌生人
            self._vp_round.append((self.voiceprint.speaker_label(spk_id), text))
            if self._vp_use_cache and spk_id:
                self._vp_id_cache[tx_spk] = (spk_id, score)
        except Exception as exc:
            logger.warning("[inbound] 说话人识别异常: %s", exc)

    def _on_asr_complete(self) -> None:
        """整段识别完成 → 上行 asr_result（含 voiceprint 说话人标签前缀）。"""
        if self.asr is None:
            return
        text = (self.asr.last_text or "").strip()
        if not text:
            logger.info("[inbound] ASR 结果为空，跳过")
            return
        msg = text
        if self.voiceprint is not None and self._vp_round:
            # 每句: "[爸爸 (ID: 100)] 打开客厅灯"（多句换行）；格式见 PROTOCOL.md §5
            msg = "\n".join(f"[{label}] {t}" for label, t in self._vp_round)
            self._vp_round.clear()
        self._turn_seq += 1
        logger.info("[inbound] 用户说: %s", msg[:60])
        self._fire(P.EVT_ASR_RESULT, {
            "text": msg,
            "wake": self._current_wake,
            "message_id": f"voice-{int(time.time() * 1000)}",
            "turn_seq": self._turn_seq,
        })
        # 开始等待回复：循环等待音 + 超时兜底（→ playback）
        if self.playback is not None:
            self.playback.on_utterance_done()

    def _on_asr_error(self, message: str) -> None:
        logger.error("[inbound] ASR 错误: %s", message)

    # ─── 上行 ────────────────────────────────────────────

    def _fire(self, type_: str, data: Dict[str, Any]) -> None:
        """线程安全上行事件（SDK/音频线程调用）。"""
        if self.server.loop is None:
            logger.debug("[inbound] server 事件循环未就绪，丢弃事件 %s", type_)
            return
        try:
            import asyncio
            asyncio.run_coroutine_threadsafe(
                self.server.broadcast_event(type_, data), self.server.loop)
        except Exception as exc:
            logger.warning("[inbound] 上行 %s 失败: %s", type_, exc)
