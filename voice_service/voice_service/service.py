"""
VoiceServiceApp：server + inbound（输入）+ playback（输出）的组合。

用法（python -m voice_service）：
- 默认：仅起 WS 服务端（骨架行为，不动音频/设备）；
- --audio：启用语音输入（打开麦克风监听，唤醒词 → ASR）；
- --out：启用语音输出（TTS → 扬声器、提示音、等待音；会真实发声）。

注意：同一机器上同时只能有一个进程占用麦克风/扬声器；
不要与其它语音服务并发启动（设备/扬声器占用冲突）。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from .inbound import Inbound, DEFAULT_KWS_MODEL_NAME
from .playback import Playback
from .server import VoiceServer
from .service_config import VoiceServiceConfig
from . import kws_words

logger = logging.getLogger("voice_service.app")


class VoiceServiceApp:
    def __init__(self, cfg: VoiceServiceConfig):
        self.cfg = cfg
        self.server = VoiceServer(cfg.service)
        self.inbound: Optional[Inbound] = None
        self.playback: Optional[Playback] = None
        self._assistants: List[dict] = []   # wake_word.assistants（start 时装配）

    @property
    def audio_enabled(self) -> bool:
        return self.inbound is not None

    # ─── 唤醒词 → 助手映射（config 唯一源）──────────────

    def _load_assistants(self) -> List[dict]:
        """从 config 的 wake_word.assistants 解析助手表（空=未配置）。"""
        wake_cfg = self.cfg.raw.get("wake_word") or {}
        return kws_words.parse_assistants(wake_cfg)

    def _ensure_keywords(self, assistants: List[dict]) -> None:
        """把 config 助手表同步成模型目录 keywords.txt（KWS 实际加载的词表）。"""
        kws_cfg = self.cfg.raw.get("kws") or {}
        project_root = Path(__file__).resolve().parent.parent
        model_dir = str(project_root / "models" / "sherpa-kws")
        model_name = kws_cfg.get("model_name") or DEFAULT_KWS_MODEL_NAME
        path, _ = kws_words.ensure_keywords_file(model_dir, model_name, assistants)
        if assistants and not path.is_file():
            logger.warning("[app] keywords.txt 未生成（模型目录缺失？），"
                           "唤醒词识别将不可用: %s", path)

    async def start(self, *, audio: bool = False, audio_out: bool = False) -> None:
        """启动 WS 服务；audio=True 装语音输入，audio_out=True 装语音输出。"""
        # 助手表：config 是唯一源 → 生成 KWS 词表 + welcome 回报用
        self._assistants = self._load_assistants()
        self._ensure_keywords(self._assistants)
        self.server.set_assistants(self._assistants)

        await self.server.start()

        # outbound：播报队列（frontend 提供者 = inbound 创建的 frontend）
        def get_frontend():
            return self.inbound.frontend if self.inbound is not None else None

        self.playback = Playback(self.cfg, self.server, get_frontend,
                                 silent=not audio_out)
        self.server.playback = self.playback
        try:
            await self.playback.start()
        except Exception as exc:
            logger.exception("[app] 播报队列启动失败: %s", exc)
            self.playback = None
            self.server.playback = None

        if audio:
            self.inbound = Inbound(self.cfg, self.server, playback=self.playback)
            try:
                self.inbound.enable()
                self.inbound.start()
                logger.info("[app] 语音输入已启动（唤醒词监听中）")
            except Exception as exc:
                logger.exception("[app] 语音输入装配失败，仅保留 WS/播报: %s", exc)
                try:
                    self.inbound.stop()
                except Exception:
                    pass
                self.inbound = None

    async def stop(self) -> None:
        if self.inbound is not None:
            try:
                self.inbound.stop()
            except Exception as exc:
                logger.warning("[app] inbound stop: %s", exc)
        if self.playback is not None:
            try:
                await self.playback.stop()
            except Exception as exc:
                logger.warning("[app] playback stop: %s", exc)
        await self.server.stop()
