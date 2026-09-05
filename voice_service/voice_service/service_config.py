"""
服务配置加载（Voice Service 侧，独立于 hermes profile）。

配置来源（优先级 高→低）：
1. 环境变量（VOICE_SecretId / VOICE_SecretKey / VOICE_AppId 等腾讯云凭据）
2. 命令行指定配置文件（默认仓库根 ./config.yaml 或 ./voice_service.yaml）
3. 内置默认值（见下）

配置文件顶层是 ``voice_service:`` 段（结构见架构文档 §4.4）——
不含 agents/会话类（会话归 hermes 侧 gateway 插件）。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("voice_service.config")

DEFAULT_CONFIG_PATHS = ("config.yaml", "voice_service.yaml")

# 腾讯云凭据 env 键（与旧 voice_agent 一致；gitignore 于 voice_service/ 项目根 .env（部署时提供））
ENV_SECRET_ID = "VOICE_SecretId"
ENV_SECRET_KEY = "VOICE_SecretKey"
ENV_APP_ID = "VOICE_AppId"


@dataclass
class ServiceConfig:
    """voice_service: 顶层服务段（network/心跳/等待音超时）。"""
    host: str = "127.0.0.1"
    port: int = 8765
    token: str = ""                      # 空 = 仅限本机
    ping_interval_sec: float = 20.0
    heartbeat_timeout_sec: float = 60.0
    wait_reply_timeout_sec: float = 45.0


@dataclass
class TencentCreds:
    secret_id: str = ""
    secret_key: str = ""
    app_id: str = ""


@dataclass
class VoiceServiceConfig:
    """voice_service: 整段配置的 Python 视图。

    只落地骨架阶段必需的字段；kws/vad/asr/tts/mic/assets/voiceprint 等
    语音段在 M2/M3 装配时再展开（保留原始 dict 供装配代码取用）。
    """
    service: ServiceConfig = field(default_factory=ServiceConfig)
    tencent: TencentCreds = field(default_factory=TencentCreds)
    raw: Dict[str, Any] = field(default_factory=dict)   # voice_service: 段原始内容
    config_path: Optional[Path] = None

    # 便捷访问语音子段（M2/M3 装配使用）
    @property
    def voice(self) -> Dict[str, Any]:
        return self.raw


def _merge_service(raw_service: Dict[str, Any]) -> ServiceConfig:
    cfg = ServiceConfig()
    for k in ("host", "token"):
        if k in raw_service:
            setattr(cfg, k, raw_service[k])
    for k, cast in (("port", int), ("ping_interval_sec", float),
                    ("heartbeat_timeout_sec", float),
                    ("wait_reply_timeout_sec", float)):
        if k in raw_service:
            setattr(cfg, k, cast(raw_service[k]))
    return cfg


def _load_tencent_env() -> TencentCreds:
    return TencentCreds(
        secret_id=os.getenv(ENV_SECRET_ID, ""),
        secret_key=os.getenv(ENV_SECRET_KEY, ""),
        app_id=os.getenv(ENV_APP_ID, ""),
    )


def load_voice_service_config(config_path: Optional[str] = None,
                              *, env_file: Optional[str] = None) -> VoiceServiceConfig:
    """加载配置。env_file 存在时先载入 .env（不覆盖已有环境变量）。"""
    if env_file is None:
        env_file = ".env"
    env_p = Path(env_file)
    if env_p.is_file():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_p, override=False)
        except Exception as exc:  # dotenv 缺失时仅告警
            logger.warning("载入 %s 失败（可忽略）: %s", env_p, exc)

    path: Optional[Path] = None
    if config_path:
        path = Path(config_path)
    else:
        for name in DEFAULT_CONFIG_PATHS:
            cand = Path(name)
            if cand.is_file():
                path = cand
                break
    raw: Dict[str, Any] = {}
    if path is not None and path.is_file():
        try:
            import yaml
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            seg = data.get("voice_service", data) if isinstance(data.get("voice_service"), dict) else {}
            if isinstance(seg, dict):
                raw = seg
        except Exception as exc:
            logger.warning("配置文件解析失败 %s（使用默认）: %s", path, exc)
    elif config_path:
        logger.warning("配置文件不存在: %s（使用默认）", config_path)

    svc_raw = raw.get("service") or {}
    return VoiceServiceConfig(
        service=_merge_service(svc_raw) if isinstance(svc_raw, dict) else ServiceConfig(),
        tencent=_load_tencent_env(),
        raw=raw,
        config_path=path,
    )
