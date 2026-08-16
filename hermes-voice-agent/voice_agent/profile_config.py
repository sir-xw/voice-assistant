"""
profile 目录配置与路径约定（voice-platform 公共模块）。

约定（hermes profile 目录，默认 ~/.hermes，HERMES_HOME 可覆盖）：
- 语音配置：``<profile>/voice-agent.yaml``（结构同 voice_agent config.yaml 的 voice 段）
- 本地模型：``<profile>/models/sherpa-kws/<model_name>/``、
  ``<profile>/models/sherpa-asr/<model_name>/``、``<profile>/models/sherpa-spk/``
- 声纹特征库：``<profile>/models/voiceprint_lib/``

本模块只依赖标准库与 yaml，不 import gateway.*，供适配器与 tools 脚本共用。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

DEFAULT_KWS_NAME = "sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20"


def profile_root() -> Path:
    """hermes profile 目录（HERMES_HOME 环境变量，否则 ~/.hermes）。"""
    env = os.environ.get("HERMES_HOME", "").strip()
    if env:
        return Path(env)
    return Path.home() / ".hermes"


def voice_config_path() -> Path:
    """语音配置文件路径（约定）：<profile>/voice-agent.yaml。"""
    return profile_root() / "voice-agent.yaml"


def load_voice_config() -> Dict[str, Any]:
    """读取语音配置 voice-agent.yaml（不存在/解析失败返回空 dict）。

    文件顶层与 voice_agent 的 config.yaml 同构（含 ``voice:`` 键）；
    若无 ``voice:`` 键则把顶层直接视为语音配置段。
    """
    path = voice_config_path()
    if not path.is_file():
        return {}
    try:
        import yaml
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            return {}
        return data.get("voice", data) if isinstance(data.get("voice"), dict) else data
    except Exception:
        return {}


def models_dir() -> Path:
    """profile 模型根目录（约定）：<profile>/models/。"""
    return profile_root() / "models"


def resolve_kws_model_dir(cfg: Dict[str, Any]) -> Path:
    """解析 sherpa KWS 模型目录（约定路径）。

    约定：<profile>/models/sherpa-kws/<model_name>，配置只需指定
    ``kws.model_name``，无需指定 model_dir。显式配置 model_dir 仍兼容。
    """
    kws = (cfg.get("kws") or {}) if isinstance(cfg, dict) else {}
    raw = kws.get("model_dir")
    if raw:
        p = Path(raw)
        if p.is_absolute():
            return p
        return voice_config_path().parent / p
    name = kws.get("model_name") or DEFAULT_KWS_NAME
    return models_dir() / "sherpa-kws" / name


def resolve_voiceprint_lib_dir(cfg: Dict[str, Any]) -> Path:
    """声纹特征库目录（约定）：<profile>/models/voiceprint_lib/。"""
    vp = (cfg.get("voiceprint") or {}) if isinstance(cfg, dict) else {}
    raw = vp.get("lib_dir")
    if raw:
        p = Path(raw)
        if p.is_absolute():
            return p
        return voice_config_path().parent / p
    return models_dir() / "voiceprint_lib"
