"""
配置加载模块。
优先级：环境变量 > config.yaml > 默认值。
跨平台兼容（Windows / Linux）。
"""

import os
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 默认配置（与 config.yaml 结构一致，作为兜底）
DEFAULT_CONFIG: Dict[str, Any] = {
    "voice": {
        "wake_word": {
            "enabled": True,
            "keyword": "赫尔墨斯",
            "sensitivity": 0.5,
        },
        "vad": {
            "engine": "webrtc",
            "mode": 3,
            "silence_threshold_ms": 600,
            "min_speech_ms": 200,
            "wake_guard_sec": 2.5,
        },
        "asr": {
            "engine_model": "16k_zh",
            "needvad": False,
            "voice_format": 1,
            "send_interval_ms": 200,
            "max_reconnect": 3,
        },
        "voiceprint": {
            "enabled": False,
            "threshold": 0.6,
            "lib_dir": "models/voiceprint_lib",
            "auto_register": True,
            "min_register_sec": 1.5,
            "speaker_id_cache": True,
            "speaker_names": {},
        },
        "tts": {
            "enabled": True,
            "voice": "zh-CN-XiaoxiaoNeural",
            "rate": 0.0,
            "volume": 0.0,
        },
    }
}


def load_config(path: str | None = None) -> Dict[str, Any]:
    """
    加载配置：合并 YAML 文件 + 环境变量覆盖。
    """
    # 1. 加载 .env
    dotenv_path = PROJECT_ROOT / ".env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path)

    # 2. 加载 YAML
    config_path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    config: Dict[str, Any] = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            user_config = yaml.safe_load(f) or {}
            config = _deep_merge(DEFAULT_CONFIG, user_config)
    else:
        config = DEFAULT_CONFIG.copy()

    # 3. 环境变量覆盖（腾讯云凭据）
    secret_id = os.getenv("VOICE_SecretId")
    secret_key = os.getenv("VOICE_SecretKey")
    app_id = os.getenv("VOICE_AppId")

    if secret_id:
        config.setdefault("tencent", {})["secret_id"] = secret_id
    if secret_key:
        config.setdefault("tencent", {})["secret_key"] = secret_key
    if app_id:
        config["tencent"]["app_id"] = app_id

    return config


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """递归合并两个 dict，override 的 key 覆盖 base。"""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
