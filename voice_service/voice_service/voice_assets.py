"""
提示音资产（assets）：路径解析 + wav → 16k 单声道 int16 PCM。

资产 wav 随包分发（package-data：voice_service/assets/*.wav）。读取时统一
归一化到 16kHz/单声道/16bit PCM（与播放器/TTS 输出一致），采样率不符自动
重采样（scipy），失败返回 None。
"""

from __future__ import annotations

import logging
import wave
from pathlib import Path
from typing import Optional

logger = logging.getLogger("voice_service.voice_assets")

PKG_ASSETS = Path(__file__).resolve().parent / "assets"
ASSET_NAMES = {
    "prompt": "prompt.wav",          # 唤醒命中提示音
    "notification": "notification.wav",  # 最终回复播完提示音
    "farewell": "farewell.wav",      # 对话结束/超时告别语
    "wait_cue": "wait_cue_4_scale.wav",  # 等待回复循环提示音
}


def asset_path(name: str) -> Path:
    return PKG_ASSETS / ASSET_NAMES.get(name, name)


def asset_duration_sec(name: str) -> float:
    """读取 wav 实际时长（秒），失败返回 1.0（供唤醒静音保护期用）。"""
    try:
        with wave.open(str(asset_path(name)), "rb") as wf:
            return wf.getnframes() / wf.getframerate()
    except Exception:
        return 1.0


def asset_pcm16(name: str) -> Optional[bytes]:
    """读取资产并归一化为 16k/单声道/int16 PCM bytes；失败返回 None。"""
    path = asset_path(name)
    if not path.is_file():
        logger.warning("[assets] 缺失: %s", path)
        return None
    try:
        import numpy as np
        with wave.open(str(path), "rb") as wf:
            sr = wf.getframerate()
            ch = wf.getnchannels()
            width = wf.getsampwidth()
            data = wf.readframes(wf.getnframes())
        samples = np.frombuffer(data, dtype=np.int16)  # 假设 16bit
        if ch > 1:
            samples = samples.reshape(-1, ch).mean(axis=1).astype(np.int16)
        if sr != 16000:
            from scipy import signal
            n_target = round(len(samples) * 16000 / sr)
            out = signal.resample_poly(samples.astype(np.float32), 16000, sr)
            if len(out) > n_target:
                out = out[:n_target]
            elif len(out) < n_target:
                out = np.pad(out, (0, n_target - len(out)))
            samples = np.clip(out, -32768, 32767).astype(np.int16)
        return samples.tobytes()
    except Exception as exc:
        logger.warning("[assets] 读取失败 %s: %s", path, exc)
        return None
