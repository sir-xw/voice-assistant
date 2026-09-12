#!/usr/bin/env python3
"""
语音模型下载工具（voice_service 版）：按 `config.yaml` 把缺失的 sherpa-onnx 模型
下载到 **`voice_service/models/`**（该目录已 gitignore，部署/克隆后需自行下载）。

模型存放约定：

    models/sherpa-kws/<model_name>/     # KWS 唤醒词（.tar.bz2 解压）
    models/sherpa-asr/<model_name>/     # 本地 ASR（可选，本仓库 tests 的离线 ASR 用）
    models/sherpa-spk/                  # 说话人分离（pyannote）+ 声纹提取（CAM++）
    models/voiceprint_lib/              # 声纹特征库（运行时生成，这里仅建目录）

配置来源 `config.yaml` 的 `voice_service:` 段：

    kws.model_name             KWS 唤醒词（必需，缺省用内置默认）
    models.asr                 本地 ASR（可选，配置了才下载）
    models.speaker / campplus  说话人分离 / 声纹提取（voiceprint.enabled=true 时下载）

用法（在 `voice_service/` 目录下）：

    python -u tools/download_models.py            # 检查并下载缺失的模型
    python -u tools/download_models.py --check    # 仅检查缺失，不下载
    python -u tools/download_models.py --force    # 已存在也重新下载
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]        # voice_service/
CONFIG_PATH = ROOT / "config.yaml"
ENV_PATH = ROOT / ".env"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("download_models")

# sherpa-onnx 官方模型发布源
BASE_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/"

DEFAULT_KWS_NAME = "sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20"
DEFAULT_PYANNOTE = "sherpa-onnx-pyannote-segmentation-3-0"
DEFAULT_CAMPPLUS = "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx"

MODELS_ROOT = ROOT / "models"


def load_cfg() -> dict:
    """读取 voice_service 配置（缺文件/解析失败时返回空 dict → 用内置默认）。"""
    try:
        from voice_service.service_config import load_voice_service_config
        cfg = load_voice_service_config(str(CONFIG_PATH), env_file=str(ENV_PATH))
        return cfg.raw or {}
    except Exception as exc:
        logger.warning("读取配置失败（用内置默认）: %s", exc)
        return {}


# ─── 检查与下载 ──────────────────────────────────────────


def dir_ready(path: Path) -> bool:
    return path.is_dir() and any(path.iterdir())


def download_to(url: str, dest: Path) -> None:
    """下载 url 到 dest（dest 为文件路径）。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("⬇️  %s", url)
    logger.info("   → %s", dest)
    with urllib.request.urlopen(url, timeout=60) as resp, open(dest, "wb") as f:
        shutil.copyfileobj(resp, f)
    logger.info("   完成（%d 字节）", dest.stat().st_size)


def ensure_tar_model(name: str, category: str, force: bool = False) -> bool:
    """下载并解压 <category>/<name>.tar.bz2 到 models/<category>/<name>/。"""
    dest_dir = MODELS_ROOT / category / name
    if dir_ready(dest_dir) and not force:
        logger.info("✔ %s/%s 已就绪", category, name)
        return True
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    url = f"{BASE_URL}{category}-models/{name}.tar.bz2"
    tmp_dir = Path(tempfile.mkdtemp(prefix="sherpa-dl-"))
    try:
        archive = tmp_dir / f"{name}.tar.bz2"
        download_to(url, archive)
        dest_dir.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, "r:bz2") as tf:
            tf.extractall(dest_dir.parent)
        # 解压产物可能直接是 <name>/ 或含一层目录，规整到 <name>/
        if not dest_dir.is_dir():
            candidates = [p for p in dest_dir.parent.iterdir()
                          if p.is_dir() and p.name != name]
            if candidates:
                src = candidates[0]
                if src != dest_dir:
                    src.rename(dest_dir)
        logger.info("✔ %s/%s 就绪", category, name)
        return True
    except Exception as exc:
        logger.error("下载 %s/%s 失败: %s", category, name, exc)
        return False
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def ensure_file_model(filename: str, category: str, force: bool = False) -> bool:
    """下载单个文件模型（如 CAM++ onnx）到 models/<category>/<filename>。"""
    dest = MODELS_ROOT / category / filename
    if dest.is_file() and not force:
        logger.info("✔ %s/%s 已就绪", category, filename)
        return True
    url = f"{BASE_URL}speaker-recongition-models/{filename}"
    try:
        download_to(url, dest)
        logger.info("✔ %s/%s 就绪", category, filename)
        return True
    except Exception as exc:
        logger.error("下载 %s/%s 失败: %s", category, filename, exc)
        return False


# ─── 主流程 ──────────────────────────────────────────────


def main() -> int:
    global MODELS_ROOT
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="仅检查缺失，不下载")
    parser.add_argument("--force", action="store_true", help="已存在的模型也重新下载")
    parser.add_argument("--models-dir", default=None, help="模型根目录（默认 voice_service/models）")
    parser.add_argument("--config", default=str(CONFIG_PATH), help="配置文件路径")
    args = parser.parse_args()

    if args.models_dir:
        MODELS_ROOT = Path(args.models_dir).expanduser().resolve()

    cfg = load_cfg()
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    logger.info("模型目录: %s", MODELS_ROOT)
    logger.info("配置文件: %s", args.config)

    ok = True
    kws = (cfg.get("kws") or {}) if isinstance(cfg, dict) else {}
    models_cfg = (cfg.get("models") or {}) if isinstance(cfg, dict) else {}
    vp = (cfg.get("voiceprint") or {}) if isinstance(cfg, dict) else {}

    # 1) KWS 唤醒词（必需）
    kws_name = kws.get("model_name") or DEFAULT_KWS_NAME
    logger.info("== KWS 唤醒词: %s ==", kws_name)
    if args.check:
        if not dir_ready(MODELS_ROOT / "sherpa-kws" / kws_name):
            logger.warning("缺失: sherpa-kws/%s", kws_name)
            ok = False
    else:
        ok &= ensure_tar_model(kws_name, "sherpa-kws", force=args.force)

    # 2) 本地 ASR（可选，models.asr 配置了才下载）
    asr_name = models_cfg.get("asr")
    if asr_name:
        logger.info("== 本地 ASR: %s ==", asr_name)
        if args.check:
            if not dir_ready(MODELS_ROOT / "sherpa-asr" / asr_name):
                logger.warning("缺失: sherpa-asr/%s", asr_name)
                ok = False
        else:
            ok &= ensure_tar_model(asr_name, "sherpa-asr", force=args.force)

    # 3) 说话人识别（voiceprint.enabled 时）
    if vp.get("enabled", False):
        spk_name = models_cfg.get("speaker") or DEFAULT_PYANNOTE
        camp_name = models_cfg.get("campplus") or DEFAULT_CAMPPLUS
        logger.info("== 说话人识别: %s / %s ==", spk_name, camp_name)
        if args.check:
            if not dir_ready(MODELS_ROOT / "sherpa-spk" / spk_name):
                logger.warning("缺失: sherpa-spk/%s", spk_name)
                ok = False
            if not (MODELS_ROOT / "sherpa-spk" / camp_name).is_file():
                logger.warning("缺失: sherpa-spk/%s", camp_name)
                ok = False
        else:
            ok &= ensure_tar_model(spk_name, "sherpa-spk", force=args.force)
            ok &= ensure_file_model(camp_name, "sherpa-spk", force=args.force)

    # 4) 声纹特征库目录（运行时生成，仅确保存在）
    lib_dir = MODELS_ROOT / "voiceprint_lib"
    lib_dir.mkdir(parents=True, exist_ok=True)
    logger.info("✔ %s 就绪（特征库由运行时生成）", lib_dir)

    logger.info("完成%s", "" if ok else "（存在缺失）")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
