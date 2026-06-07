#!/usr/bin/env python3
"""
sherpa-onnx 离线 TTS 合成测试。

使用 sherpa_onnx.OfflineTts 进行本地语音合成（无需腾讯云）。
性能太差，没有用到
模型下载：
https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-zh_CN-xiao_ya-medium-int8.tar.bz2


用法:
    # 默认模型（vits-piper-zh_CN-xiao_ya-medium-int8）合成文本
    python -u tests/test_sherpa_tts.py "你好，欢迎使用语音助手。"

    # 保存到指定文件
    python -u tests/test_sherpa_tts.py "今天天气怎么样？" --output test.wav

    # 指定模型目录
    python -u tests/test_sherpa_tts.py "你好世界" --model-dir /path/to/model

    # 播放合成结果（需要 sounddevice）
    python -u tests/test_sherpa_tts.py "你好" --play

    # 批量测试多段文本
    python -u tests/test_sherpa_tts.py --batch

依赖:
    pip install sherpa-onnx soundfile
    pip install sounddevice          (--play 模式)
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional
import os

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sherpa_tts_test")

SAMPLE_RATE = 16000

# 默认模型路径（相对项目根目录）
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "sherpa-tts"
DEFAULT_MODEL_NAME = "vits-piper-zh_CN-xiao_ya-medium-int8"
DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / DEFAULT_MODEL_NAME

# 常用中文测试文本
TEST_TEXTS = [
    "你好，欢迎使用智能语音助手。",
    "今天天气怎么样？",
    "请帮我查一下明天北京的天气。",
    "好的，已为您设置提醒。",
    "123456，拨打 110 或者 18920240511。",
]


# ---------------------------------------------------------------------------
# 模型文件查找
# ---------------------------------------------------------------------------

def find_model_files(model_path: Path):
    """
    在模型目录中查找 VITS-Piper 模型所需的文件。
    自动尝试常见文件名。

    返回 (model_file, tokens_file, data_dir) 或抛异常。
    """
    if not model_path.exists():
        raise FileNotFoundError(f"模型目录不存在: {model_path}")

    # --- 模型文件：尝试常见命名 ---
    model_candidates = [
        "model.onnx",
        "model.int8.onnx",
        "zh_CN-xiao_ya-medium.onnx",
        "zh_CN-xiao_ya-medium.int8.onnx",
    ]
    model_file = None
    for name in model_candidates:
        f = model_path / name
        if f.exists():
            model_file = f
            break
    # 兜底：任意 .onnx
    if model_file is None:
        onnx_files = sorted(model_path.glob("*.onnx"))
        if onnx_files:
            model_file = onnx_files[0]

    if model_file is None:
        raise FileNotFoundError(f"未找到 .onnx 模型文件（{model_path}）")

    # --- tokens ---
    tokens_file = model_path / "tokens.txt"
    if not tokens_file.exists():
        raise FileNotFoundError(f"未找到 tokens.txt（{model_path}）")

    # --- espeak-ng-data（Piper 模型可选） ---
    data_dir = model_path / "espeak-ng-data"
    if not data_dir.exists() or not data_dir.is_dir():
        data_dir = None

    return model_file, tokens_file, data_dir


# ---------------------------------------------------------------------------
# TTS 引擎
# ---------------------------------------------------------------------------

def create_tts(model_path: Path, num_threads: int = 2):
    """
    创建 sherpa-onnx TTS 引擎。
    返回 (tts, model_file, tokens_file, data_dir, model_dir_name)。
    """
    import sherpa_onnx

    model_file, tokens_file, data_dir = find_model_files(model_path)

    logger.info("加载 TTS 模型:")
    logger.info("  模型目录: %s", model_path)
    logger.info("  模型文件: %s", model_file)
    logger.info("  tokens:   %s", tokens_file)
    if data_dir:
        logger.info("  espeak-ng-data: %s", data_dir)
    else:
        logger.info("  espeak-ng-data: 无")

    tts_config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                model=str(model_file),
                tokens=str(tokens_file),
                data_dir=str(data_dir) if data_dir else "",
                lexicon=os.path.join(model_path, 'lexicon.txt'),
            ),
            provider="cpu",
            debug=False,
            num_threads=num_threads,
        ),
        rule_fsts="",
        max_num_sentences=1,
    )
    if not tts_config.validate():
        raise ValueError("TTS 配置验证失败，请检查模型文件")

    tts = sherpa_onnx.OfflineTts(tts_config)
    return tts, model_file, tokens_file, data_dir, model_path.name


def synthesize(tts, text: str, sid: int = 0, speed: float = 1.0,
               output_path: Optional[Path] = None,
               play: bool = False, verbose: bool = True) -> float:
    """
    合成一段文本，返回音频时长（秒）。

    参数:
        tts: OfflineTts 实例
        text: 待合成文本
        sid: 说话人 ID（多说话人模型有效）
        speed: 语速（1.0 正常）
        output_path: 保存为 WAV 文件（None 则不保存）
        play: 是否播放
        verbose: 是否打印详情

    返回:
        合成音频时长（秒）
    """
    import sherpa_onnx  # noqa

    t0 = time.time()
    gen_config = sherpa_onnx.GenerationConfig()
    gen_config.sid = sid
    gen_config.speed = speed
    gen_config.silence_scale = 0.2

    audio = tts.generate(text, gen_config)
    elapsed = time.time() - t0

    if len(audio.samples) == 0:
        logger.error("合成失败，请检查模型和文本")
        return 0.0

    audio_duration = len(audio.samples) / audio.sample_rate
    rtf = elapsed / audio_duration if audio_duration > 0 else 0

    if verbose:
        print(f"  📝 文本:    {text}")
        print(f"  ⏱  合成耗时: {elapsed:.2f}s")
        print(f"  📏 音频时长: {audio_duration:.2f}s")
        print(f"  ⚡ RTF:     {elapsed:.2f}/{audio_duration:.2f} = {rtf:.3f}")
        print(f"  🔊 采样率:   {audio.sample_rate} Hz")
        print(f"  📦 样本数:   {len(audio.samples)}")

    # 保存 WAV
    if output_path:
        import soundfile as sf
        sf.write(
            str(output_path),
            audio.samples,
            samplerate=audio.sample_rate,
            subtype="PCM_16",
        )
        if verbose:
            print(f"  💾 保存到:  {output_path}")

    # 播放
    if play:
        _play_audio(audio.samples, audio.sample_rate)

    return audio_duration


def _play_audio(samples: np.ndarray, sample_rate: int):
    """播放 PCM float32 音频。"""
    try:
        import sounddevice as sd
        logger.info("🔊 播放中...")
        sd.play(samples, samplerate=sample_rate)
        sd.wait()
    except ImportError:
        logger.warning("sounddevice 未安装，无法播放")
    except Exception as e:
        logger.warning(f"播放失败: {e}")


def run_batch(tts, texts: List[str], model_name: str,
              output_dir: Optional[Path] = None,
              play: bool = False):
    """批量合成多段文本。"""
    print(f"\n{'='*60}")
    print(f"📚 批量 TTS 测试 — {model_name}")
    print(f"{'='*60}")

    total_duration = 0.0

    for i, text in enumerate(texts, 1):
        print(f"\n[{i}/{len(texts)}] ", end="")
        out_path = None
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            out_path = output_dir / f"test_{i:02d}.wav"
        dur = synthesize(tts, text, output_path=out_path, play=play)
        total_duration += dur
        if i < len(texts):
            print()

    print(f"\n{'='*60}")
    print(f"📊 合计: {len(texts)} 段, 总时长 {total_duration:.1f}s")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="sherpa-onnx 离线 TTS 合成测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("text", type=str, nargs="?",
                   help="待合成文本（不指定则使用默认测试文本）")
    p.add_argument("--model-dir", type=str, default=None,
                   help=f"模型目录（默认 {DEFAULT_MODEL_PATH}）")
    p.add_argument("--output", "-o", type=str, default=None,
                   help="输出 WAV 文件路径（默认不保存）")
    p.add_argument("--play", "-p", action="store_true",
                   help="合成后播放音频（需 sounddevice）")
    p.add_argument("--sid", type=int, default=0,
                   help="说话人 ID（多说话人模型，默认 %(default)s）")
    p.add_argument("--speed", type=float, default=1.0,
                   help="语速（0.5-2.0，默认 %(default)s）")
    p.add_argument("--num-threads", type=int, default=2,
                   help="推理线程数（默认 %(default)s）")
    p.add_argument("--batch", action="store_true",
                   help="批量测试多段预设文本")
    p.add_argument("--output-dir", type=str, default=None,
                   help="批量测试的输出目录（与 --batch 搭配）")

    args = p.parse_args()

    # --- 确定模型路径 ---
    if args.model_dir:
        model_path = Path(args.model_dir)
    else:
        model_path = DEFAULT_MODEL_PATH

    # --- 创建 TTS 引擎 ---
    try:
        tts, model_file, tokens_file, data_dir, model_name = create_tts(
            model_path, args.num_threads
        )
    except (FileNotFoundError, ValueError, ImportError) as e:
        logger.error("❌ %s", e)
        sys.exit(1)

    # --- 确定输出路径 ---
    output_path = Path(args.output) if args.output else None

    # --- 执行 ---
    if args.batch:
        run_batch(tts, TEST_TEXTS, model_name,
                  output_dir=Path(args.output_dir) if args.output_dir else None,
                  play=args.play)

    elif args.text:
        print(f"\n{'='*60}")
        print(f"🔊 TTS 合成 — {model_name}")
        print(f"{'='*60}")
        synthesize(tts, args.text, sid=args.sid, speed=args.speed,
                   output_path=output_path, play=args.play)
        print(f"{'='*60}")

    else:
        text = TEST_TEXTS[0]
        print(f"\n{'='*60}")
        print(f"🔊 TTS 合成 — {model_name}")
        print(f"{'='*60}")
        logger.info("未指定文本，使用默认测试文本: %s", text)
        synthesize(tts, text, sid=args.sid, speed=args.speed,
                   output_path=output_path, play=args.play)
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
