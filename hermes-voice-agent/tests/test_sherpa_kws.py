#!/usr/bin/env python3
"""
sherpa-onnx 关键词唤醒测试（支持中文）。

自动下载模型，从麦克风实时检测自定义唤醒词。
中文关键词使用 pypinyin 自动转换为拼音格式。

用法:
    python -u tests/test_sherpa_kws.py --keyword "你好小娜"
    python -u tests/test_sherpa_kws.py --keyword "赫尔墨斯" --threshold 0.3
    python -u tests/test_sherpa_kws.py --list-devices

首次运行自动下载模型（~38MB）。
"""

import argparse
import json
import logging
import sys
import urllib.request
import tarfile
from pathlib import Path

import sherpa_onnx
# 注意：sounddevice 在用到时才 import（避免与 sherpa_onnx C 扩展冲突）

logging.basicConfig(level=logging.INFO, format="%(message)s")

SAMPLE_RATE = 16000
MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "sherpa-kws"
MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/"
    "sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20.tar.bz2"
)
MODEL_NAME = "sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20"


def download_model():
    model_path = MODEL_DIR / MODEL_NAME
    if model_path.exists():
        return model_path
    print(f"📥 下载模型 (~38MB)...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tarball = MODEL_DIR / "model.tar.bz2"
    urllib.request.urlretrieve(MODEL_URL, tarball)
    print(f"   解压中...")
    with tarfile.open(tarball, "r:bz2") as tar:
        tar.extractall(path=MODEL_DIR)
    tarball.unlink()
    print(f"   ✅ {model_path}")
    return model_path


def _split_pinyin_syllable(syllable: str) -> str:
    """
    将带声调的拼音音节拆分为声母+韵母（声调跟在韵母后），
    符合 sherpa-onnx keywords.txt 格式。

    例如:
        nǐ  → n ǐ
        hǎo → h ǎo
        xiǎo → x iǎo
        nà  → n à
        a   → a
        é   → é
    """
    # 声母列表（包括复合声母）
    initials = [
        "zh", "ch", "sh",
        "b", "p", "m", "f", "d", "t", "n", "l",
        "g", "k", "h", "j", "q", "x",
        "r", "z", "c", "s", "y", "w",
    ]
    # 先尝试匹配最长的声母
    matched_init = ""
    for init in sorted(initials, key=len, reverse=True):
        if syllable.startswith(init):
            matched_init = init
            break
    if matched_init:
        final = syllable[len(matched_init):]
        if final:
            return f"{matched_init} {final}"
        else:
            return matched_init
    else:
        # 没有声母（如 "a", "é", "an" 等）
        return syllable


def auto_format(text: str) -> str:
    """
    自动生成 sherpa-onnx 关键词格式：
    - 纯英文：直接保留（alexa → alexa）
    - 含中文：使用 pypinyin 转换为拼音，每个音节拆成声母+韵母
    """
    has_cjk = any('\u4e00' <= c <= '\u9fff' for c in text)
    if has_cjk:
        import pypinyin
        syllables = pypinyin.pinyin([text])
        print(syllables)
        parts = []
        for syl in syllables:
            for syl1 in syl:
                parts.append(_split_pinyin_syllable(syl1))
        return "  ".join(parts)
    # 英文直接保留
    return text.strip().lower()


def test_keyword(keyword: str = None, pinyin_text: str = None,
                 threshold: float = 0.25, score: float = 1.0,
                 chunk: int = 8):
    import sounddevice as sd

    model_path = download_model()

    if pinyin_text:
        final_kw = pinyin_text
    elif keyword:
        final_kw = auto_format(keyword)
    else:
        print("❌ 请指定 --keyword 或 --pinyin")
        return

    print(f"\n{'='*50}")
    print(f"🔊 sherpa-onnx KWS")
    print(f"{'='*50}")
    print(f"   唤醒词:  '{keyword or pinyin_text}'")
    print(f"   拼音:    {final_kw}")
    print(f"   阈值:    {threshold}")
    print(f"   chunk:   {chunk}")

    kw_file = MODEL_DIR / "keywords.txt"
    kw_file.write_text(final_kw, encoding="utf-8")

    sfx = f"chunk-{chunk}-left-64"
    # 优先使用 int8 模型（体积小，加载快）
    enc = model_path / f"encoder-epoch-13-avg-2-{sfx}.int8.onnx"
    if not enc.exists():
        enc = model_path / f"encoder-epoch-13-avg-2-{sfx}.onnx"
    dec = model_path / f"decoder-epoch-13-avg-2-{sfx}.onnx"
    joi = model_path / f"joiner-epoch-13-avg-2-{sfx}.int8.onnx"
    if not joi.exists():
        joi = model_path / f"joiner-epoch-13-avg-2-{sfx}.onnx"
    tok = model_path / "tokens.txt"

    sys.stdout.flush()
    print(f"   模型路径: {model_path}"); sys.stdout.flush()
    print(f"   编码器: {enc}"); sys.stdout.flush()
    print(f"   解码器: {dec}"); sys.stdout.flush()
    print(f"   联合器: {joi}"); sys.stdout.flush()
    print(f"   词表:   {tok}"); sys.stdout.flush()
    print(f"   加载模型...")
    spotter = sherpa_onnx.KeywordSpotter(
        tokens=str(tok), encoder=str(enc), decoder=str(dec), joiner=str(joi),
        num_threads=1, max_active_paths=4,
        keywords_file=str(kw_file),
        keywords_score=score, keywords_threshold=threshold,
        num_trailing_blanks=1, provider="cpu",
    )

    print(f"\n🎤 监听中 — 说出「{keyword or pinyin_text}」")
    print(f"   Ctrl+C 退出\n")

    idx = 0
    step = int(0.1 * SAMPLE_RATE)
    try:
        stream = spotter.create_stream()
    except Exception as e:
        print(f"❌ create_stream 失败: {e}")
        import traceback; traceback.print_exc()
        return

    try:
        with sd.InputStream(channels=1, dtype="float32",
                            samplerate=SAMPLE_RATE) as s:
            while True:
                samples, _ = s.read(step)
                samples = samples.reshape(-1)
                stream.accept_waveform(SAMPLE_RATE, samples)
                while spotter.is_ready(stream):
                    spotter.decode_stream(stream)
                    result = spotter.get_result(stream)
                    if result:
                        idx += 1
                        print(f"\n🔊 检测到! (第{idx}次)")
                        try:
                            d = json.loads(result)
                            print(f"   关键词: {d.get('keyword','?')}")
                        except json.JSONDecodeError:
                            print(f"   {result}")
                        spotter.reset_stream(stream)
    except KeyboardInterrupt:
        print(f"\n\n📊 检测 {idx} 次")


def list_devices():
    import sounddevice as sd
    print(sd.query_devices())


def main():
    p = argparse.ArgumentParser(description="sherpa-onnx 关键词唤醒")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--keyword", help="中文/英文唤醒词（自动转拼音）")
    g.add_argument("--pinyin", help="手动指定拼音格式（兼容旧用法）")
    p.add_argument("--threshold", type=float, default=0.25)
    p.add_argument("--score", type=float, default=1.0)
    p.add_argument("--chunk", type=int, default=8, choices=[8, 16])
    p.add_argument("--list-devices", action="store_true")
    args = p.parse_args()

    if args.list_devices:
        list_devices()
        return
    if not args.keyword and not args.pinyin:
        p.print_help()
        return

    test_keyword(args.keyword, args.pinyin, args.threshold, args.score, args.chunk)


if __name__ == "__main__":
    main()
