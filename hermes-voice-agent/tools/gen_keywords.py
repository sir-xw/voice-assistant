#!/usr/bin/env python3
"""
sherpa-onnx 唤醒词生成工具。

从 raw_keywords.txt 读取原始关键词列表，生成 sherpa-onnx 格式的 keywords.txt。

raw_keywords.txt 格式（每行）:
    <phoneme_text> @<original_with_underscores>

    - <phoneme_text>: sherpa-onnx 格式的发音文本。中文会被自动转换为拼音；
                      英文保持不变（用户提供 ARPABET 或小写均可）。
    - @<original>:    原始关键词文本。下划线 _ 代替空格。

示例 raw_keywords.txt:
    LIGHT UP @LIGHT_UP
    文森特卡索 @文森特卡索
    你好小娜 @你好小娜

转换后 keywords.txt:
    L AY1 T AH1 P @LIGHT_UP
    w én s ēn t è k ǎ s uǒ @文森特卡索
    n ǐ h ǎo x iǎo n à @你好小娜

用法:
    # 使用默认路径生成
    python -u tools/gen_keywords.py

    # 指定 raw 文件
    python -u tools/gen_keywords.py --input /path/to/raw_keywords.txt

    # 指定模型目录
    python -u tools/gen_keywords.py --model-dir /path/to/model

依赖:
    pip install pypinyin   (中文转拼音)
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

# 默认模型路径（与 config.yaml 默认值一致）
DEFAULT_MODEL_DIR = (
    Path(__file__).resolve().parent.parent
    / "models" / "sherpa-kws"
    / "sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20"
)


# ---------------------------------------------------------------------------
# 拼音转换（与 voice_frontend.py 中的 _split_pinyin_syllable / _auto_format_keyword 一致）
# ---------------------------------------------------------------------------

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
    initials = [
        "zh", "ch", "sh",
        "b", "p", "m", "f", "d", "t", "n", "l",
        "g", "k", "h", "j", "q", "x",
        "r", "z", "c", "s", "y", "w",
    ]
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
        return syllable


def auto_format_keyword(text: str) -> str:
    """
    自动生成 sherpa-onnx 关键词格式：
    - 纯英文：直接保留（传入什么返回什么）
    - 含中文：使用 pypinyin 转换为拼音，每个音节拆成声母+韵母
    """
    has_cjk = any('\u4e00' <= c <= '\u9fff' for c in text)
    if has_cjk:
        import pypinyin
        syllables = pypinyin.pinyin([text])
        parts = []
        for syl in syllables:
            for syl1 in syl:
                parts.append(_split_pinyin_syllable(syl1))
        return "  ".join(parts)
    return text.strip()


# ---------------------------------------------------------------------------
# raw_keywords.txt 解析 → keywords.txt 生成
# ---------------------------------------------------------------------------

def parse_raw_line(line: str) -> Tuple[str, str]:
    """
    解析 raw_keywords.txt 的一行。

    格式: <phoneme_text> @<original>

    返回 (phoneme_text, original_suffix)，original_suffix 包含 @ 前缀。
    例如 "LIGHT UP @LIGHT_UP" → ("LIGHT UP", "@LIGHT_UP")
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return None, None
    if "@" not in line:
        # 没有 @ 后缀，整个行作为 phoneme_text
        return line, ""
    at_idx = line.index("@")
    phoneme_text = line[:at_idx].strip()
    original_suffix = line[at_idx:].strip()  # 包含 @
    return phoneme_text, original_suffix


def convert_phoneme(phoneme_text: str) -> str:
    """
    将 phoneme_text 转换为 sherpa-onnx 格式。
    - 含中文：自动转拼音并拆分声韵母
    - 纯英文/其他：原样保留
    """
    has_cjk = any('\u4e00' <= c <= '\u9fff' for c in phoneme_text)
    if has_cjk:
        return auto_format_keyword(phoneme_text)
    return phoneme_text


def generate_keywords(raw_path: Path, output_path: Path) -> List[str]:
    """
    从 raw_keywords.txt 生成 keywords.txt。

    返回生成的每一行文本（用于日志输出）。
    """
    if not raw_path.exists():
        print(f"❌ 原始文件不存在: {raw_path}")
        sys.exit(1)

    raw_lines = raw_path.read_text(encoding="utf-8").splitlines()
    output_lines = []
    converted_count = 0
    skip_count = 0

    for line in raw_lines:
        phoneme_text, original_suffix = parse_raw_line(line)
        if phoneme_text is None:
            skip_count += 1
            continue

        converted = convert_phoneme(phoneme_text) + " " + original_suffix if original_suffix else convert_phoneme(phoneme_text)
        output_lines.append(converted.strip())

        # 标记哪些被自动转换了
        if phoneme_text != convert_phoneme(phoneme_text):
            print(f"   🔄 {phoneme_text:<30} → {converted.strip()}")
        else:
            print(f"   ✅ {converted.strip()}")
        converted_count += 1

    output_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    print(f"\n  💾 已写入 {converted_count} 个唤醒词 → {output_path}")
    if skip_count:
        print(f"  ⏭️  跳过 {skip_count} 行（空行/注释）")
    return output_lines


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="sherpa-onnx 唤醒词生成工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model-dir", type=str, default=None,
                   help=f"模型目录（默认 {DEFAULT_MODEL_DIR}）")
    p.add_argument("--input", "-i", type=str, default=None,
                   help="raw_keywords.txt 路径（默认 <model-dir>/raw_keywords.txt）")
    p.add_argument("--output", "-o", type=str, default=None,
                   help="输出 keywords.txt 路径（默认 <model-dir>/keywords.txt）")
    p.add_argument("--init", action="store_true",
                   help="创建 raw_keywords.txt 模板（使用 config.yaml 中的 wake_word.keyword）")

    args = p.parse_args()

    # --- 确定路径 ---
    if args.model_dir:
        model_path = Path(args.model_dir)
    else:
        model_path = DEFAULT_MODEL_DIR

    if not model_path.exists():
        print(f"❌ 模型目录不存在: {model_path}")
        print("   请先下载模型或指定正确的 --model-dir")
        sys.exit(1)

    raw_path = Path(args.input) if args.input else model_path / "raw_keywords.txt"
    output_path = Path(args.output) if args.output else model_path / "keywords.txt"

    # --- --init: 创建模板 ---
    if args.init:
        if raw_path.exists():
            print(f"⚠️  文件已存在: {raw_path}")
            print("   使用 --input 指定其他路径，或删除已有文件")
            return
        # 尝试从 config.yaml 读取第一个 agent 的名称
        config_path = model_path.parent.parent / "config.yaml"
        default_keyword = "赫尔墨斯"
        if config_path.exists():
            import yaml
            try:
                cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
                agents = cfg.get("voice", {}).get("agents", [])
                if agents:
                    default_keyword = agents[0].get("name", default_keyword)
            except Exception:
                pass
        raw_path.write_text(
            f"# sherpa-onnx 唤醒词列表\n"
            f"# 格式: <phoneme_text> @<original_with_underscores>\n"
            f"# 中文自动转拼音，英文原样保留\n"
            f"#\n"
            f"{default_keyword} @{default_keyword}\n",
            encoding="utf-8",
        )
        print(f"  ✅ 已创建模板: {raw_path}")
        print(f"     编辑后运行: python -u tools/gen_keywords.py")
        return

    # --- 生成 ---
    generate_keywords(raw_path, output_path)


if __name__ == "__main__":
    main()

