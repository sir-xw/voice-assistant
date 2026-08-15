#!/usr/bin/env python3
"""
sherpa-onnx 唤醒词生成工具。

默认行为：自动读取 config.yaml 中每个 agent 的 keywords 配置（列表形式），
同步生成 raw_keywords.txt，再转换为 sherpa-onnx 格式的 keywords.txt。
agent 未配置 keywords 时回退为 [name]。

raw_keywords.txt 格式（每行）:
    <phoneme_text> @<original_with_underscores>

    - <phoneme_text>: sherpa-onnx 格式的发音文本。中文会被自动转换为拼音；
                      英文保持不变（用户提供 ARPABET 或小写均可）。
    - @<original>:    原始关键词文本，即 config.yaml 中 agent 的 name
                      （下划线 _ 代替空格）。

示例 raw_keywords.txt:
    LIGHT UP @LIGHT_UP
    小布小布 @小布
    你好小布 @小布

转换后 keywords.txt:
    L AY1 T AH1 P @LIGHT_UP
    x iǎo  b ù  x iǎo  b ù @小布
    n ǐ  h ǎo  x iǎo  b ù @小布

用法:
    # 默认：自动读取 config.yaml 的 agents[].keywords，同步 raw_keywords.txt
    #       并生成 keywords.txt
    python -u tools/gen_keywords.py

    # 手动模式：指定 raw 文件（跳过 config.yaml 自动同步）
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
# config.yaml 自动同步
# ---------------------------------------------------------------------------

def load_agents_keywords(config_path: Path) -> List[Tuple[str, List[str]]]:
    """
    从 config.yaml 读取每个 agent 的唤醒词配置。

    返回 [(name, [keywords...]), ...]：
    - keywords 取 agents[].keywords 列表；
    - agent 未配置 keywords 时回退为 [name]；
    - 每个唤醒词文本去空、去重（全局只保留首次出现）。
    配置缺失/读取失败/无 agents 时返回 []。
    """
    if not config_path.exists():
        return []
    try:
        import yaml
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception as e:
        print(f"  ⚠️  读取配置失败，跳过自动同步: {config_path} ({e})")
        return []
    agents = cfg.get("voice", {}).get("agents", [])
    if not agents:
        return []
    result: List[Tuple[str, List[str]]] = []
    seen: set = set()  # 全局去重
    for ac in agents:
        if not isinstance(ac, dict):
            continue
        name = str(ac.get("name") or "").strip()
        if not name:
            continue
        keywords = ac.get("keywords") or [name]
        cleaned = []
        for kw in keywords:
            kw = str(kw).strip()
            if not kw or kw in seen:
                continue
            seen.add(kw)
            cleaned.append(kw)
        if cleaned:
            result.append((name, cleaned))
    return result


def write_raw_keywords(raw_path: Path, agents_keywords: List[Tuple[str, List[str]]]) -> int:
    """
    根据 config.yaml 中的 agents[].keywords 重写 raw_keywords.txt。

    每行格式: <keyword> @<name>（name 中的空格以下划线代替）。
    返回写入的唤醒词总数。
    """
    lines = [
        "# 本文件由 tools/gen_keywords.py 从 config.yaml 的 agents[].keywords 自动生成",
        "# 格式: <phoneme_text> @<original>，@ 后缀为 agent 的 name（下划线代替空格）",
        "# 手动修改会被下次运行脚本覆盖；请直接编辑 config.yaml 后重跑脚本",
        "#",
    ]
    count = 0
    for name, keywords in agents_keywords:
        original = name.replace(" ", "_")
        for kw in keywords:
            lines.append(f"{kw} @{original}")
            count += 1
    raw_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return count


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
        # 从 config.yaml 读取第一个 agent 的 name/keywords 作为默认模板
        config_path = Path(__file__).resolve().parent.parent / "config.yaml"
        default_keyword = "赫尔墨斯"
        agents_keywords = load_agents_keywords(config_path)
        if agents_keywords:
            name, keywords = agents_keywords[0]
            default_keyword = keywords[0] if keywords else name
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

    # --- 自动同步：未显式指定 --input 时，从 config.yaml 同步 raw_keywords.txt ---
    if not args.input:
        config_path = Path(__file__).resolve().parent.parent / "config.yaml"
        agents_keywords = load_agents_keywords(config_path)
        if agents_keywords:
            n = write_raw_keywords(raw_path, agents_keywords)
            print(f"  📋 已从 {config_path} 同步 {n} 个唤醒词 → {raw_path}")
            print(f"     源配置: {config_path.name} 中 agents[].keywords")
        else:
            print(f"  ⚠️  config.yaml 无 agents 配置，使用已有 {raw_path}")

    # --- 生成 ---
    generate_keywords(raw_path, output_path)


if __name__ == "__main__":
    main()

