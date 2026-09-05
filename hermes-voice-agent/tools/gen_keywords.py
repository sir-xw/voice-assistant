#!/usr/bin/env python3
"""
sherpa-onnx 唤醒词生成工具。

【已退役声明】Voice Service 架构下（voice_assistant 三子项目）本脚本不再使用：
唤醒词↔助手映射的唯一源是 `voice_service/config.yaml` 的 `wake_word.assistants`，
由 `voice_service/voice_service/kws_words.py` 在 Voice Service 启动时自动生成/
校验 `models/sherpa-kws/<model>/keywords.txt`；hermes 侧 `platforms.voice.extra.
wakewords` 已不再配置。本文件仅作旧 hermes-voice-agent 架构的代码素材保留。

默认行为：从 profile 配置读取唤醒词（每个唤醒词一个独立会话），
同步生成 raw_keywords.txt，再转换为 sherpa-onnx 格式的 keywords.txt。

唤醒词来源（合并去重）：
    1. ~/.hermes/config.yaml 的 platforms.voice.extra.wakewords（dict: {名: ...}）
    2. ~/.hermes/voice-agent.yaml 的 wakewords（list 或 dict，可选）

keywords 文件写入约定模型目录（与 voice-agent.yaml 的 kws.model_name 对应）：
    <profile>/models/sherpa-kws/<model_name>/raw_keywords.txt
    <profile>/models/sherpa-kws/<model_name>/keywords.txt

raw_keywords.txt 格式（每行）:
    <phoneme_text> @<original_with_underscores>

    - <phoneme_text>: sherpa-onnx 格式的发音文本。中文会被自动转换为拼音；
                      英文保持不变（用户提供 ARPABET 或小写均可）。
    - @<original>:    原始关键词文本，即唤醒词名（下划线 _ 代替空格）。

示例 raw_keywords.txt:
    LIGHT UP @LIGHT_UP
    小布 @小布

转换后 keywords.txt:
    L AY1 T AH1 P @LIGHT_UP
    x iǎo  b ù @小布

用法:
    # 默认：自动读取 profile 配置的唤醒词，同步 raw_keywords.txt
    #       并生成 keywords.txt 到约定模型目录
    python -u tools/gen_keywords.py

    # 手动模式：指定 raw 文件（跳过自动同步）
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

from voice_agent.profile_config import (
    load_voice_config,
    profile_root,
    resolve_kws_model_dir,
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
# profile 配置读取（唤醒词）
# ---------------------------------------------------------------------------

def load_wakewords() -> List[Tuple[str, List[str]]]:
    """
    从 profile 配置读取唤醒词（合并去重）。

    配置结构：``wakewords: {名字: {keywords: [触发词, ...]}}``
    - **名字**：会话身份（chat_id = "wake:<名字>"），raw 行 @ 后缀
    - **keywords**：触发词（sherpa KWS 实际检测的短语，可多个；
      建议用较长短语如"小布小布"避免单字/双字误判）；
      未配置 keywords 时回退为 [名字]（不推荐）

    来源：
    1. ~/.hermes/config.yaml 的 platforms.voice.extra.wakewords
    2. ~/.hermes/voice-agent.yaml 的 wakewords（可选）

    返回 [(name, [keywords...]), ...]。配置缺失/读取失败时返回 []。
    """
    combined: dict = {}
    # 1) gateway 平台配置（唤醒词的权威来源）
    try:
        import yaml
        gw_path = profile_root() / "config.yaml"
        if gw_path.is_file():
            gw = yaml.safe_load(gw_path.read_text(encoding="utf-8")) or {}
            ww = (((gw.get("platforms") or {}).get("voice") or {})
                  .get("extra", {}).get("wakewords"))
            if isinstance(ww, dict):
                combined.update(ww)
            elif isinstance(ww, list):
                combined.update({str(w): {} for w in ww})
    except Exception as e:
        print(f"  ⚠️  读取 {profile_root() / 'config.yaml'} 失败: {e}")
    # 2) voice-agent.yaml 的 wakewords（可选补充）
    vcfg = load_voice_config()
    ww2 = vcfg.get("wakewords")
    if isinstance(ww2, dict):
        combined.update(ww2)
    elif isinstance(ww2, list):
        combined.update({str(w): {} for w in ww2})

    result: List[Tuple[str, List[str]]] = []
    seen: set = set()
    for name, conf in combined.items():
        name = str(name).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        if isinstance(conf, dict):
            kws = conf.get("keywords")
            keywords = [str(k).strip() for k in kws if str(k).strip()] \
                if isinstance(kws, list) else [name]
        else:
            keywords = [name]
        result.append((name, keywords))
    return result


def write_raw_keywords(raw_path: Path, agents_keywords: List[Tuple[str, List[str]]]) -> int:
    """
    根据 profile 配置中的唤醒词重写 raw_keywords.txt。

    每行格式: <keyword> @<name>（name 中的空格以下划线代替）。
    返回写入的唤醒词总数。
    """
    lines = [
        "# 本文件由 tools/gen_keywords.py 从 profile 配置（platforms.voice.extra.wakewords）自动生成",
        "# 格式: <phoneme_text> @<original>，@ 后缀为唤醒词名（下划线代替空格）",
        "# 手动修改会被下次运行脚本覆盖；请直接修改唤醒词配置后重跑脚本",
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
    default_model_dir = resolve_kws_model_dir(load_voice_config())
    p = argparse.ArgumentParser(
        description="sherpa-onnx 唤醒词生成工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model-dir", type=str, default=None,
                   help=f"模型目录（默认 {default_model_dir}）")
    p.add_argument("--input", "-i", type=str, default=None,
                   help="raw_keywords.txt 路径（默认 <model-dir>/raw_keywords.txt）")
    p.add_argument("--output", "-o", type=str, default=None,
                   help="输出 keywords.txt 路径（默认 <model-dir>/keywords.txt）")
    p.add_argument("--init", action="store_true",
                   help="创建 raw_keywords.txt 模板（使用 profile 配置中的第一个唤醒词）")

    args = p.parse_args()

    # --- 确定路径 ---
    if args.model_dir:
        model_path = Path(args.model_dir)
    else:
        model_path = default_model_dir

    if not model_path.exists():
        print(f"❌ 模型目录不存在: {model_path}")
        print("   请先下载模型（python -u tools/download_models.py）或指定 --model-dir")
        sys.exit(1)

    raw_path = Path(args.input) if args.input else model_path / "raw_keywords.txt"
    output_path = Path(args.output) if args.output else model_path / "keywords.txt"

    # --- --init: 创建模板 ---
    if args.init:
        if raw_path.exists():
            print(f"⚠️  文件已存在: {raw_path}")
            print("   使用 --input 指定其他路径，或删除已有文件")
            return
        # 从 profile 配置读取第一个唤醒词作为默认模板
        default_keyword = "赫尔墨斯"
        wakewords = load_wakewords()
        if wakewords:
            name, keywords = wakewords[0]
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

    # --- 自动同步：未显式指定 --input 时，从 profile 配置同步 raw_keywords.txt ---
    if not args.input:
        wakewords = load_wakewords()
        if wakewords:
            n = write_raw_keywords(raw_path, wakewords)
            print(f"  📋 已从 profile 配置同步 {n} 个触发词 → {raw_path}")
            print(f"     源配置: platforms.voice.extra.wakewords（名字 → keywords 触发词）")
        else:
            print(f"  ⚠️  profile 配置无唤醒词，使用已有 {raw_path}")

    # --- 生成 ---
    generate_keywords(raw_path, output_path)


if __name__ == "__main__":
    main()

