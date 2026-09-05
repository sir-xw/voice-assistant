"""
KWS 词表生成（唤醒词 ↔ 助手映射的唯一源 = voice_service/config.yaml）。

背景：唤醒词与助手的对应关系由 Voice Service 维护，hermes voice gateway
不再配置 platforms.voice.extra.wakewords —— gateway 只消费上行事件里的
助手名（chat_id = "wake:<助手名>"）。因此本模块把 config 中的：

    wake_word:
      assistants:
        - name: 泡泡            # 助手名（= KWS keywords.txt 的 @后缀 = gateway chat_id 名）
          keywords: [泡泡管家, 你好泡泡]   # 实际触发短语（可多个，都归一为 name）
        - name: 小布
          keywords: [小布小布]

渲染成 sherpa-onnx 的 keywords.txt（写入模型目录）供 VoiceFrontend KWS 加载。
sherpa 命中某触发短语时 get_result() 返回其 @ 后缀 —— 即助手名，天然满足
「识别后把助手名称上行给 gateway」。

渲染规则与旧 tools/gen_keywords.py 一致：含中文用 pypinyin 拆声母+韵母，
每行 "<phoneme> @助手名"。

本模块零 hermes 依赖、不 import voice_frontend（避免拉起 numpy 等），
拼音拆分逻辑与 voice_frontend/gen_keywords 同源保持一致。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

logger = logging.getLogger("voice_service.kws_words")


# ─── 拼音拆分（与 voice_frontend._auto_format_keyword 同逻辑）──────────

def _split_pinyin_syllable(syllable: str) -> str:
    """带声调拼音音节 → "声母 韵母"（声调跟随韵母），sherpa keywords 格式。"""
    initials = [
        "zh", "ch", "sh",
        "b", "p", "m", "f", "d", "t", "n", "l",
        "g", "k", "h", "j", "q", "x",
        "r", "z", "c", "s", "y", "w",
    ]
    for init in sorted(initials, key=len, reverse=True):
        if syllable.startswith(init):
            final = syllable[len(init):]
            return f"{init} {final}" if final else init
    return syllable


def _auto_format_keyword(text: str) -> str:
    """中文短语 → sherpa 拼音文本（音节间双空格，声母韵母间单空格）。"""
    has_cjk = any('\u4e00' <= c <= '\u9fff' for c in text)
    if has_cjk:
        import pypinyin
        parts: List[str] = []
        for syl in pypinyin.pinyin([text]):
            for syl1 in syl:
                parts.append(_split_pinyin_syllable(syl1))
        return "  ".join(parts)
    return text.strip().lower()


# ─── assistants 解析 ────────────────────────────────────────

def parse_assistants(wake_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """解析 wake_word 配置段 → [{name, keywords:[...]}, ...]。

    兼容两种写法：
    - 新结构（推荐）：assistants: [{name, keywords:[...]}, ...]
    - 宽松兜底：keyword: "名"（单助手，keywords=[名]）
    返回空表表示未配置。
    """
    assistants = wake_cfg.get("assistants") or []
    if isinstance(assistants, dict):
        assistants = [{"name": k, "keywords": (v.get("keywords") if isinstance(v, dict)
                                               else [v])} for k, v in assistants.items()]
    out: List[Dict[str, Any]] = []
    for a in assistants if isinstance(assistants, list) else []:
        if not isinstance(a, dict):
            continue
        name = str(a.get("name") or "").strip()
        if not name:
            continue
        kws = a.get("keywords")
        if isinstance(kws, str):
            kws = [kws]
        kws = [str(k).strip() for k in (kws or []) if str(k).strip()]
        if not kws:
            kws = [name]
        out.append({"name": name, "keywords": kws})
    # 宽松兜底：无 assistants 但配置了 keyword（单助手）
    if not out:
        kw = str(wake_cfg.get("keyword") or "").strip()
        if kw:
            out.append({"name": kw, "keywords": [kw]})
    return out


# ─── 渲染 / 写入 ────────────────────────────────────────────

def render_keywords(assistants: List[Dict[str, Any]]) -> str:
    """assistants → keywords.txt 文本（每行 "<phoneme> @助手名"）。"""
    lines: List[str] = []
    for a in assistants:
        name = a["name"]
        for kw in a["keywords"]:
            lines.append(f"{_auto_format_keyword(kw)} @{name}")
    return "\n".join(lines) + ("\n" if lines else "")


def ensure_keywords_file(kws_dir: str, model_name: str,
                         assistants: List[Dict[str, Any]]) -> Tuple[Path, bool]:
    """确保模型目录 keywords.txt 与 assistants 一致；返回 (path, 是否写入)。

    - assistants 为空：不写（保留现状，交给 VoiceFrontend 的单词兜底）；
    - 模型目录不存在（未部署模型）：跳过并告警，不创建空目录；
    - 内容与现有文件一致：跳过（避免无谓 mtime/磁盘写）；
    - 不一致/缺失：按 config 重写（config 是唯一源，文件只是产物）。
    """
    model_path = Path(kws_dir) / model_name
    target = model_path / "keywords.txt"
    text = render_keywords(assistants)
    if not text:
        logger.info("[kws] assistants 为空，跳过 keywords.txt 生成（%s）", target)
        return target, False
    if not model_path.is_dir():
        logger.warning("[kws] 模型目录不存在，跳过 keywords.txt 生成: %s", model_path)
        return target, False
    try:
        existing = target.read_text(encoding="utf-8") if target.is_file() else None
    except OSError:
        existing = None
    if existing == text:
        return target, False
    target.write_text(text, encoding="utf-8")
    logger.info("[kws] keywords.txt 已按 config assistants 生成: %s (%d 行)",
                target, len(text.strip().splitlines()))
    return target, True
