"""
协议契约（两端共享）— M0。

Voice Service ↔ hermes voice gateway（hermes_gateway_plugin）之间的 WebSocket
帧定义与工具函数。**两端必须共用本模块**，防止契约漂移：
- 服务端：voice_service/voice_service/protocol.py（本文件）
- 客户端：依赖安装 voice_service 后 `from voice_service.protocol import ...`

帧格式：每个 WS 消息是一行 JSON（``\n`` 分隔），统一 envelope：:

    {"v": 1, "type": "<帧类型>", "seq": 123, "client_id": "...", "ts": 1710000000.123, "data": {}}

帧类型：
- C→S（命令，hermes voice gateway → Voice Service）：hello / speak / control /
  interrupt / speaker_alias / ping / bye
- S→C（事件/应答，Voice Service → hermes voice gateway）：welcome / ack /
  wake_word / asr_interim / asr_sentence / asr_result / speak_done / error / pong

说话人身份：`asr_result.data.speakers` 给出结构化说话人（`{spk_id, name, label, text}`），
`asr_result.data.text` 是给 agent 看的 `[名字 (ID: 编号)] 内容` 文本；agent 侧要绑定身份时
发 `speaker_alias`，服务端写 names.json 并热更新显示名（详见 PROTOCOL.md §5/§6）。

唤醒词 → 助手映射由 **Voice Service** 维护（config ``wake_word.assistants``，
KWS 命中返回 ``@助手名``）。hello 不再携带期望词表，客户端只用上行事件里
的助手名路由会话；welcome 回报服务端实际助手表（my_wakewords）供知情。

语音状态机变化 v1 不上行（只写服务日志）；未来做可视化时再加 state 事件。
本模块保持零第三方依赖（仅标准库 json/re/typing）。
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

# ─── 协议版本与类型常量 ───────────────────────────────────

PROTOCOL_VERSION = 1

# 命令：C→S
CMD_HELLO = "hello"              # 连接建立后首帧：认证 + 上报能力（caps）
CMD_SPEAK = "speak"              # 有文本要朗读（唯一播放入口）
CMD_CONTROL = "control"          # 流程控制（close_window 等）
CMD_INTERRUPT = "interrupt"      # gateway 主动取消播放（罕见）
CMD_SPEAKER_ALIAS = "speaker_alias"  # 说话人身份绑定（agent 工具写入，ack 回结果）
CMD_PING = "ping"
CMD_BYE = "bye"

# speaker_alias.action
ALIAS_SET = "set"                # 绑定/覆盖：{spk_id, name}
ALIAS_UNSET = "unset"            # 解除绑定：{spk_id}

# 事件/应答：S→C
EVT_WELCOME = "welcome"          # hello 应答：回报服务端实际助手表（my_wakewords）
EVT_ACK = "ack"                  # 命令应答（同 seq）
EVT_WAKE_WORD = "wake_word"      # KWS 命中（携带助手名；本地已播提示音/打断）
EVT_ASR_INTERIM = "asr_interim"  # ASR 中间结果（需 hello 声明 caps.interim）
EVT_ASR_SENTENCE = "asr_sentence"  # 每完成句（voiceprint 开时带说话人）
EVT_ASR_RESULT = "asr_result"    # VAD 判定整段结束、ASR final（inbound 主事件）
EVT_SPEAK_DONE = "speak_done"    # 一段/一 final 播报完成
EVT_ERROR = "error"
EVT_PONG = "pong"

# speak.kind：播报类别（2026-09 恢复区分语义，服务端据此决定播后行为）
# - "final"（最终回答，含 raw 兼容）：播完 → 通知音 → 进入连续对话窗口；
# - "interim"（中间轮文字，如工具轮）：播完若仍在等最终回复 → 恢复等待音，
#   不播通知音、不进对话窗口、不误触发 farewell。
# 客户端（hermes_gateway_plugin）：finish_reason=stop → final，其余 → interim。
SPEAK_FINAL = "final"
SPEAK_INTERIM = "interim"
SPEAK_RAW = "raw"

# control.action
CTRL_CLOSE_WINDOW = "close_window"   # 关对话窗口（[FINISH]）
CTRL_DISMISS_REPLY = "dismiss_reply" # M4 后可选增强：结束 waiting_reply（见架构文档 §9-A）
CTRL_RELOAD_KWS = "reload_kws"       # 可选增强：运行期热更 KWS 词表

# client caps
CAP_INTERIM = "interim"          # hello.caps: 是否需要 asr_interim 上行
CAP_VOICEPRINT = "voiceprint"    # hello.caps: 说话人识别是否可用（句子级标注）


# ─── 帧构建/编解码 ────────────────────────────────────────


def make_frame(
    type_: str,
    data: Optional[Dict[str, Any]] = None,
    *,
    seq: Optional[int] = None,
    client_id: str = "",
    ts: Optional[float] = None,
) -> Dict[str, Any]:
    """构造一帧 envelope。"""
    import time as _t
    return {
        "v": PROTOCOL_VERSION,
        "type": type_,
        "seq": seq,
        "client_id": client_id,
        "ts": _t.time() if ts is None else ts,
        "data": data or {},
    }


def dumps(frame: Dict[str, Any]) -> str:
    """帧 → 一行 JSON（可含中文，不转义）。"""
    return json.dumps(frame, ensure_ascii=False)


def loads(line: str) -> Dict[str, Any]:
    """一行 JSON → 帧；校验 v/type 基本结构，非法输入抛 ValueError。"""
    frame = json.loads(line)
    if not isinstance(frame, dict) or frame.get("v") != PROTOCOL_VERSION:
        raise ValueError(f"非法帧（版本不符或非对象）: {line[:80]!r}")
    if not isinstance(frame.get("type"), str):
        raise ValueError(f"非法帧（缺 type）: {line[:80]!r}")
    if "data" not in frame or not isinstance(frame.get("data"), dict):
        frame["data"] = {}
    return frame


def ack_for(seq: int, ok: bool, error: Optional[str] = None) -> Dict[str, Any]:
    """构造一条 ack 应答帧（纯 ok/error；带数据的 ack 由调用方自行 make_frame）。"""
    return make_frame(EVT_ACK, {"seq": seq, "ok": ok, "error": error}, seq=seq)


# 声纹库内部 id 形如 ``spk_<N>``（自动注册从 spk_100 起）
_SPK_ID_PATTERN = re.compile(r"spk_(\d+)\Z")


def normalize_spk_id(value: Any) -> str:
    """把 agent/工具给的说话人编号归一成声纹库内部 id。

    接受 ``"101"``（消息前缀里显示的编号）/ ``"spk_101"``（库内 id）/ 整数 101；
    其它形式返回 ``""``（调用方按非法参数处理）。
    """
    if isinstance(value, bool) or value is None:
        return ""
    if isinstance(value, int):
        return f"spk_{value}" if value >= 0 else ""
    text = str(value).strip()
    if not text:
        return ""
    if text.isdigit():
        return f"spk_{text}"
    if _SPK_ID_PATTERN.fullmatch(text):
        return text
    return ""


def speaker_number(spk_id: str) -> str:
    """内部 id（``spk_101``）→ 展示用编号（``101``）；非 spk_<N> 形式原样返回。"""
    m = _SPK_ID_PATTERN.fullmatch(str(spk_id or ""))
    return m.group(1) if m else str(spk_id or "")


# ─── (情绪)文字 分段解析 ──────────────────────────────────

# 合法情绪集合（与 TTS 引擎一致；不得单独改动 —— 与 speak schema 同步）
VALID_EMOTIONS = {
    "neutral", "sad", "happy", "angry", "fear",
    "story", "poetry", "sajiao", "disgusted", "amaze",
    "exciting", "aojiao", "jieshuo",
}
EMOTION_PATTERN = re.compile(
    r"\((?:%s)\)" % "|".join(sorted(VALID_EMOTIONS)), re.IGNORECASE)

# 分段类型：(emotion, text)；emotion 为空表示该段未指定情绪
EmotionSegment = Tuple[str, str]


def parse_emotion_segments(response: str) -> List[EmotionSegment]:
    """
    解析 AI 回复中的 (情绪)文字 格式，支持多个情绪标记分段。

    语义与旧 voice-platform adapter 一致：
    - 全角括号（（））自动转半角；
    - 无任何情绪标记 → [("", response)]；
    - 首标记之前存在无标记文本 → 归入 emotion="" 段；
    - 空文本段被丢弃。

    Args:
        response: AI 的文字回复。

    Returns:
        非空段列表；调用方（客户端）随 speak 帧下发 segments。
    """
    response = response.strip().replace('（', '(').replace('）', ')')
    if not response:
        return []
    matches = list(EMOTION_PATTERN.finditer(response))
    if not matches:
        return [("", response)]
    segments: List[EmotionSegment] = []
    if matches[0].start() > 0:
        head = response[:matches[0].start()].strip()
        if head:
            segments.append(("", head))
    for i, m in enumerate(matches):
        emotion = m.group(0)[1:-1].strip().lower()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(response)
        text = response[start:end].strip()
        if text:
            segments.append((emotion, text))
    return segments


def _selftest() -> None:
    """M0 自检：帧编解码 + 情绪分段（无第三方依赖，python -m voice_service.protocol）。"""
    # 1) 帧 roundtrip
    frame = make_frame(CMD_SPEAK, {
        "id": "r1", "wake": "小布", "kind": SPEAK_FINAL,
        "segments": [("happy", "你好")], "turn_seq": 3,
    }, seq=7, client_id="voice-gateway-1")
    line = dumps(frame)
    parsed = loads(line)
    assert parsed["v"] == 1
    assert parsed["type"] == CMD_SPEAK
    assert parsed["seq"] == 7
    assert parsed["data"]["kind"] == SPEAK_FINAL
    # JSON roundtrip 后 tuple 段变为嵌套 list（协议线上形态就是 list）
    assert parsed["data"]["segments"] == [["happy", "你好"]]

    # 2) 无情绪标记
    assert parse_emotion_segments("这是回答") == [("", "这是回答")]
    # 3) 单段情绪
    assert parse_emotion_segments("(happy)你好") == [("happy", "你好")]
    # 4) 多段 + 全角括号 + 段首无标记文本
    got = parse_emotion_segments("好嘞（neutral）再来一个…(happy)哈哈(angry)不")
    assert got == [("", "好嘞"), ("neutral", "再来一个…"), ("happy", "哈哈"), ("angry", "不")]
    # 5) 无效情绪不匹配（保留整段）
    assert parse_emotion_segments("(foo)内容") == [("", "(foo)内容")]
    # 6) 空文本段被丢弃
    assert parse_emotion_segments("(happy)") == []
    # 7) 空白输入
    assert parse_emotion_segments("   ") == []
    # 8) 非法帧
    try:
        loads('{"type":"x"}')
        raise AssertionError("版本缺失应报错")
    except ValueError:
        pass

    # 9) 说话人编号归一
    assert normalize_spk_id("101") == "spk_101"
    assert normalize_spk_id(" 101 ") == "spk_101"
    assert normalize_spk_id("spk_101") == "spk_101"
    assert normalize_spk_id(101) == "spk_101"
    assert normalize_spk_id("") == ""
    assert normalize_spk_id(None) == ""
    assert normalize_spk_id(True) == ""
    assert normalize_spk_id("爸爸") == ""
    assert normalize_spk_id("spk_x") == ""
    assert speaker_number("spk_101") == "101"
    assert speaker_number("爸爸") == "爸爸"

    # 10) 说话人绑定帧 + 带数据的 ack
    frame = make_frame(CMD_SPEAKER_ALIAS, {
        "action": ALIAS_SET, "spk_id": "spk_101", "name": "辰辰"}, seq=31)
    assert loads(dumps(frame))["data"]["action"] == ALIAS_SET
    ack = make_frame(EVT_ACK, {"seq": 31, "ok": True,
                               "spk_id": "spk_101", "name": "辰辰"}, seq=31)
    assert loads(dumps(ack))["data"]["name"] == "辰辰"
    print("[voice_service.protocol] 自检通过 ✅")


if __name__ == "__main__":
    _selftest()
