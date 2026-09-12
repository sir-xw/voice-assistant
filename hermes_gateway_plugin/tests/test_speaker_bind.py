#!/usr/bin/env python3
"""
voice_speaker_bind 工具单测（gateway plugin 侧，无 hermes 运行时）。

用桩客户端替代 WS 连接，覆盖：
- 会话归属：非语音会话 / 无语音记录 / 记录过旧 → 拒绝；
- 编号选择：默认取当前说话人；一轮多说话人时必须显式指定；显式编号必须属于最近轮次；
- 请求帧内容与 ack 处理：成功、冲突（需 overwrite）、超时/断连。

用法：
    /usr/local/lib/hermes-agent/venv/bin/python -u hermes_gateway_plugin/tests/test_speaker_bind.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hermes_gateway_plugin import adapter as A  # noqa: E402


class FakeClient:
    """记录请求帧并返回预设 ack。"""

    def __init__(self, ack=None):
        self.ack = ack
        self.calls: list[tuple] = []

    def request(self, frame_type, data, *, timeout=5.0):
        self.calls.append((frame_type, data, timeout))
        return self.ack


def make_adapter(ack=None):
    """绕过 hermes 基类，只装工具需要的状态。"""
    a = A.VoiceAdapter.__new__(A.VoiceAdapter)
    a._client = FakeClient(ack)
    a._current_wake = ""
    a._session_wake = {}
    a._last_utt_by_wake = {}
    return a


def dt(spk_ids, *, age=0.0, turn=1):
    """构造一轮语音记录（spk_ids 为空 = 没识别出编号）。"""
    now = time.time() - age
    speakers = [{"spk_id": s, "name": "未知", "label": f"未知 (ID: {s[4:]})",
                 "text": "x"} for s in spk_ids]
    return {"turn_seq": turn, "speakers": speakers, "ts": now,
            "recent": [(now, s) for s in spk_ids]}


def main() -> None:
    # ── 非语音会话（session_id 没有 wake 归属）→ 拒绝 ──
    a = make_adapter()
    out = json.loads(a.bind_speaker({"name": "辰辰"}, session_id="cli-1"))
    assert "error" in out and "不是语音会话" in out["error"], out

    # ── 有语音会话但记录过旧 → 拒绝 ──
    a = make_adapter()
    a._session_wake["s1"] = "小布"
    a._last_utt_by_wake["小布"] = dt(["spk_101"], age=A.SPEAKER_CURRENT_MAX_AGE_SEC + 5)
    out = json.loads(a.bind_speaker({"name": "辰辰"}, session_id="s1"))
    assert "error" in out and "较久" in out["error"], out

    # ── 一轮多个说话人且未指定编号 → 拒绝 ──
    a = make_adapter()
    a._session_wake["s1"] = "小布"
    a._last_utt_by_wake["小布"] = dt(["spk_101", "spk_102"])
    out = json.loads(a.bind_speaker({"name": "辰辰"}, session_id="s1"))
    assert "error" in out and "多个说话人" in out["error"], out

    # ── 显式编号不属于最近轮次 → 拒绝（防模型编错编号）──
    a = make_adapter()
    a._session_wake["s1"] = "小布"
    a._last_utt_by_wake["小布"] = dt(["spk_101"])
    out = json.loads(a.bind_speaker({"name": "辰辰", "spk_id": "100"}, session_id="s1"))
    assert "error" in out and "不属于" in out["error"], out

    # ── 正常路径：默认当前说话人，帧内容与返回 ──
    ack = {"seq": 1, "ok": True, "spk_id": "spk_101", "name": "辰辰",
           "previous": None, "also_bound": []}
    a = make_adapter(ack)
    a._session_wake["s1"] = "小布"
    a._last_utt_by_wake["小布"] = dt(["spk_101"])
    out = json.loads(a.bind_speaker({"name": "辰辰"}, session_id="s1"))
    assert out.get("ok") and out["spk_id"] == "spk_101", out
    assert out["label"] == "辰辰 (ID: 101)", out
    frame_type, data, timeout = a._client.calls[0]
    assert frame_type == "speaker_alias" and data["action"] == "set", data
    assert data["spk_id"] == "spk_101" and data["name"] == "辰辰"
    assert data["overwrite"] is False and timeout == A.SPEAKER_BIND_TIMEOUT_SEC

    # ── 显式编号 + overwrite 透传 ──
    ack = {"seq": 2, "ok": True, "spk_id": "spk_100", "name": "辰辰",
           "previous": "爸爸", "also_bound": ["spk_101"]}
    a = make_adapter(ack)
    a._session_wake["s1"] = "小布"
    a._last_utt_by_wake["小布"] = dt(["spk_100"])
    out = json.loads(a.bind_speaker(
        {"name": "辰辰", "spk_id": "spk_100", "overwrite": True}, session_id="s1"))
    assert out["ok"] and out["previous"] == "爸爸", out
    assert out["also_bound"] == ["spk_101"], out
    assert a._client.calls[0][1]["overwrite"] is True

    # ── 冲突 ack（服务端要求 overwrite）→ 以工具错误回给模型，附提示 ──
    ack = {"seq": 3, "ok": False, "spk_id": "spk_100",
           "error": "spk_100 当前绑定为「爸爸」；如已向用户确认要更正，请带 overwrite=true 重试"}
    a = make_adapter(ack)
    a._session_wake["s1"] = "小布"
    a._last_utt_by_wake["小布"] = dt(["spk_100"])
    out = json.loads(a.bind_speaker({"name": "辰辰"}, session_id="s1"))
    assert "error" in out and "overwrite=true" in out["error"], out

    # ── 超时/断连（request 返回 None）──
    a = make_adapter(None)
    a._session_wake["s1"] = "小布"
    a._last_utt_by_wake["小布"] = dt(["spk_101"])
    out = json.loads(a.bind_speaker({"name": "辰辰"}, session_id="s1"))
    assert "error" in out and "未响应" in out["error"], out

    # ── 无编号（声纹未启用）→ 拒绝 ──
    a = make_adapter()
    a._session_wake["s1"] = "小布"
    a._last_utt_by_wake["小布"] = dt([])
    out = json.loads(a.bind_speaker({"name": "辰辰"}, session_id="s1"))
    assert "error" in out and "没有识别到说话人编号" in out["error"], out

    # ── 校验：空名字 / 非法编号 ──
    a = make_adapter({"ok": True})
    a._session_wake["s1"] = "小布"
    a._last_utt_by_wake["小布"] = dt(["spk_101"])
    assert "error" in json.loads(a.bind_speaker({"name": " "}, session_id="s1"))
    out = json.loads(a.bind_speaker({"name": "辰辰", "spk_id": "辰辰"},
                                    session_id="s1"))
    assert "error" in out and "spk_id 非法" in out["error"], out

    # ── _remember_speakers：累计最近编号（供显式指定时校验）──
    a = make_adapter()
    a._remember_speakers("小布", [{"spk_id": "spk_101"}, {"spk_id": "spk_102"}], 3)
    a._remember_speakers("小布", [{"spk_id": "spk_103"}], 4)
    assert a._recent_speaker_ids("小布") == ["spk_101", "spk_102", "spk_103"]
    assert a._last_utt_by_wake["小布"]["turn_seq"] == 4
    assert [s["spk_id"] for s in a._last_utt_by_wake["小布"]["speakers"]] == ["spk_103"]

    # ── 工具已注册（假 ctx）──
    class Ctx:
        def __init__(self):
            self.tools = []
        def register_platform(self, **kw):
            pass
        def register_hook(self, *a, **k):
            pass
        def register_tool(self, **kw):
            self.tools.append(kw)

    ctx = Ctx()
    A.register(ctx)
    assert len(ctx.tools) == 1, ctx.tools
    tool = ctx.tools[0]
    assert tool["name"] == "voice_speaker_bind"
    assert tool["toolset"] == A.VOICE_TOOLSET == "voice_speaker"
    assert tool["check_fn"] is A.is_connected
    props = tool["schema"]["parameters"]["properties"]
    assert tool["schema"]["parameters"]["required"] == ["name"]
    assert {"name", "spk_id", "overwrite"} <= set(props)
    # 未连接时工具不可用
    assert A.is_connected() is False

    print("voice_speaker_bind 工具单测通过 ✅")


if __name__ == "__main__":
    main()
