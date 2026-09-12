#!/usr/bin/env python3
"""
asr_result 上行契约测试（不需要麦克风/声纹模型/腾讯云凭据）。

用桩件替换 ASR 与声纹管理器，验证 `inbound.py` 的拼装：
- `text` 按句带 `[名字 (ID: 编号)]` 前缀、多句换行；
- `speakers` 结构化字段与文本行一一对应（含未识别到编号的情形）；
- `turn_seq` 自增、`wake` 带上；
- 同一轮结果写入 chat log（逐句、带 spk_id/name，完整内容不截断）。

用法（在 voice_service/ 目录下运行）:
    python -u tests/test_inbound_speakers.py
"""

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json  # noqa: E402

import numpy as np  # noqa: E402

from voice_service.chat_log import ChatLog  # noqa: E402
from voice_service.inbound import Inbound  # noqa: E402
from voice_service.voiceprint import UNKNOWN_SPEAKER  # noqa: E402


class FakeAsr:
    last_text = ""

    def __init__(self):
        self.sentences: list[dict] = []

    def get_sentence_audio(self, info):
        return np.zeros(16000 * 2, dtype=np.float32)   # 2s，足够自动注册


class FakeVoiceprint:
    """identify 永远认不出；register 依次发号（模拟自动注册）。"""

    def __init__(self):
        self._next = 100

    def identify(self, samples):
        return None, 0.31

    def register(self, samples):
        self._next += 1
        return f"spk_{self._next}"

    def speaker_label(self, spk_id):
        if not spk_id:
            return UNKNOWN_SPEAKER
        return f"{UNKNOWN_SPEAKER} (ID: {spk_id[4:]})"

    def display_name(self, spk_id):
        return UNKNOWN_SPEAKER


def make_inbound(chat_log=None):
    ib = Inbound.__new__(Inbound)
    ib.cfg = None
    ib.server = None
    ib.playback = None
    ib.chat_log = chat_log
    ib.asr = FakeAsr()
    ib.voiceprint = FakeVoiceprint()
    ib._current_wake = "小布"
    ib._turn_seq = 0
    ib._vp_round = []
    ib._vp_id_cache = {}
    ib._vp_min_register_sec = 1.5
    ib._vp_use_cache = True
    ib.fired = []
    ib._fire = lambda t, d: ib.fired.append((t, d))
    return ib


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="inbound-"))
    try:
        chat = ChatLog(tmp, enabled=True)
        ib = make_inbound(chat)

        # 两句不同说话人（第二句腾讯云换了 speaker_id）
        ib.asr.last_text = "查查我明天有什么课"
        ib._on_asr_sentence({"text": "查查我明天有什么课", "speaker_id": 0})
        ib._on_asr_sentence({"text": "顺便看看天气", "speaker_id": 1})
        ib._on_asr_complete()

        assert len(ib.fired) == 1, ib.fired
        type_, data = ib.fired[0]
        assert type_ == "asr_result", type_
        assert data["text"] == ("[未知 (ID: 101)] 查查我明天有什么课\n"
                                "[未知 (ID: 102)] 顺便看看天气"), data
        assert data["wake"] == "小布" and data["turn_seq"] == 1, data
        assert data["speakers"] == [
            {"spk_id": "spk_101", "name": UNKNOWN_SPEAKER,
             "label": "未知 (ID: 101)", "text": "查查我明天有什么课"},
            {"spk_id": "spk_102", "name": UNKNOWN_SPEAKER,
             "label": "未知 (ID: 102)", "text": "顺便看看天气"},
        ], data["speakers"]
        assert ib._vp_round == [], "上行后应清空本轮缓冲"

        # 第二轮：turn_seq 自增；无 voiceprint 结果时回落到整段文本
        ib._current_wake = "泡泡"
        ib.voiceprint = None
        ib.asr.last_text = "你好呀"
        ib._on_asr_complete()
        _, data2 = ib.fired[1]
        assert data2["turn_seq"] == 2 and data2["speakers"] == [], data2
        assert data2["text"] == "你好呀" and data2["wake"] == "泡泡", data2

        # 空结果不上行
        ib.asr.last_text = "   "
        ib._on_asr_complete()
        assert len(ib.fired) == 2

        chat.close()
        recs = [json.loads(l) for l in
                sorted(tmp.glob("chat-*.log"))[0].read_text(
                    encoding="utf-8").splitlines() if l.strip()]
        assert len(recs) == 3, recs     # 两句 + 无声纹的一整条
        assert recs[0]["speaker"]["spk_id"] == "spk_101"
        assert recs[1]["speaker"]["id"] == "102" and recs[1]["text"] == "顺便看看天气"
        assert recs[2]["speaker"] is None and recs[2]["text"] == "你好呀"
        assert all(r["wake"] for r in recs)

        print("asr_result 上行契约测试通过 ✅")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
