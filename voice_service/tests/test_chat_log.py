#!/usr/bin/env python3
"""
对话历史日志（ChatLog）单测 —— 不起服务、不碰音频。

覆盖：JSONL 字段与完整内容（含换行）、按天文件命名、无说话人记录、
保留期清理、关闭开关。

用法（在 voice_service/ 目录下运行）:
    python -u tests/test_chat_log.py
"""

import json
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voice_service.chat_log import ChatLog  # noqa: E402


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="chat-log-"))
    try:
        log = ChatLog(tmp, enabled=True, retention_days=3)
        log.user(wake="小布", text="打开客厅灯", spk_id="spk_100", name="爸爸")
        log.user(wake="小布", text="你好")                 # 无声纹/未识别
        log.assistant(wake="小布", kind="final", text="好嘞\n已经打开了")
        log.user(text="   ")                               # 空文本不写
        log.close()

        files = sorted(tmp.glob("chat-*.log"))
        assert len(files) == 1, files
        today = datetime.now().strftime("%Y-%m-%d")
        assert files[0].name == f"chat-{today}.log", files[0].name
        recs = _read(files[0])
        assert len(recs) == 3, recs

        u1, u2, a1 = recs
        assert u1["role"] == "user" and u1["wake"] == "小布"
        assert u1["speaker"] == {"spk_id": "spk_100", "name": "爸爸", "id": "100"}, u1
        assert u1["text"] == "打开客厅灯"
        assert u2["speaker"] is None, u2
        assert a1["role"] == "assistant" and a1["kind"] == "final"
        assert a1["text"] == "好嘞\n已经打开了", a1   # 完整内容（含换行）
        assert all("ts" in r and "T" in r["ts"] for r in recs), recs

        # ── 保留期清理：过期文件删，未来日期（异常）不删 ──
        old = tmp / "chat-2000-01-01.log"
        old.write_text("{}\n", encoding="utf-8")
        future = tmp / "chat-2099-12-31.log"
        future.write_text("{}\n", encoding="utf-8")
        other = tmp / "notes.txt"                          # 非本模块命名，不动
        other.write_text("keep\n", encoding="utf-8")
        log2 = ChatLog(tmp, enabled=True, retention_days=3)
        log2.user(text="hi")
        log2.close()
        assert not old.exists(), "过期历史应被清理"
        assert future.exists() and other.exists(), "不相关文件不得被删"

        # ── enabled=false 不写任何文件 ──
        tmp2 = Path(tempfile.mkdtemp(prefix="chat-log-off-"))
        try:
            off = ChatLog(tmp2, enabled=False)
            off.user(text="x")
            off.assistant(text="y")
            off.close()
            assert not list(tmp2.glob("chat-*.log")), "关闭时不应写文件"
        finally:
            shutil.rmtree(tmp2, ignore_errors=True)

        # ── 启动预建当天文件（让"写哪儿"可见）──
        tmp3 = Path(tempfile.mkdtemp(prefix="chat-log-ready-"))
        try:
            ready = ChatLog(tmp3, enabled=True)
            ready.ensure_ready()
            assert len(list(tmp3.glob("chat-*.log"))) == 1, "启动应预建当天文件"
            off_ready = ChatLog(tmp3, enabled=False)
            off_ready.ensure_ready()
            assert len(list(tmp3.glob("chat-*.log"))) == 1, "关闭时不应建文件"
        finally:
            shutil.rmtree(tmp3, ignore_errors=True)

        print("对话历史日志单测通过 ✅")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
