"""
对话历史日志（chat log）—— 供人工回顾并维护说话人身份。

Voice Service 把每轮会话落盘为 JSONL（一行一条，按天一个文件）：

    {"ts":"2026-09-24T21:03:11.512+08:00","role":"user","wake":"小布",
     "speaker":{"spk_id":"spk_101","name":"未知","id":"101"},
     "text":"查查我明天有什么课"}
    {"ts":"2026-09-24T21:03:14.008+08:00","role":"assistant","wake":"小布",
     "kind":"final","text":"你是辰辰还是淘淘？"}

字段：
- ``ts``       本地时间（含时区，ISO8601 毫秒）；
- ``role``     ``user``（用户语音）/ ``assistant``（助手播报）；
- ``wake``     助手名（用户说话的对象 / 回复者）；
- ``speaker``  仅 user：``{spk_id, name, id}``；未启用/未识别声纹时为 ``null``；
- ``kind``     仅 assistant：``final`` / ``interim`` / ``raw``（播报时的类别）；
- ``text``     **完整内容**（不截断），用户侧按句记录（一句一条，各带自己的说话人）。

用途：人工核对"哪个编号说了什么"，据此修正 ``config.yaml`` 的
``voiceprint.speaker_names`` 或 ``models/voiceprint_lib/names.json`` 里的身份绑定。

文件与保留：``<dir>/chat-YYYY-MM-DD.log``；``retention_days`` 天前的文件在启动/跨天时
自动删除（0 = 永久保留）。写入线程安全（音频/SDK/播报线程都会调用）。
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path

from . import protocol as P

logger = logging.getLogger("voice_service.chat_log")

FILE_PREFIX = "chat-"
FILE_SUFFIX = ".log"


class ChatLog:
    """按天滚动的 JSONL 对话历史写入器。"""

    def __init__(self, log_dir: str | Path, *, enabled: bool = True,
                 retention_days: int = 30):
        self.log_dir = Path(log_dir)
        self.enabled = bool(enabled)
        self.retention_days = int(retention_days or 0)
        self._lock = threading.Lock()
        self._day = ""              # 当前已打开文件对应的日期（YYYY-MM-DD）
        self._fh = None

    @classmethod
    def from_config(cls, cfg: dict | None, project_root: Path) -> "ChatLog":
        """从 config 的 ``chat_log`` 段构造；``dir`` 相对 voice_service 项目根解析。"""
        cfg = cfg or {}
        log_dir = Path(cfg.get("dir") or "logs")
        if not log_dir.is_absolute():
            log_dir = project_root / log_dir
        return cls(log_dir, enabled=bool(cfg.get("enabled", True)),
                   retention_days=int(cfg.get("retention_days", 30)))

    def ensure_ready(self) -> None:
        """启动时预建当天文件：让"日志开着吗/写哪儿"一眼可见（内容仍逐条追加）。"""
        if not self.enabled:
            return
        try:
            with self._lock:
                self._ensure_file()
        except Exception as exc:
            logger.warning("[chat_log] 初始化失败: %s", exc)

    # ─── 记录 ────────────────────────────────────────────

    def user(self, *, text: str, wake: str = "", spk_id: str = "",
             name: str = "") -> None:
        """记录一句用户语音（按句调用，各自带说话人）。"""
        text = (text or "").strip()
        if not text:
            return
        rec = {"ts": _now(), "role": "user", "wake": wake or ""}
        rec["speaker"] = ({"spk_id": spk_id, "name": name or "未知",
                           "id": P.speaker_number(spk_id)} if spk_id else None)
        rec["text"] = text
        self._write(rec)

    def assistant(self, *, text: str, wake: str = "", kind: str = "") -> None:
        """记录一次助手播报（完整文本）。"""
        text = (text or "").strip()
        if not text:
            return
        rec = {"ts": _now(), "role": "assistant", "wake": wake or ""}
        if kind:
            rec["kind"] = kind
        rec["text"] = text
        self._write(rec)

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                try:
                    self._fh.close()
                except Exception:
                    pass
                self._fh = None
            self._day = ""

    # ─── 内部 ────────────────────────────────────────────

    def _write(self, rec: dict) -> None:
        if not self.enabled:
            return
        try:
            line = json.dumps(rec, ensure_ascii=False)
            with self._lock:
                self._ensure_file()
                if self._fh is None:
                    return
                self._fh.write(line + "\n")
                self._fh.flush()
        except Exception as exc:      # 日志失败绝不影响语音主链路
            logger.warning("[chat_log] 写入失败: %s", exc)

    def _ensure_file(self) -> None:
        """按天滚动：日期变化时重开文件并清理过期历史。"""
        day = datetime.now().strftime("%Y-%m-%d")
        if self._fh is not None and day == self._day:
            return
        if self._fh is not None:
            try:
                self._fh.close()
            except Exception:
                pass
            self._fh = None
        self.log_dir.mkdir(parents=True, exist_ok=True)
        path = self.log_dir / f"{FILE_PREFIX}{day}{FILE_SUFFIX}"
        self._fh = path.open("a", encoding="utf-8")
        self._day = day
        logger.info("[chat_log] 对话历史写入: %s", path)
        self._prune()

    def _prune(self) -> None:
        """删除超过保留期的历史文件（只认自己的命名，不动目录里其它文件）。"""
        if self.retention_days <= 0:
            return
        cutoff = datetime.now().date() - timedelta(days=self.retention_days)
        try:
            for path in self.log_dir.glob(f"{FILE_PREFIX}*{FILE_SUFFIX}"):
                stamp = path.stem[len(FILE_PREFIX):]
                try:
                    day = datetime.strptime(stamp, "%Y-%m-%d").date()
                except ValueError:
                    continue
                if day < cutoff:
                    path.unlink()
                    logger.info("[chat_log] 清理过期对话历史: %s", path.name)
        except Exception as exc:
            logger.warning("[chat_log] 清理过期历史失败: %s", exc)


def _now() -> str:
    """本地时间 ISO8601（毫秒 + 时区偏移）。"""
    return datetime.now().astimezone().isoformat(timespec="milliseconds")
