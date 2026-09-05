"""
协调器核心：intent 状态机 + hold 计数 + effective 规则 + MPD 写路径。

模型（架构文档 §10.3）::

    intent       ∈ {playing, paused, stopped}   // 最新 agent/MCP 指令（last-wins）
    hold_count   ≥ 0                             // Voice Service TTS 避让计数
    effective    = (hold_count > 0) ? paused : intent

- intent API 操作（agent 经 MCP）：play/resume/next/previous → intent=playing；
  pause → paused；stop → stopped。**hold 激活期间只更新 intent、不立即打扰
  正在播的 TTS**；由下一次 sync（release 后）按新意图落盘。
- hold API 操作（Voice Service 经 Unix socket）：hold(reason) / release(reason)，
  计数式（幂等/可嵌套）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger("music_coordinator.coordinator")

# 可接受的 MCP 命令
CMD_PLAY = "play"
CMD_RESUME = "resume"
CMD_PAUSE = "pause"
CMD_STOP = "stop"
CMD_NEXT = "next"
CMD_PREVIOUS = "previous"

# 播放类命令 → 意图映射
_INTENT_BY_CMD = {
    CMD_PLAY: "playing",
    CMD_RESUME: "playing",
    CMD_PAUSE: "paused",
    CMD_STOP: "stopped",
    CMD_NEXT: "playing",
    CMD_PREVIOUS: "playing",
}


class Intent(str, Enum):
    PLAYING = "playing"
    PAUSED = "paused"
    STOPPED = "stopped"


class MpdBackend:
    """MPD 操作后端抽象（真实现见 mpd_conn.RealMpd；无 MPD 用 DummyMpd 记录）。"""

    def apply(self, intent: Intent) -> None:
        """把意图落盘到 MPD（play/pause/stop）。"""

    def status(self) -> dict:
        return {}

    def transport_op(self, name: str) -> None:
        """next/previous 等与播放/暂停无关的传输操作。"""


@dataclass
class MusicCoordinator:
    """intent/hold 状态机。线程模型：调用方保证串行（asyncio 单事件循环或加锁）。"""

    backend: Optional[MpdBackend] = None
    intent: Intent = Intent.STOPPED
    hold_count: int = 0
    # MCP 指令序号（last-wins 追踪，日志用）
    last_cmd_seq: int = 0
    # 最近一次实际同步结果日志
    _log: list = field(default_factory=list, repr=False)

    # ─── intent API（agent 经 MCP 调用）──────────────────

    def apply_command(self, cmd: str, *, seq: Optional[int] = None) -> dict:
        """处理 MCP 命令：更新意图并按需同步/执行传输操作。"""
        if cmd not in _INTENT_BY_CMD:
            return {"ok": False, "error": f"未知命令: {cmd}"}
        new_intent = Intent(_INTENT_BY_CMD[cmd])
        self.intent = new_intent
        self.last_cmd_seq = seq if seq is not None else self.last_cmd_seq + 1
        logger.info("[mc] 命令 %s → intent=%s (seq=%s, hold=%d)",
                    cmd, new_intent.value, self.last_cmd_seq, self.hold_count)

        # next/previous：曲目切换与播放/暂停无关 → 立即透传（不打断 TTS）
        if cmd in (CMD_NEXT, CMD_PREVIOUS) and self.backend is not None:
            try:
                self.backend.transport_op(cmd)
            except Exception as exc:
                logger.warning("[mc] 传输操作 %s 失败: %s", cmd, exc)

        # hold 激活期间：只记意图，播完（release）后 sync 生效 —— 避免
        # pause/stop 指令把正在播的 TTS 打断
        if self.hold_count <= 0:
            self._sync()
        return self.snapshot()

    # ─── hold API（Voice Service 经 Unix socket 调用）────

    def hold(self, reason: str = "tts") -> None:
        self.hold_count += 1
        logger.info("[mc] hold(%s) → hold_count=%d，effective=paused", reason, self.hold_count)
        if self.hold_count == 1:
            self._sync(force=True)  # 首次 hold 立即暂停

    def release(self, reason: str = "tts") -> None:
        if self.hold_count <= 0:
            logger.warning("[mc] release 多余（hold_count=0）: %s", reason)
            return
        self.hold_count -= 1
        logger.info("[mc] release(%s) → hold_count=%d", reason, self.hold_count)
        if self.hold_count == 0:
            self._sync()  # 回到意图状态（播放中恢复 / 暂停则保持暂停 / 停止保持停止）

    # ─── effective 与同步 ────────────────────────────────

    def effective(self) -> Intent:
        return Intent.PAUSED if self.hold_count > 0 else self.intent

    def _bootstrap_from_mpd(self) -> None:
        """启动引导：以 MPD 实际状态为初始意图（backend 不可用时保持 stopped）。"""
        if self.backend is None:
            return
        try:
            st = self.backend.status()
            state = st.get("state")  # "play" / "pause" / "stop"
            if state == "play":
                self.intent = Intent.PLAYING
            elif state == "pause":
                self.intent = Intent.PAUSED
            else:
                self.intent = Intent.STOPPED
            logger.info("[mc] 启动引导 intent=%s（MPD state=%s）", self.intent.value, state)
        except Exception as exc:
            logger.warning("[mc] 启动引导失败，保持 intent=%s: %s", self.intent.value, exc)

    def _sync(self, *, force: bool = False) -> None:
        if self.backend is None:
            return
        target = self.effective()
        # force=True（首次 hold）即便 target=paused 也应确保暂停已落盘
        if target == Intent.PLAYING:
            self.backend.apply(Intent.PLAYING)
        else:
            self.backend.apply(target)  # paused / stopped

    def snapshot(self) -> dict:
        return {
            "ok": True,
            "intent": self.intent.value,
            "hold_count": self.hold_count,
            "effective": self.effective().value,
            "seq": self.last_cmd_seq,
            "backend_status": self.backend.status() if self.backend else {},
        }


def _selftest() -> None:
    """状态机自检（Dummy 后端）。python -m music_coordinator.coordinator"""
    from .mpd_conn import DummyMpd

    mpd = DummyMpd()
    mc = MusicCoordinator(backend=mpd)

    # 1) 对话前在放歌 → 意图初始应来自 MPD 实际状态（启动引导）
    mc._bootstrap_from_mpd()
    assert mc.intent == Intent.PLAYING, mc.intent

    # 2) TTS 播报：hold → 暂停；release → 恢复（意图未变）
    mc.hold()
    assert mc.effective() == Intent.PAUSED
    assert mpd.ops[-1] == ("apply", "paused")
    mc.release()
    assert mc.effective() == Intent.PLAYING
    assert mpd.ops[-1] == ("apply", "playing")

    # 3) 用户"暂停音乐"（agent MCP pause）：意图=paused；之后任何 hold/release
    #    都不会把音乐放起来
    mc.apply_command(CMD_PAUSE)
    mc.hold()
    mc.release()
    assert mc.intent == Intent.PAUSED
    assert mc.effective() == Intent.PAUSED
    assert mpd.ops[-1] == ("apply", "paused")

    # 4) stop 同理；release 不会恢复
    mc.apply_command(CMD_STOP)
    mc.hold()
    mc.release()
    assert mc.effective() == Intent.STOPPED
    assert mpd.ops[-1] == ("apply", "stopped")

    # 5) hold 期间发 pause 指令 → 只记意图不打断；release 后保持暂停
    mc.apply_command(CMD_RESUME)  # intent=playing
    mc.hold()
    mc.apply_command(CMD_PAUSE)   # hold 中：仅 intent=paused
    assert mc.hold_count == 1
    assert mc.intent == Intent.PAUSED
    assert mpd.ops[-1] == ("apply", "paused")   # 仍来自 hold，未被 stop/clear 打断
    mc.release()                  # 释放后不恢复
    assert mc.effective() == Intent.PAUSED

    # 6) next/previous 透传（不依赖播放/暂停；随后无 hold → sync 回 playing）
    mc.apply_command(CMD_NEXT)
    assert ("transport", "next") in mpd.ops
    assert mc.intent == Intent.PLAYING
    assert mpd.ops[-1] == ("apply", "playing")

    print(f"[music_coordinator.coordinator] 自检通过 ✅（MPD 操作序列 {len(mpd.ops)} 条）")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _selftest()
