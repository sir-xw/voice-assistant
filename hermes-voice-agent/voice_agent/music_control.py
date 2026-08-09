"""
工具函数库 — 播放器控制等通用工具。

使用 playerctl 控制媒体播放器的暂停与恢复。
"""

import subprocess
import logging

logger = logging.getLogger("utils")

# 全局变量：保存暂停前播放器状态
_previous_status: str | None = None


def set_expected_status(status: str | None):
    """
    设置预期的播放状态。供 mpd_tool 在 AI 调用 mpd 播放控制后调用，
    更新对话结束后 player_resume() 的行为。

    Args:
        status: "Playing" / "Paused" / "Stopped" / None（清除）
    """
    global _previous_status
    old = _previous_status
    _previous_status = status
    logger.info("Expected status updated: %s → %s", old, status)


def get_expected_status() -> str | None:
    """获取当前预期的播放状态。"""
    return _previous_status


def player_pause(force: bool = False) -> str | None:
    """
    暂停当前播放器。

    获取当前播放器的状态（Playing / Paused / Stopped），
    保存到全局变量 _previous_status 中，然后调用 playerctl pause。
    如果已经是暂停状态且 force=False，则不执行操作。
    如果 force=True，则无论当前状态如何都强制暂停并保存原状态。

    Args:
        force: 是否强制暂停。在 TTS 播放前应设为 True，
               以确保播放期间音乐被暂停（即使 agent 执行期间用户恢复了播放）。

    Returns:
        保存的状态字符串，或 None（无需暂停时 / playerctl 不可用时）。
    """
    global _previous_status

    try:
        # 获取当前播放器状态
        result = subprocess.run(
            ["playerctl", "status"],
            capture_output=True, text=True, timeout=3
        )
        status = result.stdout.strip()
        if result.returncode != 0 or not status:
            logger.debug("playerctl status returned empty or error: %s", result.stderr.strip())
            return None

        # 如果已经是暂停/停止状态且非强制模式，则跳过
        if not force and (status == "Paused" or status == "Stopped"):
            logger.info("Player already %s, skip pause", status)
            return None

        # 保存当前状态（即使当前是 Paused/Stopped 也保存，确保 resume 能正确判断）
        _previous_status = status
        logger.info("Saving player status: %s (force=%s)", status, force)

        if status == "Paused" or status == "Stopped":
            # 在 force 模式下，状态已保存但无需执行 pause
            logger.info("Player already %s, status saved", status)
            return status

        # 执行暂停
        subprocess.run(
            ["playerctl", "pause"],
            capture_output=True, text=True, timeout=3,
            check=True
        )
        logger.info("Player paused")
        return status

    except subprocess.TimeoutExpired:
        logger.warning("playerctl pause timed out")
        return None
    except subprocess.CalledProcessError as e:
        logger.warning("playerctl pause failed: %s", e.stderr.strip())
        return None
    except FileNotFoundError:
        logger.warning("playerctl not found, is it installed?")
        return None


def player_resume() -> None:
    """
    恢复播放。

    检查全局变量 _previous_status，如果之前的状态是 "Playing"，
    则调用 playerctl play 恢复播放，并重置全局变量。

    否则不执行任何操作。可安全重复调用（只会执行一次）。
    """
    global _previous_status

    if _previous_status is None:
        logger.debug("No previous status saved, skip resume")
        return

    status = _previous_status
    _previous_status = None  # 重置状态，确保只恢复一次

    if status == "Playing":
        try:
            subprocess.run(
                ["playerctl", "play"],
                capture_output=True, text=True, timeout=3,
                check=True
            )
            logger.info("Player resumed")
        except subprocess.TimeoutExpired:
            logger.warning("playerctl play timed out")
        except subprocess.CalledProcessError as e:
            logger.warning("playerctl play failed: %s", e.stderr.strip())
        except FileNotFoundError:
            logger.warning("playerctl not found, is it installed?")
    else:
        logger.debug("Previous status was %s, no resume needed", status)
