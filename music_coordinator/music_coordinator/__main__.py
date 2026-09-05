"""
music_coordinator 进程入口。

用法：
    python -m music_coordinator [--socket PATH] [--selfcheck] [--dummy-mpd]
                                [--enable-mcp [--mcp-host H] [--mcp-port P]]
    music-coordinator [...]                     （console script）

- 默认连接真实 MPD（MPD_HOST/MPD_PORT）；连不上时自动以 Dummy 模式运行；
- --dummy-mpd：强制 Dummy 后端（冒烟/开发，不触碰真实 MPD）；
- --enable-mcp：以 **streamable HTTP（web 端口）** 暴露 mpd_* MCP 工具面
  （需 pip install -e "./music_coordinator[mcp]"），供 hermes 等 MCP client 调用；
  MCP 服务跑在独立 daemon 线程（uvicorn 自带事件循环，不占主 asyncio）；
- --selfcheck：运行协调器状态机自检后退出（无需 MPD）。
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import threading
from typing import Optional, Tuple


def _default_socket_path() -> str:
    return f"/run/user/{os.getuid()}/music-coordinator.sock"


def _build_coordinator(log, dummy: bool = False) -> Tuple:
    """构造 (coordinator, backend)。

    - dummy=True：强制 DummyMpd（冒烟/开发，不触碰真实 MPD）；
    - 否则连真实 MPD（MPD_HOST/MPD_PORT），不可达时退回 Dummy。
    """
    from .coordinator import MusicCoordinator
    from .mpd_conn import DummyMpd, RealMpd

    if dummy:
        log.warning("强制 Dummy 模式运行（不触碰真实 MPD）")
        backend = DummyMpd(initial_state="play")
    else:
        backend = RealMpd()
        if not backend.status():
            log.warning("MPD 不可达，以 Dummy 模式运行（不执行真实操作）")
            backend = DummyMpd(initial_state="play")
    mc = MusicCoordinator(backend=backend)
    mc._bootstrap_from_mpd()
    return mc, backend


def _start_mcp_thread(mc, host: str, port: int) -> Optional[threading.Thread]:
    """独立 daemon 线程运行 MCP web（uvicorn 自带事件循环）。"""
    try:
        from .tools_mcp import run_web
    except ImportError:
        logging.getLogger("music_coordinator").error(
            "缺少 mcp/uvicorn：请先 pip install -e \"./music_coordinator[mcp]\"")
        return None
    t = threading.Thread(target=run_web, args=(mc, host, port),
                         daemon=True, name="mcp-web")
    t.start()
    logging.getLogger("music_coordinator").info(
        "MCP(streamable HTTP) 线程已启动：http://%s:%d/mcp", host, port)
    return t


async def _run_ipc(args: argparse.Namespace, mc) -> int:
    """hold IPC 服务主循环（asyncio）。"""
    from .hold_ipc import HoldIpcServer

    server = HoldIpcServer(mc, args.socket)
    await server.start()
    logging.getLogger("music_coordinator").info(
        "hold IPC 就绪（socket=%s, intent=%s）", args.socket, mc.intent.value)
    try:
        await asyncio.Event().wait()
    finally:
        await server.stop()
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="music-coordinator",
                                     description="音乐状态协调器")
    parser.add_argument("--socket", default=_default_socket_path(),
                        help="hold IPC Unix socket 路径")
    parser.add_argument("--enable-mcp", action="store_true",
                        help="以 streamable HTTP（web 端口）暴露 mpd_* MCP 工具")
    parser.add_argument("--mcp-host", default="127.0.0.1",
                        help="MCP HTTP 监听地址（默认 127.0.0.1）")
    parser.add_argument("--mcp-port", type=int, default=8766,
                        help="MCP HTTP 监听端口（默认 8766）")
    parser.add_argument("--dummy-mpd", action="store_true",
                        help="强制 Dummy 后端（冒烟/开发用，不触碰真实 MPD）")
    parser.add_argument("--selfcheck", action="store_true",
                        help="运行状态机自检后退出（无需 MPD）")
    parser.add_argument("--verbose", action="store_true", help="DEBUG 日志")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.selfcheck:
        from .coordinator import _selftest
        _selftest()
        return 0

    log = logging.getLogger("music_coordinator")
    mc, backend = _build_coordinator(log, dummy=args.dummy_mpd)

    if args.enable_mcp:
        if _start_mcp_thread(mc, args.mcp_host, args.mcp_port) is None:
            return 1
    else:
        log.info("MCP 未启用（--enable-mcp 可开启 web 端口工具面）")

    try:
        return asyncio.run(_run_ipc(args, mc))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
