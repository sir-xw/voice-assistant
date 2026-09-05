"""
MCP(streamable HTTP) 冒烟：连接 music_coordinator 的 web 端口，调用 mpd_* 工具。

用法（先起服务）：
    pip install -e "./music_coordinator[mcp]"
    python -m music_coordinator --enable-mcp --mcp-port 18766 --socket /tmp/mc.sock
    python tests/mcp_smoke.py --url http://127.0.0.1:18766/mcp

无 MPD 时协调器以 Dummy 后端运行，本冒烟仍可通过（只验证工具面/意图状态机）。
"""

from __future__ import annotations

import argparse
import asyncio


async def main(url: str) -> int:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async with streamablehttp_client(url) as (read, write, _session_ref):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            names = [t.name for t in tools.tools]
            print(">>> 可用工具:", names)
            for want in ("mpd_play", "mpd_pause", "mpd_stop", "mpd_next",
                         "mpd_previous", "mpd_get_status"):
                assert want in names, f"缺少工具 {want}"

            print(">>> mpd_pause:", (await session.call_tool("mpd_pause", {})))
            print(">>> mpd_get_status:", (await session.call_tool("mpd_get_status", {})))
            print(">>> mpd_next:", (await session.call_tool("mpd_next", {})))
            print(">>> mpd_resume:", (await session.call_tool("mpd_resume", {})))
            print(">>> mpd_get_playlist:", (await session.call_tool("mpd_get_playlist", {})))
            print(">>> MCP 冒烟通过 ✅")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8766/mcp")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(args.url)))
