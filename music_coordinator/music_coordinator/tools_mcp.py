"""
MCP 工具面：以 **web（streamable HTTP）端口**暴露 mpd_* 工具 → intent API。

接入形态（用户已定）：music_coordinator 监听一个 HTTP 端口（默认
127.0.0.1:8766/mcp，streamable HTTP transport），hermes 等 MCP client 经
HTTP 连接调用工具。播放控制类工具把「用户想要的音乐状态」写进 intent
（coordinator.apply_command），只读查询直接读 MPD（backend）。

- 安装：pip install -e "./music_coordinator[mcp]"（mcp + uvicorn）
- 启动：python -m music_coordinator --enable-mcp [--mcp-port 8766]
- 冒烟：python tests/mcp_smoke.py --url http://127.0.0.1:8766/mcp

播放控制工具的 schema 名称与旧 mpd_tool 保持一致（mpd_play/mpd_pause/...），
便于 hermes agent 侧无缝替换；handler 语义从「直连 MPD」改为「更新 intent →
按 effective 同步（见 coordinator.py）」，语音播报期间的避让由 Voice Service
的 hold/release 处理，agent 无需感知。
"""

from __future__ import annotations

from typing import Any, Dict

from .coordinator import (
    CMD_NEXT,
    CMD_PAUSE,
    CMD_PLAY,
    CMD_PREVIOUS,
    CMD_RESUME,
    CMD_STOP,
    MusicCoordinator,
)

# 播放控制类工具名 → coordinator 命令
_INTENT_TOOL_CMDS = {
    "mpd_play": CMD_PLAY,
    "mpd_resume": CMD_RESUME,
    "mpd_pause": CMD_PAUSE,
    "mpd_stop": CMD_STOP,
    "mpd_next": CMD_NEXT,
    "mpd_previous": CMD_PREVIOUS,
}


def build_mcp_server(coordinator: MusicCoordinator, *, host: str = "127.0.0.1",
                     port: int = 8766):
    """构建 FastMCP server（惰性 import mcp，未安装时抛 ImportError）。

    FastMCP v1 的 host/port 在构造参数里指定；run(transport="streamable-http")
    时按此监听（默认端点 /mcp）。
    """
    from mcp.server.fastmcp import FastMCP

    backend = coordinator.backend

    mcp = FastMCP(
        "music-coordinator",
        host=host,
        port=port,
        instructions=(
            "音乐播放控制协调器。播放/暂停/停止/切歌工具会改变音乐的「目标状态」"
            "（intent，最新指令胜出）；当语音助手在朗读（TTS 避让）时，指令只记录意图、"
            "朗读结束后按新意图生效。回答用户前可用 mpd_get_status 查看当前实际状态。"
        ),
    )

    # ─── 播放控制类（写 intent）──────────────────────────

    for tool_name, cmd in _INTENT_TOOL_CMDS.items():
        # 为每个工具生成独立的闭包 docstring（FastMCP 用函数 docstring 作为工具描述）
        def make_handler(_cmd: str, _desc: str):
            def handler() -> Dict[str, Any]:
                return coordinator.apply_command(_cmd)
            handler.__name__ = _cmd
            handler.__doc__ = _desc
            return handler

        descriptions = {
            CMD_PLAY: "开始/恢复播放（intent=playing）",
            CMD_RESUME: "恢复播放（同 mpd_play）",
            CMD_PAUSE: "暂停播放（intent=paused；朗读避让结束后保持暂停）",
            CMD_STOP: "停止播放（intent=stopped）",
            CMD_NEXT: "切到下一曲（透传 MPD，intent=playing）",
            CMD_PREVIOUS: "切到上一曲（透传 MPD，intent=playing）",
        }
        mcp.tool(name=tool_name)(make_handler(cmd, descriptions[cmd]))

    # ─── 只读查询类（不更新 intent）──────────────────────

    @mcp.tool()
    def mpd_get_status() -> Dict[str, Any]:
        """获取协调器与 MPD 的当前状态（intent/hold/effective + MPD state）。"""
        return coordinator.snapshot()

    @mcp.tool()
    def mpd_get_current_song() -> Dict[str, Any]:
        """获取当前播放歌曲信息。"""
        return dict(backend.currentsong())

    @mcp.tool()
    def mpd_get_playlist() -> list:
        """获取当前播放列表（前 50 首关键字段）。"""
        return backend.playlist()

    @mcp.tool()
    def mpd_search(artist: str = "", album: str = "", title: str = "",
                   genre: str = "", album_artist: str = "") -> Dict[str, Any]:
        """在 MPD 音乐库中搜索歌曲（至少提供一项条件）。"""
        filters = {k: v for k, v in {
            "artist": artist, "album": album, "title": title,
            "genre": genre, "albumartist": album_artist}.items() if v}
        if not filters:
            return {"ok": False, "error": "请提供至少一个搜索条件"}
        return {"ok": True, "songs": backend.search(filters)}

    # ─── 播放列表编辑类（写 MPD，不改 intent）────────────

    @mcp.tool()
    def mpd_clear_playlist() -> Dict[str, Any]:
        """清空当前播放列表（不影响播放/暂停意图）。"""
        return coordinator.clear_playlist()

    @mcp.tool()
    def mpd_add_to_playlist(uri: str) -> Dict[str, Any]:
        """把音乐库条目追加到播放列表；uri 用 mpd_search 返回的 file 字段。"""
        return coordinator.add_to_playlist(uri)

    return mcp


def run_web(coordinator: MusicCoordinator, host: str = "127.0.0.1",
            port: int = 8766) -> None:
    """以 streamable HTTP 方式运行 MCP server（阻塞，直至进程退出）。

    供 __main__ 放入后台线程调用：用 FastMCP 的 Starlette app 直接起 uvicorn。
    uvicorn>=0.50 的信号处理已移出 Server（子线程可直接 run）；老版本
    （<0.50）子线程运行需 capture_signals=False，如遇 add_signal_handler 报错
    请升级 uvicorn 或在 Config 传入 capture_signals=False。
    """
    import uvicorn

    mcp = build_mcp_server(coordinator, host=host, port=port)
    app = mcp.streamable_http_app()
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    uvicorn.Server(config).run()
