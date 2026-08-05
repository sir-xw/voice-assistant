"""
MPD (Music Player Daemon) 工具集 — 供 Hermes Agent 调用以控制音乐播放。

用法：
    import mpd_tool
    mpd_tool.register_all()

依赖：
    pip install python-mpd2

MPD 连接配置通过环境变量：
    MPD_HOST（默认 localhost）
    MPD_PORT（默认 6600）
"""

import json
import logging
import os
import socket

from tools.registry import registry
from music_control import set_expected_status

logger = logging.getLogger("mpd_tool")

# ─── MPD 连接 ────────────────────────────────────────────


def _connect():
    """建立 MPD 连接。"""
    host = os.environ.get("MPD_HOST", "localhost")
    port = int(os.environ.get("MPD_PORT", "6600"))
    try:
        import mpd as mpd_client
        client = mpd_client.MPDClient()
        client.timeout = 10
        client.idletimeout = None
        client.connect(host, port)
        return client
    except ImportError:
        raise RuntimeError("python-mpd2 not installed. Run: pip install python-mpd2")
    except socket.error as e:
        raise RuntimeError(f"Failed to connect to MPD at {host}:{port}: {e}")


def _format_song(song: dict) -> dict:
    """格式化歌曲信息，只保留关键字段。"""
    return {
        "file": song.get("file", ""),
        "artist": song.get("artist", "unknown"),
        "album": song.get("album", "unknown"),
        "title": song.get("title", song.get("file", "unknown")),
        "duration": song.get("duration", "0"),
    }


# ─── Handler 函数 ────────────────────────────────────────


def _escape_filter_value(value: str) -> str:
    """转义 filter 字符串中的值，使用双引号防止单引号问题。"""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def handle_search(args, **kw) -> str:
    """搜索 MPD 数据库中的歌曲（新版 filter 参数语法）。"""
    artist = args.get("artist", "")
    album = args.get("album", "")
    title = args.get("title", "")
    genre = args.get("genre", "")
    album_artist = args.get("album_artist", "")

    filters = []
    if artist:
        filters.append(f'(artist == {_escape_filter_value(artist)})')
    if album:
        filters.append(f'(album == {_escape_filter_value(album)})')
    if title:
        filters.append(f'(title == {_escape_filter_value(title)})')
    if genre:
        filters.append(f'(genre == {_escape_filter_value(genre)})')
    if album_artist:
        filters.append(f'(albumartist == {_escape_filter_value(album_artist)})')

    if not filters:
        return "请提供至少一个搜索条件（artist/album/title/genre/album_artist）"

    filter_str = " AND ".join(filters)

    try:
        client = _connect()
        try:
            results = client.search(filter_str)
            songs = [_format_song(s) for s in results[:50]]
            return json.dumps(songs, ensure_ascii=False, indent=2)
        finally:
            client.close()
    except Exception as e:
        return f"搜索失败: {e}"


def handle_play(args, **kw) -> str:
    """播放指定位置的歌曲。"""
    pos = args.get("pos")
    if pos is None:
        return "缺少 pos 参数（歌曲在播放列表中的位置）"
    try:
        client = _connect()
        try:
            client.play(int(pos))
            set_expected_status("Playing")
            return f"正在播放第 {int(pos)} 首"
        finally:
            client.close()
    except Exception as e:
        return f"播放失败: {e}"


def handle_pause(args, **kw) -> str:
    """暂停播放。"""
    try:
        client = _connect()
        try:
            client.pause(1)
            set_expected_status("Paused")
            return "已暂停播放"
        finally:
            client.close()
    except Exception as e:
        return f"暂停失败: {e}"


def handle_stop(args, **kw) -> str:
    """停止播放。"""
    try:
        client = _connect()
        try:
            client.stop()
            set_expected_status(None)
            return "已停止播放"
        finally:
            client.close()
    except Exception as e:
        return f"停止失败: {e}"


def handle_resume(args, **kw) -> str:
    """恢复播放。"""
    try:
        client = _connect()
        try:
            client.pause(0)
            set_expected_status("Playing")
            return "已恢复播放"
        finally:
            client.close()
    except Exception as e:
        return f"恢复播放失败: {e}"


def handle_next(args, **kw) -> str:
    """下一曲。"""
    try:
        client = _connect()
        try:
            client.next()
            set_expected_status("Playing")
            return "已切换到下一曲"
        finally:
            client.close()
    except Exception as e:
        return f"切换下一曲失败: {e}"


def handle_previous(args, **kw) -> str:
    """上一曲。"""
    try:
        client = _connect()
        try:
            client.previous()
            set_expected_status("Playing")
            return "已切换到上一曲"
        finally:
            client.close()
    except Exception as e:
        return f"切换上一曲失败: {e}"


def handle_get_status(args, **kw) -> str:
    """获取 MPD 服务器状态。"""
    try:
        client = _connect()
        try:
            status = client.status()
            return json.dumps(status, ensure_ascii=False, indent=2)
        finally:
            client.close()
    except Exception as e:
        return f"获取状态失败: {e}"


def handle_get_current_song(args, **kw) -> str:
    """获取当前播放歌曲信息。"""
    try:
        client = _connect()
        try:
            song = client.currentsong()
            if not song:
                return "当前没有播放歌曲"
            return json.dumps(_format_song(song), ensure_ascii=False, indent=2)
        finally:
            client.close()
    except Exception as e:
        return f"获取当前歌曲失败: {e}"


def handle_get_playlist(args, **kw) -> str:
    """获取当前播放列表。"""
    try:
        client = _connect()
        try:
            songs = client.playlistinfo()
            if not songs:
                return "播放列表为空"
            formatted = [_format_song(s) for s in songs]
            return json.dumps(formatted, ensure_ascii=False, indent=2)
        finally:
            client.close()
    except Exception as e:
        return f"获取播放列表失败: {e}"


def handle_clear_playlist(args, **kw) -> str:
    """清空播放列表。"""
    try:
        client = _connect()
        try:
            client.clear()
            return "已清空播放列表"
        finally:
            client.close()
    except Exception as e:
        return f"清空播放列表失败: {e}"


def handle_add_to_playlist(args, **kw) -> str:
    """添加歌曲到播放列表。"""
    uri = args.get("uri", "")
    if not uri:
        return "缺少 uri 参数（歌曲的 URI）"
    try:
        client = _connect()
        try:
            client.add(uri)
            return f"已添加歌曲: {uri}"
        finally:
            client.close()
    except Exception as e:
        return f"添加歌曲失败: {e}"


# ─── Schema 定义 ─────────────────────────────────────────

SCHEMAS = [
    {
        "name": "mpd_search",
        "description": "在 MPD 音乐库中搜索歌曲。可指定 artist/album/title/genre/album_artist",
        "parameters": {
            "type": "object",
            "properties": {
                "artist": {"type": "string", "description": "艺术家名称"},
                "album": {"type": "string", "description": "专辑名称"},
                "title": {"type": "string", "description": "歌曲标题"},
                "genre": {"type": "string", "description": "音乐风格"},
                "album_artist": {"type": "string", "description": "专辑艺术家"},
            },
        },
    },
    {
        "name": "mpd_play",
        "description": "播放播放列表中指定位置的歌曲。pos 从 0 开始",
        "parameters": {
            "type": "object",
            "properties": {
                "pos": {"type": "integer", "description": "歌曲在播放列表中的位置（从0开始）"},
            },
            "required": ["pos"],
        },
    },
    {
        "name": "mpd_pause",
        "description": "暂停当前播放的歌曲",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "mpd_stop",
        "description": "停止当前播放的歌曲",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "mpd_resume",
        "description": "恢复播放已暂停的歌曲",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "mpd_next",
        "description": "切换到下一曲",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "mpd_previous",
        "description": "切换到上一曲",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "mpd_get_status",
        "description": "获取 MPD 服务器的当前状态（播放/暂停/音量/播放进度等）",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "mpd_get_current_song",
        "description": "获取当前正在播放的歌曲信息",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "mpd_get_playlist",
        "description": "获取当前播放列表中的所有歌曲",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "mpd_clear_playlist",
        "description": "清空当前播放列表",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "mpd_add_to_playlist",
        "description": "添加一首歌曲到播放列表",
        "parameters": {
            "type": "object",
            "properties": {
                "uri": {"type": "string", "description": "歌曲的 URI 路径"},
            },
            "required": ["uri"],
        },
    },
]

# name → handler 映射
_HANDLERS = {
    "mpd_search": handle_search,
    "mpd_play": handle_play,
    "mpd_pause": handle_pause,
    "mpd_stop": handle_stop,
    "mpd_resume": handle_resume,
    "mpd_next": handle_next,
    "mpd_previous": handle_previous,
    "mpd_get_status": handle_get_status,
    "mpd_get_current_song": handle_get_current_song,
    "mpd_get_playlist": handle_get_playlist,
    "mpd_clear_playlist": handle_clear_playlist,
    "mpd_add_to_playlist": handle_add_to_playlist,
}


def register_all():
    """注册所有 MPD 工具到 Hermes Agent 的 tools registry。"""
    for schema in SCHEMAS:
        name = schema["name"]
        handler = _HANDLERS.get(name)
        if handler is None:
            logger.warning("No handler for tool: %s", name)
            continue
        registry.register(
            name=name,
            toolset="voice_agent",
            schema=schema,
            handler=handler,
        )
        logger.info("MPD tool registered: %s", name)