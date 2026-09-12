# music_coordinator — 音乐状态协调器

MPD 的**唯一写入口**（意图状态机 + hold 计数 + 实际 MPD 操作），解决
「谁暂停了音乐、会话结束后该播放还是暂停」的归因问题（架构文档 §10）：

- **intent（意图）**：用户 → agent → MCP 指令（play/pause/stop/next/…）表达的
  期望状态 `{playing, paused, stopped}`，最新指令胜出（last-wins）；
- **hold（避让）**：Voice Service 每段 TTS 播报前 `hold()` / 播完 `release()`，
  只让路、不改意图；
- **effective = (hold_count > 0) ? paused : intent** —— 唯一落盘 MPD 的目标态。

零 hermes 依赖，可安装到任意 venv：

```bash
python -m venv .venv-music && . .venv-music/bin/activate
pip install -e ./music_coordinator
music-coordinator --socket /run/user/0/music-coordinator.sock   # 或 python -m music_coordinator
```

- 连接 MPD：环境变量 `MPD_HOST`（默认 localhost）、`MPD_PORT`（默认 6600）；
- hold IPC：Unix socket，JSON lines（`{"op":"hold"|"release"|"status"}`）；
- MCP 工具面（`tools_mcp.py`）为可选：`pip install -e "./music_coordinator[mcp]"`。

## 部署（systemd user 服务，本机实测）

依赖前置：MPD 运行中（默认 localhost:6600）；MCP 工具面需已安装 `[mcp]` extras。

1. **安装**（同 hermes venv；需 MCP 依赖）：

```bash
cd /root/git/voice-assistant
/usr/local/lib/hermes-agent/venv/bin/python -m pip install -e "./music_coordinator[mcp]"
```

2. **unit 文件** `~/.config/systemd/user/music-coordinator.service`：

```ini
[Unit]
Description=Music Coordinator（MPD 唯一写入口：intent/hold IPC + MCP web）
After=network-online.target

[Service]
Type=simple
ExecStart=/usr/local/lib/hermes-agent/venv/bin/python -u -m music_coordinator --enable-mcp
Environment=XDG_RUNTIME_DIR=/run/user/0
Environment=HOME=/root
# MPD 非本机默认端口时：Environment="MPD_HOST=..." / "MPD_PORT=..."
Restart=always
RestartSec=3

[Install]
WantedBy=default.target
```

3. **启用与验证**：

```bash
systemctl --user daemon-reload
systemctl --user enable --now music-coordinator
systemctl --user status music-coordinator     # active (running)
ss -tlnp | grep 8766                          # MCP web 监听 http://127.0.0.1:8766/mcp
ls -l /run/user/0/music-coordinator.sock      # hold IPC socket（0o666，voice_service 可连）
journalctl --user -u music-coordinator -f
```

4. **hermes 侧接入 MCP**（`~/.hermes/config.yaml`）：

```yaml
mcp_servers:
  music:
    url: http://127.0.0.1:8766/mcp
    enabled: true
```

   Agent 即以 `mcp__music__mpd_*` 工具控制音乐；Voice Service 的播报避让经 hold IPC
   socket 直连（无需 hermes）。两服务互相独立，重启任一不影响另一侧连接。

## 实现进度

- [x] `coordinator.py`：intent/hold/effective 状态机（含自检）
- [x] `hold_ipc.py`：Unix socket hold/release/status 服务
- [x] `mpd_conn.py`：python-mpd2 封装 + 只读查询（currentsong/playlist/search）+ 播放列表编辑
      （clear/add；Dummy 模式同接口）
- [x] `tools_mcp.py`：MCP server 工具面 —— **web（streamable HTTP）端口**暴露 12 个 `mpd_*` 工具
      （`--enable-mcp`，需 `[mcp]` extras）：
      `mpd_play` / `mpd_resume` / `mpd_pause` / `mpd_stop` / `mpd_next` / `mpd_previous`（intent）、
      `mpd_get_status` / `mpd_get_current_song` / `mpd_get_playlist` / `mpd_search`（只读）、
      `mpd_clear_playlist` / `mpd_add_to_playlist`（播放列表编辑，不改 intent）
- [x] hermes 侧 MCP 接入联调：`~/.hermes/config.yaml` `mcp_servers.music`
      （`url: http://127.0.0.1:8766/mcp`）→ Agent 以 `mcp__music__mpd_*` 调用音乐控制
- [x] systemd 部署：`music-coordinator.service`（`python -u -m music_coordinator --enable-mcp`）运行中，
      经 hold IPC 服务 Voice Service 播报避让、经 MCP web 服务 hermes Agent 意图
