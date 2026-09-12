# voice_service — Voice Service（语音服务，WS 服务端）

独立语音服务进程：拥有麦克风/扬声器、sherpa 唤醒词、WebRTC VAD、腾讯云 ASR/TTS、
播放队列与提示音。对外暴露 **WebSocket 接入点**（一行一个 JSON 帧），供
`hermes_gateway_plugin`（WS 客户端）接收语音识别结果、下发待朗读文本。

**本包零 hermes 依赖**，可安装到任意独立 venv：

```bash
python -m venv .venv-voice-service && . .venv-voice-service/bin/activate
pip install -e ./voice_service
python -m voice_service --config config.yaml
```

组件代码（`vad.py`/`voice_frontend.py`/`asr_engine.py`/`tts_engine.py`/`audio_player.py`/
`voiceprint.py`/`tencentcloud_speech/`）复制自旧 `hermes-voice-agent/voice_agent/`，
运行时不 import 旧包；vendored 腾讯云 SDK 随本包分发。

## 接入其他 agent

**接口文档：[`PROTOCOL.md`](PROTOCOL.md)** —— 面向其他 agent / 前端的 WebSocket
协议说明（握手、帧类型、播报语义、心跳与关闭码、最小接入示例），无需依赖 hermes。

协议契约代码见 `voice_service/protocol.py`（零第三方依赖，可直接复制到客户端；
`hermes_gateway_plugin` 即复用同一模块）。详细架构设计见
`hermes-voice-agent/docs/voice-service-websocket-architecture.md`。

## 部署（systemd user 服务，本机实测）

依赖前置（均已就绪）：Hermes venv（`/usr/local/lib/hermes-agent/venv`）、KWS 模型与
`models/voiceprint_lib`（在 `voice_service/models/`）、腾讯云凭据（`voice_service/.env`，
`VOICE_SecretId/Key/AppId`）、本仓库 `config.yaml`。

1. **安装**（与 gateway 同 venv，便于复用协议模块）：

```bash
cd /root/git/voice-assistant
/usr/local/lib/hermes-agent/venv/bin/python -m pip install -e ./voice_service
```

2. **配置文件**（`voice_service/config.yaml`）：语音硬件/云 SDK 参数、`wake_word.assistants`
   （唤醒词↔助手映射唯一源，修改后重启生效并自动重生成 KWS `keywords.txt`）、
   `music_coordinator.socket`（播报避让用，见 music_coordinator README）。

3. **unit 文件** `~/.config/systemd/user/voice-service.service`：

```ini
[Unit]
Description=Voice Service（语音服务：唤醒/ASR/TTS/播放；WS 接入点）
After=pipewire.service pipewire-pulse.service pulseaudio.service music-coordinator.service
Wants=pipewire-pulse.service

[Service]
Type=simple
ExecStart=/usr/local/lib/hermes-agent/venv/bin/python -u -m voice_service --audio --out --config config.yaml
WorkingDirectory=/root/git/voice-assistant/voice_service
Environment=XDG_RUNTIME_DIR=/run/user/0
Environment=HOME=/root
EnvironmentFile=-/root/git/voice-assistant/voice_service/.env
Restart=always
RestartSec=3

[Install]
WantedBy=default.target
```

4. **启用与验证**：

```bash
systemctl --user daemon-reload
systemctl --user enable --now voice-service
systemctl --user status voice-service        # active (running)
curl http://127.0.0.1:8765/healthz           # ok（WS 探活）
journalctl --user -u voice-service -f        # 日志：KWS initialized / 客户端接入
```

> - `--audio`（打开麦克风监听）/`--out`（扬声器播报）为真实硬件开关；不加则只起 WS 骨架。
> - 服务重启不影响 hermes-gateway：voice 平台是 WS 客户端，本服务恢复后自动重连。

## 实现进度

- [x] M0 `protocol.py`：帧 envelope / 帧类型 / `parse_emotion_segments`（含自检）
- [x] 组件复制与 import 改写（`voice_agent.*` → `voice_service.*`）
- [x] 服务端骨架：监听、`/healthz`、`hello` 握手、`ping/pong`，dispatch 接入 playback
- [x] M2 inbound：`inbound.py`（VoiceFrontend+ASR+voiceprint → `wake_word`/`asr_result` 上行）
- [x] M3 outbound：`playback.py`（speak 串行队列 → TTS/资产/等待音、final→通知音+连续对话窗口、
      interim→等待音恢复、按会话 turn 丢迟到旧回复、无回复超时兜底）；`music_client.py`
      （Music Coordinator hold/release 音乐避让）；`--audio`/`--out` 开关（默认不出声/不占设备）
- [x] `kws_words.py`：唤醒词↔助手映射唯一源 `wake_word.assistants`（config.yaml）→ 启动时
      生成/校验 KWS `keywords.txt`；`hello` 不再收客户端词表，`welcome.my_wakewords` 回填实际助手表
- [x] 采集失效检测：麦克风无数据 → 关流 + 节流告警（恢复靠重启 voice-service 进程）
- [x] 真机联调部署：systemd user 服务 `voice-service.service` 以 `--audio --out --config config.yaml`
      运行，配合 hermes-gateway（WS 客户端）与 music-coordinator 全链路可用
- [ ] M1 server 拆分为 hub/session 多连接管理（v1 保持单 client，为既定取舍；多路扩展按需再做）
