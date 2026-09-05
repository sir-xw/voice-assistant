# hermes_gateway_plugin — Hermes voice gateway 客户端（voice-platform 插件）

把 `voice-platform` 从「进程内嵌语音组件」改造为 **WS 客户端**：
- 上行：接收 Voice Service 的 `wake_word` / `asr_result`（语音识别结果）→ 构造
  `MessageEvent`（chat_id=`wake:<助手名>`）→ hermes gateway 会话路由 → agent；
- 下行：`send()` / `post_api_request` 钩子拿到的 LLM 回复文本 → `speak` 帧下发
  Voice Service 朗读（中间轮/最终轮/`[FINISH]` → `control{close_window}`）。

**唤醒词 ↔ 助手映射由 Voice Service 维护**（`voice_service/config.yaml` 的
`wake_word.assistants`；KWS 命中返回 `@助手名`，上行事件直接携带助手名）。
本插件**不再配置/上报 wakewords** —— `hello` 只报能力（caps），welcome 回报
服务端实际助手表（`my_wakewords`）供知情。hermes 配置只保留连接参数：

```yaml
platforms:
  voice:
    enabled: true
    extra:
      service:
        url: ws://127.0.0.1:8765
        token: ''
```

**本包是三个新包里唯一与 hermes 绑定的**（依赖 hermes venv 的
`gateway.*` / `hermes_cli.plugins`），必须安装进 hermes 所在 venv，并与
`voice_service`（协议契约）同 venv：

```bash
# 在 hermes venv 中
pip install -e ./voice_service
pip install -e ./hermes_gateway_plugin
```

## 部署（systemd user 服务，本机实测）

本插件是 hermes gateway 进程内加载的平台，**没有独立进程**：hermes-gateway.service
启动时经 entry point 注册 voice 平台，作为 WS 客户端连接 Voice Service。

1. **安装**（同 hermes venv）：

```bash
cd /root/git/voice-assistant
/usr/local/lib/hermes-agent/venv/bin/python -m pip install -e ./voice_service
/usr/local/lib/hermes-agent/venv/bin/python -m pip install -e ./hermes_gateway_plugin
```

2. **hermes 配置** `~/.hermes/config.yaml`（插件启用 + voice 平台连接参数）：

```yaml
plugins:
  enabled:
    - voice-platform          # 本插件 entry point

platforms:
  voice:
    enabled: true
    gateway_restart_notification: false
    extra:
      service:
        url: ws://127.0.0.1:8765
        token: ''             # 与 voice_service config.yaml 的 service.token 一致
```

   > 唤醒词↔助手映射在 Voice Service（无需在此配置）；如需按助手差异化
   > system_prompt/model，用 hermes 原生 `channel_overrides`（键 `wake:<助手名>`）。

3. **重启并验证**：

```bash
systemctl --user restart hermes-gateway
journalctl --user -u hermes-gateway -f | grep -E "voice|🎙"
# 预期：voice-platform 已注册 → [voice client] 已连接 …（服务端助手表: [...]）→ ✓ voice connected
```

> - gateway 正常运行时若重启 voice-service，voice 平台会自动重连，无需操作 gateway。
> - Voice Service 未启动时 voice 平台会持续退避重连（日志见 warning），起服务后自动恢复。

## 实现进度

- [x] `client.py`：可独立运行的 WS 客户端（hello/welcome、ping、线程安全 send）
- [x] `adapter.py`：接入 hermes gateway 平台（WS 客户端 + `post_api_request` 钩子）
- [x] entry point（voice-platform）；旧 `voice_agent.gateway_plugin` 已退役
