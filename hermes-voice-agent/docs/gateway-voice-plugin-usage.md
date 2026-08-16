# voice-platform 插件：安装与使用

`voice-platform` 是 hermes gateway 的语音平台插件：sherpa 唤醒词 + WebRTC VAD +
腾讯云 ASR/TTS，把语音交互接入 hermes gateway（每个唤醒词一个独立会话）。

## 安装

```bash
# 1. 安装本包（pip 安装即注册 hermes 插件，entry point: hermes_agent.plugins）
pip install -e /path/to/hermes-voice-agent

# 2. 启用插件（写入 ~/.hermes/config.yaml 的 plugins.enabled）
hermes plugins enable voice-platform

# 3. 配置腾讯云凭据（写入 ~/.hermes/.env 或环境变量）
#    VOICE_SecretId / VOICE_SecretKey / VOICE_AppId
```

## 配置

`~/.hermes/config.yaml`：

```yaml
plugins:
  enabled: [voice-platform]

platforms:
  voice:
    enabled: true
    extra:
      # 唤醒词 → 会话映射（每个唤醒词 = 独立 gateway 会话，chat_id = "wake:<名>"）
      wakewords:
        小布: { session_id: "hermes-voice-小布" }
        翻译助手: { session_id: "hermes-voice-翻译助手" }
      # 连续对话窗口期（最终回复播完后 VAD 直接听，无需再喊唤醒词）
      conversation_window_sec: 5.0
      # 播报身份前缀：多唤醒词共用扬声器时加 "我是{唤醒词}，"
      identity_prefix: true
      # sherpa KWS 模型（默认项目 models/ 下，可改）
      kws:
        model_dir: "models/sherpa-kws"
        model_name: "sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20"
      # VAD（可选，默认与 VoiceApp 一致）
      vad:
        mode: 3
        silence_threshold_ms: 600
      # TTS 音色（可选，默认 101001）
      tts:
        voice_type: 101001
      # 说话人识别（可选，复用 voice_agent/voiceprint.py）
      voiceprint:
        enabled: false

    # 每个唤醒词独立的 system_prompt / model / provider（gateway 原生机制）
    channel_overrides:
      "wake:小布":
        system_prompt: "你是家庭助手小布，负责日常问答。"
      "wake:翻译助手":
        system_prompt: "你是翻译助手，把用户输入翻译成英文。"
        model: "openai/gpt-4o-mini"

# 工具条件披露：只对语音平台开放本项目工具（mpd 音乐控制等）
platform_toolsets:
  voice: [voice_agent]
```

## 启动

```bash
hermes gateway
```

`hermes gateway status` 应显示 `voice` 平台已连接（🎙️ 语音）。

## systemd 部署（替代原 hermes-voice-agent 服务）

原 `services/hermes-voice-agent.service`（`python -m voice_agent`）停用，
改为启动 hermes gateway（可新建 `services/hermes-gateway.service`）：

```ini
[Unit]
Description=Hermes Agent Gateway (voice platform)
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/git/voice-assistant/hermes-voice-agent
Environment=HERMES_ENABLE_PROJECT_PLUGINS=1
EnvironmentFile=/root/git/voice-assistant/hermes-voice-agent/.env
ExecStart=/usr/local/lib/hermes-agent/venv/bin/hermes gateway
Restart=always

[Install]
WantedBy=multi-user.target
```

> 插件以 entry-point 分发时无需 `HERMES_ENABLE_PROJECT_PLUGINS`；
> 若从项目 `.hermes/plugins/` 目录加载则需要该环境变量。
> `hermes-voice-agent-volume.service`（USB 音量）与语音平台无关，可保留。

## 行为说明

- **inbound**：唤醒词命中 → VAD 确认人声 → 腾讯云 ASR → 进入对应会话
- **outbound**：回复经 `(情绪)文字` 分段解析（情绪列表与 VoiceApp 一致），
  流式 TTS 播报 → 通知音 → 进入对话窗口期（窗口期内无需唤醒词）
- **中间轮播报**：工具轮文字回复由 `post_api_request` 钩子直接播报
  （最终轮由 gateway `send()` 播放，天然去重）
- **播放控制**：模型使用 `mpd_*` 工具后返回 `[FINISH]` → 关闭对话窗口
- **打断**：再次唤醒/说话 → 物理打断 TTS + 清播报队列；
  agent 层打断由 gateway `busy_input_mode: interrupt`（默认）处理
- **多唤醒词**：不同唤醒词 = 不同 chat_id = 独立会话与历史，互不干扰；
  身份前缀可选（`identity_prefix`）区分谁在说话

## 真机冒烟清单

1. `hermes gateway` 启动，`gateway status` 显示 voice 已连接
2. 喊唤醒词 → 听到提示音 → 提问 → agent 回答（TTS 播报 + 通知音）
3. 播报结束 5s 内直接说话（对话窗口期）→ 无需唤醒词即可追问
4. 播放音乐时喊唤醒词 → 音乐暂停、回答后恢复
5. 说话中途再次唤醒 → 立即打断当前播报
6. 多唤醒词分别提问 → 各自会话历史独立
