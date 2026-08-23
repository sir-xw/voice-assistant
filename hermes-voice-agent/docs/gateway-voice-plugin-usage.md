# voice-platform 插件：安装与使用

`voice-platform` 是 hermes gateway 的语音平台插件：sherpa 唤醒词 + WebRTC VAD +
腾讯云 ASR/TTS，把语音交互接入 hermes gateway（每个唤醒词一个独立会话）。

## 安装

```bash
# 1. 安装本包（pip 安装即注册 hermes 插件，entry point: hermes_agent.plugins）
pip install -e /path/to/hermes-voice-agent

# 2. 启用插件（写入 ~/.hermes/config.yaml 的 plugins.enabled）
hermes plugins enable voice-platform

# 3. 配置腾讯云凭据（写入 profile 的 ~/.hermes/.env）
#    VOICE_SecretId / VOICE_SecretKey / VOICE_AppId
```

## 配置

插件配置**全部归 profile 目录**（hermes home，默认 `~/.hermes/`），不读取插件安装位置的任何文件：

- **语音参数**（kws/vad/mic/asr/tts/voiceprint）：`~/.hermes/voice-agent.yaml`（约定文件，结构见下）
- **平台开关/唤醒词/每会话提示词**：`~/.hermes/config.yaml` 的 `platforms.voice`
- **腾讯云凭据**：`~/.hermes/.env`（适配器自动加载，不 fallback 源码目录）
- 优先级：`platforms.voice.extra` > `voice-agent.yaml` > 内置默认

### 1) `~/.hermes/voice-agent.yaml`（语音参数，全部可省略 = 内置默认）

本地模型统一存放在 `<profile>/models/`，配置只指定模型**名称**，不指定路径：

```yaml
# sherpa KWS 唤醒词（目录约定 <profile>/models/sherpa-kws/<model_name>/）
kws:
  model_name: "sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20"

# 本地模型清单（tools/download_models.py 按此检查/下载）
models:
  asr: "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20"  # 可选
  # speaker: "sherpa-onnx-pyannote-segmentation-3-0"   # voiceprint 启用时
  # campplus: "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx"

# VAD（wake_guard_sec 唤醒后静音保护期，太短会导致 ASR 提前结束）
vad:
  mode: 3
  silence_threshold_ms: 1000
  wake_guard_sec: 5.0

mic:
  sample_rate: 0        # 0=自动探测
  device: auto

asr:
  engine_model: "16k_zh_en_speaker_2.0"   # 腾讯云在线 ASR

tts:
  voice_type: 601009

voiceprint:
  enabled: false
```

### 模型下载

```bash
# 检查并下载缺失模型（KWS 必需；models.asr 可选；voiceprint 启用时下载声纹模型）
python -u tools/download_models.py
python -u tools/download_models.py --check   # 仅检查
```

### 2) `~/.hermes/config.yaml`（平台开关与每会话配置）

```yaml
plugins:
  enabled: [voice-platform]

platforms:
  voice:
    enabled: true
    gateway_restart_notification: false   # voice 只播 LLM 回复，抑制系统广播
    extra:
      # 唤醒词 → 会话映射（每个唤醒词 = 独立 gateway 会话，chat_id = "wake:<名>"）
      wakewords:
        小布: {}
        翻译助手: {}
      # 连续对话窗口期（最终回复播完后 VAD 直接听，无需再喊唤醒词）
      conversation_window_sec: 5.0

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
# 凭据在 profile 目录 ~/.hermes/.env（插件自动加载），无需项目 .env
EnvironmentFile=/root/.hermes/.env
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
