# Hermes Agent 语音输入前端

为 Hermes Agent 增加语音交互能力——**唤醒词唤醒 → 语音识别 → AI 对话 → 语音合成**全链路。

## 架构

```
麦克风 → PipeWire AEC → Sherpa-onnx(唤醒词) → WebRTC VAD → 腾讯云ASR
                                                              ↓
扬声器 ← 腾讯云TTS ← Hermes Agent ←───────────────────────────┘
```

- **唤醒词检测**：sherpa-onnx KWS（始终监听）
- **语音活动检测**：WebRTC VAD（对话窗口期内启动 ASR）
- **语音识别**：腾讯云实时 ASR（WebSocket）
- **对话引擎**：Hermes Agent（嵌入运行）
- **语音合成**：腾讯云 TTS
- **回声消除**：系统级 PipeWire（非软件 AEC）

## 多 Agent 机制

系统支持多个 AI Agent 实例，每个由不同的唤醒词触发，拥有独立会话历史和线程池。

### 工作方式

```
唤醒词 "赫尔墨斯"  ──→  Agent A（通用助手）
唤醒词 "翻译助手"  ──→  Agent B（翻译助手）
唤醒词 "code"      ──→  Agent C（编程助手）
```

每个 Agent 在 `config.yaml` 的 `agents` 列表中配置：

```yaml
agents:
  - name: "赫尔墨斯"
    model: "deepseek-v4-flash"
    session_id: "hermes-voice-session"
    system_prompt: |
      你是一个语音助手。
      ...

  - name: "翻译助手"
    model: "openai/gpt-4o-mini"
    session_id: "translator-session"
    system_prompt: |
      你是一个翻译助手。将用户的输入翻译成英文。
```

### 唤醒词绑定

唤醒词通过 `keywords.txt` 中的 `@original` 部分匹配：

```
# raw_keywords.txt
n ǐ h ǎo @你好小娜          → 匹配 agents[].name: "你好小娜"
f ān y ì @翻译助手           → 匹配 agents[].name: "翻译助手"
```

使用 `tools/gen_keywords.py` 管理唤醒词列表。

### 待汇报队列

当一个 Agent 正在处理时，用户说出另一唤醒词切换 Agent：

```
① 用户说 "赫尔墨斯"                           → Agent A 开始处理
② 用户说 "翻译助手"（Agent A 未完成）         → 活跃 name 切换为 "翻译助手"
③ Agent A 完成 → 不是当前活跃 → 进入待汇报队列
④ Agent B 完成 → 仍是当前活跃 → 立即 TTS 播报
⑤ Agent B 连续对话窗口过期（8s 无说话）       → 递送队列中 Agent A 的结果
⑥ "主人，我是赫尔墨斯，{回答}"
```

- 每个 Agent 有**独享的 ThreadPoolExecutor**，后台处理互不阻塞
- Agent 完成时若仍是当前活跃的，立即播报；否则进入**待汇报队列**
- 队列在**连续对话窗口过期**（frontend 回到 IDLE）时自动递送
- 递送格式：`"{user_title}，我是{keyword}，{回答文本}"`

## AI 语音输出（插件回调驱动，无 speak 工具）

大模型**不再需要任何播报工具**：hermes 的 `post_api_request` 插件回调会在每次 API 调用返回后
触发，VoiceApp 通过项目插件 **`speech-relay`** 把回复转发到播放队列自动播报。

### 机制

```
AI 收到用户问题
   ↓
[工具调用循环...]
   │
   ├─ 工具轮（finish_reason=tool_calls）
   │     → 若该轮带文字回复（如"正在搜索网络"），直接播报（阶段性，不进对话窗口）
   │     → 无文字则静默执行工具
   │
   └─ 最终轮（finish_reason=stop）
         → 以 (情绪)文字 格式的最终回答
         → VoiceApp 解析情绪 + 文本
         → TTS 流式播报 + 播放提示音 + 进入对话窗口
```

- 链路：hermes `post_api_request` 钩子 → `.hermes/plugins/speech-relay` 插件 →
  `voice_agent/speech_bridge.py`（进程内桥）→ `VoiceApp._on_llm_reply_hook` → 播放队列
- 插件加载：`voice_agent/plugin_hooks.py` 在 main.py 导入时开启
  `HERMES_ENABLE_PROJECT_PLUGINS=1` 并在进程内替换 `plugins.enabled` 白名单，
  不修改全局 `~/.hermes/config.yaml`

### 最终回答格式

AI 的最终回答请以 **`(情绪)文字内容`** 格式返回，例如：

| 示例 | 效果 |
|------|------|
| `(happy)你好，有什么可以帮助你的？` | 用高兴情绪播报 |
| `(sad)很抱歉听到这个消息` | 用悲伤情绪播报 |
| `(neutral)这是回答` | 用中性情绪播报 |
| `你好` | 未指定情绪，默认中性 |

如果不按格式（无括号包裹、无效情绪等），`emotion` 默认为空（中性），文字照常播报。

### 提示词约定

每个 Agent 的 `system_prompt` 尾部会自动追加以下说明，引导 AI 直接输出文字（会被自动播报）：

> 你的每条文字回复都会被系统实时语音播报给用户（无需调用任何播报工具）：
> - 工具调用过程中的阶段性说明（如"正在搜索网络"）请直接作为文字回复输出，系统会立即播报
> - 最终回答请直接以 `(情绪)文字` 格式返回
> - 播放控制工具（mpd_*）执行后直接返回 `[FINISH]`，系统自动关闭对话窗口

### 播放控制（mpd_*）

- 播放控制类工具执行后模型直接返回 `[FINISH]` 作为文字回复
- VoiceApp 在回调中识别 `[FINISH]`：不播报，直接关闭对话窗口

## 部署方案

本项目提供两种部署方式：

### 方案一：Hermes Gateway 插件（推荐）

将语音交互做成 hermes gateway 平台插件 `voice-platform`：

- 会话/记忆由 gateway 统一管理，每个唤醒词一个独立固定会话
- 可与飞书等消息平台共存于同一 gateway
- 配置统一在 profile 目录（`~/.hermes/`）：语音参数在 `voice-agent.yaml`，
  模型/凭据不再依赖项目目录
- 通过 pip entry-point 分发（`hermes_agent.plugins`），可开放给所有 hermes 用户

**详细安装与配置见 [`docs/gateway-voice-plugin-usage.md`](docs/gateway-voice-plugin-usage.md)。**

```bash
pip install -e .
hermes plugins enable voice-platform
python -u tools/download_models.py                 # 下载模型到 ~/.hermes/models/
python -u tools/gen_keywords.py                    # 生成唤醒词 keywords.txt
systemctl --user enable --now hermes-gateway       # 用户服务（可访问音频设备）
```

### 方案二：独立应用（VoiceApp，原方案）

`python -m voice_agent` 独立运行、自管 agent 会话。见下方「快速开始」。

## 快速开始

### 1. 安装 Hermes Agent

参考 [Hermes Agent 官方安装指南](https://github.com/NousResearch/hermes-agent)：

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

### 2. 安装 Python 依赖

在 Hermes Agent 的 uv 虚拟环境中安装：

```bash
source /usr/local/lib/hermes-agent/venv/bin/activate
sudo apt install libportaudio-ocaml-dev  # PortAudio 编译依赖
pip install -r requirements.txt
```

### 3. 配置凭据

项目根目录下创建 `.env`：

```env
# 腾讯云语音识别/合成（必填）
# 登录 https://console.cloud.tencent.com/cam/capi 创建子账号
VOICE_SecretId=AKIDxxxxxxxx
VOICE_SecretKey=xxxxxxxx
VOICE_AppId=125922xxxx
```

### 4. 下载唤醒词模型

```bash
mkdir -p models/sherpa-kws
cd models/sherpa-kws
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20.tar.bz2
tar -xf sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20.tar.bz2
```

### 5. 配置音频设备

```bash
# 查看可用音频设备
python -c "import sounddevice; print(sounddevice.query_devices())"
```

确认默认输入（麦克风）和输出（扬声器）设备正确。

### 6. 运行

```bash
python -u -m voice_agent
```

说出唤醒词（默认 `赫尔墨斯`）开始对话。

---

## Linux 部署手册

### 系统要求

- **Hermes Agent**（通过官方安装脚本部署）
- **Python ≥ 3.11**（Hermes Agent uv 环境自带）
- **PortAudio**（sounddevice 依赖）
- **PipeWire**（系统级回声消除）
- **音频设备**（需有可用的麦克风和扬声器）

### 步骤

#### 1. 部署 Hermes Agent

```bash
# 官方推荐方式安装
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash

# 后续 Python 操作在 Hermes Agent 的 uv 环境中进行
source /usr/local/lib/hermes-agent/venv/bin/activate
```

#### 2. 安装系统依赖

**Ubuntu/Debian：**

```bash
sudo apt update && sudo apt install -y \
    portaudio19-dev           \
    libsndfile1-dev           \
    build-essential cmake     \
    pipewire pipewire-pulse   \  # AEC
    libspa-0.2-modules        \  # PipeWire 回声消除模块
    alsa-utils pavucontrol
```

#### 3. 部署语音助手代码

```bash
git clone <your-repo-url> hermes-voice-agent
cd hermes-voice-agent
source /usr/local/lib/hermes-agent/venv/bin/activate
pip install -r requirements.txt
```

#### 4. 配置腾讯云凭据

```bash
# 注册腾讯云账号，创建仅有语音权限的子账号
# 获取 SecretId / SecretKey / AppId
# 填写到项目根目录的 .env 文件
```

参见上方「快速开始」第 3 步。

#### 5. 下载唤醒词模型

```bash
mkdir -p models/sherpa-kws
cd models/sherpa-kws
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20.tar.bz2
tar -xf sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20.tar.bz2
# 释放到：sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20/
```

#### 6. 配置 PipeWire 回声消除

```bash
# 安装 PipeWire 及其回声消除模块（步骤 2 中已安装）
# 确认 PipeWire 运行中
systemctl --user status pipewire
```


#### 7. 作为 systemd 服务运行

```ini
# /etc/systemd/system/hermes-voice.service
[Unit]
Description=Hermes Agent Voice Frontend
After=network-online.target sound.target

[Service]
Type=simple
User=your_user
WorkingDirectory=/path/to/hermes-voice-agent
Environment=PYTHONUNBUFFERED=1
ExecStart=/usr/local/lib/hermes-agent/venv/bin/python -u -m voice_agent
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now hermes-voice.service
sudo journalctl -u hermes-voice -f  # 查看日志
```

---

## 配置说明

全部配置在 `config.yaml` 中，关键项：

| 配置路径 | 默认值 | 说明 |
|----------|--------|------|
| `user_title` | `主人` | 语音助手对用户的称呼 |
| `wake_word.threshold` | `0.5` | KWS 检测阈值（越高越严格） |
| `wake_word.score` | `1.0` | KWS 关键词加分 |
| `wake_word.model.dir` | `models/sherpa-kws` | KWS 模型目录（相对项目根） |
| `wake_word.model.name` | `sherpa-onnx-kws-...` | KWS 模型子目录名 |
| `agents[].name` | `赫尔墨斯` | 触发该 Agent 的唤醒词（匹配 keywords.txt 中 @ 后的原文） |
| `agents[].model` | `deepseek-v4-flash` | Agent 使用的 LLM 模型 |
| `agents[].max_iterations` | `10` | Agent 最大推理迭代次数 |
| `agents[].session_id` | `hermes-{keyword}` | 会话 ID（保持对话连续性） |
| `agents[].system_prompt` | (内置) | Agent 系统提示词 |
| `vad.mode` | `3` | VAD 灵敏度（0-3，3 最敏感） |
| `vad.silence_threshold_ms` | `1000` | 静音超时（ms） |
| `vad.speech_confirm_frames` | `3` | 语音确认所需连续帧数 |
| `asr.engine_model` | `16k_zh` | 腾讯云 ASR 引擎模型 |
| `tts.voice_type` | `101001` | TTS 音色 ID |
| `conversation_window.timeout_sec` | `8.0` | 连续对话窗口（秒） |

## 测试脚本

| 脚本 | 用途 |
|------|------|
| `tests/test_sherpa_kws.py` | sherpa-onnx 唤醒词测试（麦克风） |
| `tests/test_sherpa_asr.py` | sherpa-onnx ASR 识别测试（文件/麦克风） |
| `tests/test_sherpa_tts.py` | sherpa-onnx TTS 合成测试 |
| `tests/test_tencent_asr.py` | 腾讯云 ASR 端到端测试（需凭据） |

## 工具脚本

| 脚本 | 用途 |
|------|------|
| `tools/gen_keywords.py` | 从 raw_keywords.txt 生成 keywords.txt（中文自动转拼音） |
| `tools/tts_gen.py` | 文本→语音（生成 assets 提示音 WAV） |
| `tools/measure_delay.py` | 扬声器→麦克风延迟探测（PipeWire 延迟调优） |

## 文件结构

```
├── voice_agent/               # 核心代码包（pip install -e .）
│   ├── main.py               # 入口：组装全链路、回调绑定
│   ├── voice_frontend.py     # 麦克风采集 + 唤醒词 + VAD + 状态机
│   ├── asr_engine.py         # 腾讯云 ASR 引擎（WebSocket）
│   ├── tts_engine.py         # 腾讯云 TTS 引擎
│   ├── audio_player.py       # PCM 音频播放器
│   └── config.py             # 配置加载（YAML + .env + 默认值）
├── tools/                    # 工具脚本
├── tests/                    # 测试脚本
├── assets/                   # 提示音 WAV（prompt / notification / farewell）
├── models/                   # 第三方模型（sherpa-onnx KWS / ASR / TTS）
├── config.yaml               # 应用配置
├── .env                      # 腾讯云凭据（不入库）
└── requirements.txt          # Python 依赖
``````

## 鸣谢

- [Hermes Agent](https://github.com/NousResearch/hermes-agent) — AI 代理框架
- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) — 语音处理框架
- [腾讯云语音识别/合成 SDK](https://github.com/TencentCloud/tencentcloud-speech-sdk-python)
