# Hermes Agent 语音输入前端方案

> 基于 sounddevice + sherpa-onnx + webrtcvad + 腾讯云 ASR/TTS SDK

---

## 1. 背景与目标

为 [Hermes Agent](https://hermesagent.org.cn/) 增加完整语音交互能力——**语音输入**和**语音输出**双通道。

- **语音输入链路**：唤醒词 → VAD → 实时语音识别 → Hermes Agent → 文本回答
- **语音输出链路**：Hermes Agent 回答文本 → 腾讯云流式语音合成 → 音频实时播放

**技术路线：**

| 层 | 选型 | 说明 |
|----|------|------|
| 音频采集 | `sounddevice` | 跨平台，回调模式 |
| 唤醒词 | `sherpa-onnx` | 离线运行，跨平台推理框架 |
| VAD | `webrtcvad` | 纯 DSP <1ms |
| AEC | `pipewire` / WebRTC AEC | 声学回声消除 |
| ASR | 腾讯云 `SpeechRecognizer` (SDK) | `tencentcloud-speech-sdk-python` 官方 SDK |
| 对话引擎 | Hermes Agent `AIAgent` (嵌入) | 同进程调用，保持 session |
| TTS | 腾讯云 `FlowingSpeechSynthesizer` (SDK) | 流式合成，PCM 直出 |

**不使用 `AudioToTextRecorder` 的原因：**
该组件封装了完整的 麦克风→VAD→转录 管线，但不支持将转录后端替换为外部云服务。因此直接编排其底层组件，将转录环节替换为腾讯云 SDK。

---

## 2. 系统架构

```
                    ┌───────────────────────────────────────────────┐
                    │              Voice Frontend                    │
                    │                                                │
                    │  ┌──────┐  ┌──────┐  ┌──────────┐  ┌───────┐ │
── IN ───────────▶  │  │唤醒词 │  │ VAD  │  │腾讯云ASR │  │Hermes  │ │
  麦克风 ────▶ AEC ─▶  │sherpa │─▶│WebRTC│─▶│SDK       │─▶│Agent  │─┼──▶ TTS
                    │  │onnx   │  │mode=3│  │16k_zh    │  │AIAgent│ │   触发
                    │  └──────┘  └──────┘  └──────────┘  └───────┘ │
                    │                                                │
                    │  ┌────────────────────────────────────────┐   │
                    │  │       腾讯云 TTS SDK                    │   │
  扬声器 ◀──────────│◀─│  FlowingSpeechSynthesizer               │◀──┼──── Hermes 回答
                    │  │  AudioPlayer → 声卡                     │   │
                    │  └────────────────────────────────────────┘   │
                    └───────────────────────────────────────────────┘

AEC 参考路径: AudioPlayer 写入声卡 → on_audio_write → AEC.push_reference
打断路径: 新唤醒词 → generation += 1 → agent.interrupt() → TTS.interrupt() → clear()
```

### 2.1 状态机

```
IDLE → 唤醒词命中
  → prompt_duration 期间门控（不送 ASR）
  → RECORDING → VAD 静音 1s → PROCESSING
    → on_final → _ask_hermes → agent.chat()
      → TTS 播放（_tts_playing=Ture）
      → AudioPlayer 排空 → enter_conversation_window()
        → CONVERSATION(8s) → VAD检测到说话 → 新一轮
        → 超时 → IDLE
```

---

## 3. 模块说明

### 3.1 音频采集层

| 属性 | 值 |
|------|-----|
| 库 | `sounddevice` |
| 采样率 | 16000 Hz |
| 位深 | 16-bit PCM |
| 声道 | 单声道 |
| 帧长 | 30ms / 480 样本 / 960 字节 |
| 延迟模式 | `latency="low"` |

### 3.2 唤醒词检测

| 配置 | 默认值 | 平台自适应 |
|------|--------|-----------|
| 引擎 | `sherpa-onnx` | — |
| 唤醒词 | `赫尔墨斯` | 可自定义 |
| 灵敏度 | `0.5` | 范围 0~1 |

### 3.3 回声消除（AEC）

使用linux自带的pipewire创建回声消除设备， 配置为默认输入设备

> **注意：** AEC 对扬声器失真敏感，音量过大时回声消除效果下降。提示音播放期间（`prompt_duration_sec`）启用门控，不送 ASR，作为兜底。

### 3.4 语音活动检测

| 配置 | 默认值 |
|------|--------|
| 引擎 | `webrtcvad` |
| 模式 | `3`（最激进） |
| 静音超时 | 1000ms |
| 最小语音 | 200ms |

### 3.5 实时语音识别

使用腾讯云官方 SDK（`tencentcloud-speech-sdk-python/`）。

| 配置 | 默认值 |
|------|--------|
| 引擎模型 | `16k_zh` |
| 采样率 | 16000 Hz |
| 格式 | PCM |

SDK 处理签名、WS 连接、重连。封装在 `asr_engine.py` 中。

### 3.6 对话引擎

通过 `AIAgent` 同进程嵌入。

| 参数 | 配置 |
|------|------|
| 模型 | `config.yaml` → `hermes_agent.model` |
| session_id | 固定，保持多轮对话连续性 |
| context_files | 启用 |
| memory | 启用 |
| 系统提示词 | 告知 AI 输入来自语音识别，回答简洁 |

### 3.7 语音合成

使用腾讯云流式语音合成 SDK（`tts/flowing_speech_synthesizer.py`）。

| 参数 | 默认值 |
|------|--------|
| 音色 | 101001（晓晓） |
| 格式 | PCM |
| 采样率 | 16000 |

---

## 4. 项目结构

```
hermes-voice-agent/
├── README.md
├── requirements.txt
├── config.yaml                # 全部配置
├── .env                       # 凭据（SecretId / AppId / API Key）
├── assets/
│   ├── prompt.wav             # 唤醒提示音 "我在"
│   ├── notification.wav       # 对话窗口提示音（双音 chime）
│   └── farewell.wav           # 告别语 "再见"
├── models/
│   └── sherpa-kws/            # sherpa-onnx KWS 模型（按需下载）
├── src/
│   ├── main.py                # 入口
│   ├── voice_frontend.py      # 状态机 + 音频采集 + 唤醒 + VAD
│   ├── asr_engine.py          # 腾讯云 ASR SDK 封装
│   ├── tts_engine.py          # 腾讯云 TTS SDK 封装
│   ├── aec_processor.py       # WebRTC AEC
│   ├── audio_player.py        # PCM 播放器
│   └── config.py              # 配置加载
├── tools/
│   ├── tts_gen.py             # TTS → WAV 文件生成
│   ├── measure_delay.py       # AEC 延迟探测
│   ├── test_aec.py            # AEC 对比测试
│   └── quick_aec_test.py      # 快速 AEC 验证
├── tests/
│   ├── test_asr_engine.py      # ASR 引擎 mock 测试
│   ├── test_tencent_asr.py        # 端到端 ASR 测试（VAD 录音）
│   └── test_sherpa_kws.py      # sherpa-onnx KWS 测试
├── tencentcloud-speech-sdk-python/  # 腾讯云语音 SDK（子模块）
└── API-DOCS/
    └── hermes-agent-语音输入前端方案.md
```

---

## 5. 配置参考

### config.yaml 完整结构

```yaml
voice:
  wake_word:           # 唤醒词
    enabled: true
    keyword: "赫尔墨斯"
    sensitivity: 0.5

  vad:                 # 语音活动检测
    engine: "webrtc"
    mode: 3
    silence_threshold_ms: 1000
    min_speech_ms: 200

  asr:                 # 实时语音识别
    engine_model: "16k_zh"
    needvad: true
    voice_format: 1

  tts:                 # 语音合成
    enabled: true
    engine: "tencent-cloud"
    voice_type: 101001
    codec: "pcm"
    sample_rate: 16000
    speed: 0.0
    volume: 0.0

  aec:                 # 回声消除
    enabled: true
    delay_ms: 85

  hermes_agent:        # 对话引擎
    model: "deepseek-v4-flash"
    max_iterations: 10
    session_id: "hermes-voice-session"
    system_prompt: |  # 见 config.yaml 原文

  conversation_window:  # 连续对话
    timeout_sec: 8.0
```

### .env

```env
VOICE_SecretId=AKIDxxxxx
VOICE_SecretKey=xxxxx
VOICE_AppId=125922xxxx
OPENROUTER_API_KEY=sk-or-xxxxx
```

---

## 6. 代际计数器（打断机制）

```python
_generation = 0  # 每次唤醒词递增

_cancel_current_output():
  → generation += 1                # 旧 _speak 调用自动失效
  → agent.interrupt()               # 停工具调用
  → TTS.interrupt()                 # 停语音
  → audio_player.clear()            # 清空缓冲区

_ask_hermes(text):
  gen = _generation                 # 快照
  agent.chat(text)
  _speak(response, gen)             # 携带快照

_speak(text, gen):
  if gen != _generation: return     # 已过期，丢弃
  → TTS 播放
```

---

## 7. 运行方式

```bash
# 安装依赖
pip install -r requirements.txt

# 配置凭据
cp .env.example .env
# 编辑 .env 填写腾讯云凭据 + LLM API Key

# 运行
python -u src/main.py
```
