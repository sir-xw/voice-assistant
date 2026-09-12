# 语音功能独立服务化（Voice Service）— WebSocket 架构设计

> 状态：**设计稿 v5（开发中）**
> v2 变更：进程概念由 Audio Service 统一更名 **Voice Service**；§10 音乐状态协作已定稿。
> v3 变更：新架构代码收敛为仓库根三个**平级、自包含**的子项目 —— `voice_service`、`hermes_gateway_plugin`、`music_coordinator`（只有 hermes gateway 客户端与 hermes 绑定）；**不再 `from voice_agent.xxx` 复用旧组件库**，可复用代码复制进 `voice_service`；旧单体实现 `hermes-voice-agent/` 已于 2026-09 删除（历史见 git）。
> v4 变更：去掉仓库根 `src/` 包裹层（每个子项目自带 pyproject，可装不同 venv）；配置/凭据/模型归属 `voice_service/` 子项目目录；代码骨架已启动。
> v5 变更（最新）：MCP 接入形态定为 **web（streamable HTTP）端口**（music_coordinator `--enable-mcp`，10 个 mpd_* 工具已可调）；M2 inbound 与 **M3 outbound**（playback 播报队列/等待音/连续对话窗口/turn 兜底/Music Coordinator 音乐避让）代码落地并通过 silent 冒烟；`voice_service` 启动参数 `--audio`（输入）/`--out`（输出），默认不占设备。
> 术语对照：

| 概念 | 代码包 / 目录 | 与 hermes 的关系 |
|---|---|---|
| Voice Service（语音服务，WS 服务端） | `voice_service/`（子项目） | 独立（零 hermes 依赖） |
| hermes voice gateway 客户端（`voice-platform` 插件） | `hermes_gateway_plugin/`（子项目） | **唯一依赖 hermes 的部分** |
| Music Coordinator（音乐状态协调器） | `music_coordinator/`（子项目） | 独立（零 hermes 依赖） |
| 旧语音前端项目 | `hermes-voice-agent/`（`voice_agent` 包） | **已删除**（可复用组件/工具迁入 voice_service；历史见 git） |

> 目标：把语音链路里与「音频硬件 / 信号处理 / 语音云 SDK」耦合的部分从 hermes gateway 进程剥离，成为独立的 **Voice Service**（WS 服务端）；hermes 侧的 voice gateway（`voice-platform` 插件）改造为 **WS 客户端**，接收语音识别结果、下发待朗读文本。
> 已确认的决策前提：
> - 部署以同机（127.0.0.1）为主，协议层兼容跨机（token 鉴权 + 可配置地址）；
> - 只改造 hermes gateway 插件模式；`VoiceApp`（`python -m voice_agent`）退役，其**可复用代码复制进 `voice_service`**（不 import 旧包）；
> - 先做 1:1 主连接（一个 Voice Service 对一个 hermes voice 客户端），协议带 `client_id` 预留多路扩展；
> - v1 只走「文本事件上行 / 文本播报下行」，麦克风与扬声器音频只存在于 Voice Service 本地；协议预留二进制音频帧类型以便未来扩展；
> - **Voice Service 是独立服务（定位类比飞书平台：hermes 生态的接入方，但不属于 hermes 的一部分），不读取 hermes profile（`~/.hermes/voice-agent.yaml` 退役）**。自身配置落在 **`voice_service/` 项目目录的 `config.yaml`（或独立 `voice_service.yaml`，实现定稿）的 `voice_service:` 段**，凭据走 `voice_service/.env`（`VOICE_*`），模型走 `voice_service/models/`；旧 `hermes-voice-agent/config.yaml` 的 `voice:` 段只属旧实现。

---

## 1. 背景与目标

### 1.1 现状（进程内嵌）

当前 `voice-platform`（现状代码在旧包 `voice_agent/gateway_plugin/adapter.py`，目标布局见 §4.1）把完整语音链路**嵌入 hermes gateway 进程**：

```
麦克风 ──► [sherpa KWS 唤醒] ──► [WebRTC VAD] ──► 腾讯云 ASR
                                                        │ (文本)
hermes gateway 进程 ─────────────► gateway 会话路由 ─► agent
                                                        │ (回复文本)
扬声器 ◄── AudioPlayer ◄── 腾讯云 TTS ◄── (情绪)分段 ◄─┘
         （以上全部在 gateway 进程内：voice_frontend/asr/tts/player/播放队列/提示音）
```

同构的第二条路径 `VoiceApp`（`voice_agent/main.py`）也已退役在即。

### 1.2 动机（为什么拆）

| 痛点 | 拆分后 |
|---|---|
| gateway 进程被音频栈污染：sherpa-onnx / 腾讯云语音 SDK / PortAudio / PipeWire 的崩溃（含 C 层 segfault 风险）会拖垮整个 LLM 网关 | 音频故障只影响 Voice Service，独立重启、独立降级 |
| 修一次 USB 麦克风要重启整个 gateway | 修复只影响音频相关服务/音频栈，gateway 无感 |
| 音频组件强依赖 `pipewire`/用户会话等运行环境，gateway 被迫做成 user service 并挂音频依赖 | gateway 回归纯网络服务，可任意部署/迁移；音频环境只需 Voice Service 满足 |
| gateway 重启期间整条语音链路不可用 | Voice Service 持续监听；gateway 重启后客户端自动重连即恢复 |
| 语音能力与 hermes 绑定死，无法被其它 agent/前端复用 | Voice Service 暴露标准 WS 接入点，未来可服务多个/其它客户端（协议已预留） |

### 1.3 目标架构总览

```
                        ┌─────────────────────────── hermes 机器 ───────────────────────────┐
                        │  hermes gateway 进程                                            │
                        │   ┌───────────────────────────────────────────────┐              │
  （可同机，默认）        │   │ voice-platform（voice gateway，改造为 WS 客户端）│              │
┌───────────────┐  WS   │   │  - 握手/注册 唤醒词→chat_id 映射               │              │
│ Voice Service │◄──────┼───│  - 收 asr_result → MessageEvent → gateway 会话 │──► agent     │
│  （服务端）     │──────┼──►│  - send()/post_api_request 钩子 → speak 帧下发  │◄── 回复文本   │
└───────────────┘       │   └───────────────────────────────────────────────┘              │
   麦克风/扬声器           └───────────────────────────────────────────────────────────────┘
   模型/腾讯云凭据/音频自愈
```

两个进程的唯一接口就是 WebSocket；进程内部结构各自独立演进。

### 1.4 不做的事（非目标）

- 不修改 hermes-agent 源码（`/root/git/hermes-agent/` 约定不变）；
- 不把麦克风/扬声器音频经 WS 传输（v1；预留帧类型）；
- 不在 Voice Service 内管理 agent/会话 —— 会话、记忆、LLM、工具仍归 hermes gateway；
- 不保留 `VoiceApp` 模式与 `.hermes/plugins/speech-relay`、`voice_agent/speech_bridge.py`、`plugin_hooks.py`（hermes 侧的新客户端自带 `post_api_request` 钩子，直接内联，无需进程内桥）。

---

## 2. 角色与职责边界

| 职责 | Voice Service（新进程） | Hermes Voice Gateway（改造 adapter） |
|---|---|---|
| 音频采集/设备选择/重采样/采集失效检测（无数据→关流告警，恢复靠重启语音服务进程） | ✅ | ❌ |
| 唤醒词检测（sherpa KWS + keywords.txt） | ✅ | ❌ |
| WebRTC VAD / 连续对话窗口 / 提示音 / 告别语 / 等待提示音 | ✅（语音侧状态机整体迁移） | ❌ |
| 腾讯云 ASR（WebSocket、句子回调、音频缓冲） | ✅ | ❌ |
| 说话人识别 voiceprint（声纹提取需句子级音频缓冲） | ✅ | ❌（只收已标注的文本） |
| 腾讯云 TTS / 情绪分段合成 / AudioPlayer / 播报串行队列 | ✅ | ❌ |
| TTS 播报的音乐避让（每段播报前暂停、播完恢复） | ✅（每段播报前 `hold()`、播完 `release()` → Music Coordinator，见 §10） | ❌ |
| WS 接入点（服务端）：连接管理、心跳、鉴权、路由 | ✅ | ❌ |
| WS 客户端：连接/重连/认证 | ❌ | ✅ |
| 唤醒词 → 会话映射（`wake:<名>` chat_id）、channel_overrides（每会话 prompt/model） | ❌ | ✅ |
| LLM 回复捕获：`send()` 兜底 + `post_api_request` 钩子（中间轮/最终轮/`[FINISH]`） | ❌ | ✅ |
| 系统通知静默过滤、最终回复去重、身份前缀等「文本语义」 | ❌ | ✅ |
| 腾讯云凭据（`VOICE_SecretId/Key/AppId`） | ✅（只在语音进程出现） | ❌（可移除） |

> 注：MPD 的播放/暂停/停止等**实际写操作统一收敛到 Music Coordinator**（§10）——它是唯一操作 MPD 的实体；Voice Service（避让）与 hermes agent（意图）都只是它的 API 调用方。

**一句话边界**：Voice Service 负责「耳朵和嘴」，Voice Gateway 负责「大脑的接线」；两者之间只传「事件文本」与「朗读文本」。

---

## 3. 通信协议设计（核心）

### 3.1 传输与端点

- WebSocket over TCP，v1 绑定 `127.0.0.1:8765`（可配置 `service.host/port`）。
- 跨机部署：必须启用 `service.token`（握手 `Authorization: Bearer <token>`），并建议 WireGuard/VPN 或 TLS（`wss://` 由反代终止亦可）。
- 每个 WS 消息为 **一行 JSON**（`\n` 分隔，便于日志与排障）。
- 心跳：应用层 `ping`/`pong` 帧，间隔默认 20s；双方 3 次未收到即判断线。
- 可选附带 HTTP `GET /healthz`（同端口，非 upgrade 请求返回 `200 ok`）供 systemd/监控探活。

### 3.2 帧 envelope（统一外壳）

```json
{
  "v": 1,
  "type": "<帧类型>",
  "seq": 123,
  "client_id": "voice-gateway-1",
  "ts": 1710000000.123,
  "data": { }
}
```

- `seq`：发送方自增序号。命令帧要求对端回 `ack`（同 `seq`），用于日志串联、重连去重与超时诊断；事件帧 `seq` 可省（audio→client 事件以 `evt` 计数）。
- `client_id`：注册身份。v1 单客户端；服务端对第二个并发连接默认拒绝（`409 conflict`），协议字段保留以支持未来多路。

### 3.3 帧类型总表

**C→S（hermes voice gateway → Voice Service，命令）**

| type | 方向触发 | data 字段 | 说明 |
|---|---|---|---|
| `hello` | 连接建立后首帧 | `{client_id, caps:{interim:false, voiceprint:true}}` | 认证 + 上报能力。**不再上报唤醒词**：唤醒词↔助手映射由 Voice Service 维护（config `wake_word.assistants`），客户端只消费上行事件里的助手名 |
| `speak` | hook/send 有文本要朗读 | `{id, wake, kind:"interim"\|"final"\|"raw", segments:[{emotion, text}], turn_seq}` | 唯一播放入口；进入串行播报队列 |
| `control` | 流程控制 | `{action:"close_window"\|"dismiss_reply"\|"reload_kws", ...}` | 关对话窗口（`[FINISH]`）；`dismiss_reply`/`reload_kws` 为 M4 后可选增强（§9 裁决 A / §6.4） |
| `interrupt` | gateway 主动取消播放（罕见；本地打断通常不需要） | `{all:true, up_to_id?}` | 打断 TTS + 清播放队列 |
| `ping` / `bye` | 心跳 / 优雅断开 | `{}` / `{reason}` | — |

**S→C（Voice Service → hermes voice gateway，事件/应答）**

| type | 时机 | data 字段 | 说明 |
|---|---|---|---|
| `welcome` | hello 应答 | `{ok, my_wakewords:[{name, keywords:[]}, ...], version}` | 服务端回报实际助手表（config `wake_word.assistants`），供客户端知情/人工对照 |
| `ack` | 命令应答 | `{seq, ok, error?}` | 每命令一 ack；`speak` 之外的基础可靠性 |
| `wake_word` | KWS 命中（本地已播提示音/打断） | `{keyword, wake, ts}` | 客户端记录「当前活跃助手名」（wake=KWS `@助手名`） |
| `asr_interim` | ASR 中间结果（需 hello 声明 caps.interim） | `{text, wake}` | 可选，v1 默认关闭 |
| `asr_sentence` | 每个完成句（voiceprint 开时附带说话人） | `{text, speaker_id?, speaker_label?, start_ms?, end_ms?}` | 供客户端观察/日志 |
| `asr_result` | VAD 判定整段结束、ASR final（**inbound 主事件**） | `{text, wake, message_id, turn_seq}` | 客户端据此构造 `MessageEvent`；声纹启用时 `text` 已按句带 `[名字 (ID: 编号)]` 前缀 |
| `speak_done` | 一段/一 final 播报完成 | `{id, kind, ok, error?}` | 供客户端记录「播完」与告警 |
| `error` | 服务端异常 | `{code, message}` | — |
| `pong` | ping 应答 | `{}` | — |

> 说明：语音状态机变化（`idle/recording/waiting_reply/conversation/…`）v1 **不上行**（本地日志即可排障，见 §9 裁决记录 B）；客户端决策所需信息已由 `wake_word`/`asr_result`/`speak_done` 天然携带。未来做可视化/调试面板时再加 `state` 事件（协议加一帧即可，不影响其它帧）。

### 3.4 关键时序

#### 3.4.1 一问一答（正常）

```
Voice Service                                   Voice Gateway (hermes)
     │  ── KWS 命中「小布小布」，本地：打断/提示音 ──┐
     │◄────────────────────────────────────────────┘（纯本地，无需等待）
     │ ── wake_word{keyword:"小布小布"} ──────────────────────────────► │ 记录 active wake
     │ ── (VAD 语音结束, ASR final) ──► asr_result{text,wake,msg_id} ─► │ MessageEvent(wake:小布)
     │                                                                 │      → gateway 会话 → agent
     │ 本地进入 waiting_reply，循环播等待提示音                        │ hook 收到中间轮文本
     │ ◄── speak{id,kind:"interim",segments:[(happy,"好的…")]} ────────┤
     │ 停等待音 → 串行播报（只播，不进窗口）                            │
     │ ◄── speak{id,kind:"final",segments:[(neutral,"…")]} ────────────┤ hook finish_reason=stop
     │ 播 final → 通知音 → 本地进入 conversation(窗口期)                 │（或 gateway send() 兜底）
     │ ── speak_done{id} ─────────────────────────────────────────────► │
     │ (8s 内 VAD 直接听，无需唤醒词)                                   │
```

#### 3.4.2 连续对话（窗口期内再次说话）

```
     │ (conversation 状态, VAD 检测语音→直接 ASR, 不播提示音)
     │ ── asr_result{text, wake(沿用上轮)} ───────────────────────────► │ 同 chat_id 新消息
     │ …同上（gateway 按 busy_input_mode 处理同会话并发，v1 沿用 interrupt）
```

#### 3.4.3 打断（物理层与逻辑层解耦）

```
 用户再次喊唤醒词（或窗口期内抢话）
   Voice Service：本地立即 interrupt（TTS 停 + 播放器清空 + 队列清空）   ← 物理层，零延迟
   Voice Service：上行 wake_word / 新一轮 asr_result                    ← 逻辑层，尽力而为
   Voice Gateway：收到新 asr_result → 同 chat_id 新 MessageEvent
                  → gateway interrupt（旧 turn 作废，不再投递回复）
   （若旧 turn 的在途 speak 帧已发出：客户端以 turn_seq 关联，见 §3.5 可靠性）
```

#### 3.4.4 无回复 / 回复失败（兜底）

```
   Voice Service：上行 asr_result 后 waiting_reply，循环播放等待提示音
     ├─ 收到首个 speak → 停止等待音，正常播报
     └─ waiting_reply 超时（默认 45s，可配）→ 停音 → 播告别语 → 回 IDLE
   Voice Gateway：v1 不主动通知「无回复」——靠上面的超时兜底
                  （用户也可随时喊唤醒词打断等待）；
                  若 M4 验证 gateway 存在干净的 turn 结束信号，再启用
                  control{dismiss_reply} 让服务立即结束等待（体验更好，见 §9 裁决记录 A）
```

### 3.5 可靠性 / 幂等 / 去重语义

| 机制 | 设计 |
|---|---|
| 播放唯一性 | Voice Service 内全局单播报队列（消费者进程），同一时刻只有一个声音 |
| 播放去重（hermes 侧） | 沿用现状：最终回复以 hook（`finish_reason=stop`）为主通道；`send()` 收到同一文本去重跳过；系统通知前缀静默 —— 文本语义全在客户端，服务端无状态 |
| 迟到帧作废 | `speak` 带 `turn_seq`（客户端每上行一个 `asr_result` 自增）。Voice Service 收到**比当前已播报更新**的 `speak` 时，若队中还有旧 `turn_seq` 的帧则丢弃（防「被 gateway 打断的旧回复」事后播出来）。此机制是 v1 对 gateway interrupt 时序的兜底，正常情况下几乎不触发 |
| 重连去重 | `asr_result.message_id` 用 `voice-<unix_ms>` 单调生成；客户端重连后不补发历史 |
| 断线 | Voice Service 检测到客户端断开：停止播放/等待音，回 IDLE（不把语音状态留在半途）；恢复监听但**不上行**（唤醒只播提示音不识别，防隐私泄漏，见 §8） |
| 心跳 | `ping` 20s；超 60s 无响应判死，双方主动关闭由对端重连 |

### 3.6 播放语义细节（Voice Service 内，迁移自现状）

- `kind=final`：播完所有分段 → 播通知音 → 本地进入 `conversation` 窗口期（对话窗口是语音侧状态，**不再需要 hermes 参与**）。
- `kind=interim`：只播报（工具轮阶段文本），播完恢复等待提示音（若仍在 waiting_reply）。
- `kind=raw`：纯 TTS（内部/资产音之外的文本，如待汇报前缀已在文本中拼好）。
- `(情绪)文字` 分段解析：在**客户端**做——客户端持有原始文本，先判 `[FINISH]` / 系统通知前缀静默，再调用共享解析函数 `parse_emotion_segments` 得到 `segments` 数组随 `speak` 帧下发；Voice Service **只按 `segments` 播放、不做文本理解**。解析函数与 `VALID_EMOTIONS` 放入两端共享模块 `voice_service/protocol.py`（同时被 `hermes_gateway_plugin/client.py` 复用），防止两端漂移。客户端仍负责 `[FINISH]` 识别（它同时决定是否结束 turn）并转成 `control{close_window}`。

---

## 4. Voice Service 内部架构

### 4.1 代码结构（仓库根三个平级、自包含的子项目，各带独立 pyproject）

```
voice-assistant/（仓库根）
├── voice_service/                   # ★ 子项目：语音服务 = WS 服务端（零 hermes 依赖，独立 pyproject）
│   ├── pyproject.toml / README.md
│   ├── config.yaml                  #   本服务配置（voice_service: 段；或独立 voice_service.yaml）
│   ├── .env / models/               #   腾讯云凭据 / sherpa KWS 模型 / voiceprint_lib / keywords.txt
│   │                                #   （gitignore，部署时提供）
│   └── voice_service/               #   Python 包（python -m voice_service）
│       ├── __init__.py / __main__.py
│       ├── protocol.py              #   帧 schema/常量/parse_emotion_segments（与插件端共享契约）
│       ├── service_config.py        #   加载 voice_service: 段 + 腾讯云凭据 env
│       ├── server.py                #   asyncio websockets 服务端（M1 骨架版：鉴权/healthz/hello/ping）
│       ├── hub.py / session.py      #   （规划）连接管理拆分为 hub/session
│       ├── inbound.py / playback.py #   （规划，M2/M3）语音状态机编排 / 播报队列
│       ├── vad.py / voice_frontend.py / asr_engine.py / tts_engine.py /
│       │       audio_player.py / voiceprint.py   # ★ 组件（自旧 voice_agent 复制，import 已改写）
│       ├── tencentcloud_speech/     #   vendored 腾讯云语音 SDK（复制）
│       └── assets/                  #   提示音 wav（复制；package-data 随包分发）
│
├── hermes_gateway_plugin/           # ★ 子项目：hermes voice gateway 客户端（唯一 hermes 绑定）
│   ├── pyproject.toml / README.md   #   entry point：voice-platform = hermes_gateway_plugin
│   └── hermes_gateway_plugin/
│       ├── __init__.py              #   入口 register()（延迟导入 adapter）
│       ├── adapter.py               #   VoiceAdapter（规划，M4）
│       └── client.py                #   WS 客户端（已实现：hello/welcome、ping、线程安全 send）
│
└── music_coordinator/               # ★ 子项目：音乐状态协调器（零 hermes 依赖，独立 pyproject）
    ├── pyproject.toml / README.md
    └── music_coordinator/
        ├── __init__.py / __main__.py  #   python -m music_coordinator
        ├── coordinator.py           #   intent/hold/effective 状态机（已实现，含自检）
        ├── mpd_conn.py              #   RealMpd / DummyMpd 后端
        ├── hold_ipc.py              #   Unix socket hold/release/status（已实现）
        └── tools_mcp.py             #   MCP server 工具面（占位，待 hermes MCP 接入形态）
```

包与依赖关系（只有 gateway 与 hermes 绑定）：

- 三个子项目**各自独立** `pyproject.toml`，互不打包依赖，可安装到不同 venv；
  与旧 `hermes-voice-agent/pyproject.toml` 互不干扰；
- **`voice_service` 自包含**：可复用组件（`vad`/`voice_frontend`/`asr_engine`/`tts_engine`/
  `audio_player`/`voiceprint`、vendored `tencentcloud_speech/`、`assets/` 提示音）已**复制**
  进 `voice_service/voice_service/`（包根平铺），组件内 `voice_agent.*` import 已改写为
  `voice_service.*`，运行时不 import 旧包；旧实现目录已删除；
- **`hermes_gateway_plugin` 唯一依赖 hermes**：作为 hermes 插件随 gateway 加载；它
  import `voice_service.protocol` **仅取帧契约**（voice_service 无 hermes 依赖，此方向依赖
  安全）——因此部署到 gateway 的 venv 时需先安装 `voice_service`；
- **`music_coordinator` 自包含**：零 hermes 依赖；`tools_mcp.py` 的 mcp SDK 为可选依赖
  （`pip install -e "./music_coordinator[mcp]"`）。

> 注：`voice_agent/profile_config.py`（hermes profile 模型路径约定）**不复制**——模型/声纹路径改由 `voice_service/config.yaml`（相对 voice_service 项目目录）解析。
> 注：`voice_agent/music_control.py`（playerctl 直控 + 进程内全局 `_previous_status`）**不复制**——其语义由 Music Coordinator 的 intent/hold 模型取代（§10），playerctl 路径待真机核对后删除。
> 注：`hermes_gateway_plugin` 的 `client.py` 与 `voice_service` 若需**无依赖共享帧定义**（未来想彻底解耦），可把 `protocol.py` 抽成第三个共享小包；v1 直接依赖 `voice_service` 包即可（它是纯 Python、无 hermes/音频依赖）。

### 4.2 线程 / 事件模型

```
                asyncio 事件循环（主线程）
   ┌────────────────────────────────────────────┐
   │ ws_server（hub/session/播放调度协程）        │
   │ playback_consumer（串行播报队列）            │
   └────────────────────────────────────────────┘
        ▲ run_coroutine_threadsafe / call_soon_threadsafe
        │
   ┌────┴─────────── 线程池 / SDK 回调线程 ───────────┐
   │ VoiceFrontend 音频处理线程（唤醒/VAD/失效检测）   │
   │ 腾讯云 ASR SDK 回调线程（on_start/sentence/…）    │
   │ 腾讯云 TTS SDK 回调线程（on_audio_chunk→player）  │
   │ AudioPlayer 播放线程（sounddevice OutputStream）  │
   └─────────────────────────────────────────────────┘
```

- 原则沿用现状：**音频/云 SDK 的耗时与阻塞操作绝不在事件循环里做**；跨线程投递统一 `asyncio.run_coroutine_threadsafe`。
- `VoiceFrontend` 事件回调（`on_wake_word`/ASR 完成等）在音频线程触发 → 封装为「事件」放入 inbound 队列，由 asyncio 侧消费并转 WS 上行（避免 SDK 线程直接写 WS）。

### 4.3 语音侧状态机（迁移+扩展）

```
IDLE ──唤醒词命中──► (打断+TTS停+提示音) ──► RECORDING ──VAD 静音超时──► PROCESSING(等 ASR)
PROCESSING ─ASR final 上行─► WAITING_REPLY(循环等待提示音)
WAITING_REPLY ─speak(interim)─► 播报 → 回 WAITING_REPLY
WAITING_REPLY ─speak(final)──► 播报 → 通知音 → CONVERSATION(窗口期)
CONVERSATION ─VAD 说话─► RECORDING（直接听，无提示音）
CONVERSATION ─超时─► IDLE（告别语）
任何状态 ─唤醒词/抢话─► 打断 →（如上）
WAITING_REPLY ─超时─► 停等待音 → 告别语 → IDLE
（可选增强，M4 后再定：WAITING_REPLY ─control(dismiss_reply)─► 同上）
```

- 与现状（`VoiceFrontend` 状态机 + `adapter` 里播放/等待音逻辑）的差异：**对话窗口、等待音、通知音全部收敛进服务进程**，客户端不再需要任何「语音进度」回调。
- 状态变化 v1 只写服务本地日志，不上行（见 §3.3 表后说明与 §9 裁决记录 B）。

### 4.4 配置（Voice Service 侧）

**落点：voice_service 目录 `config.yaml` 新增 `voice_service:` 段**（新 Voice Service 自持；旧 `hermes-voice-agent/config.yaml` 的 `voice:` 段属旧实现，两者不同文件）。定位决策见头部决策前提与 §9 裁决记录 C：

- Voice Service 是**独立服务，不读取 hermes profile**（`~/.hermes/voice-agent.yaml` 退役、`voice_agent/profile_config.py` 的模型路径约定不复制）；
- 凭据走**voice_service 目录 `.env`**（`VOICE_SecretId/Key/AppId`，gitignore），模型走**voice_service 目录 `models/`**（从旧目录复制/迁移：sherpa KWS 模型、`voiceprint_lib`）。**KWS `keywords.txt` 由 Voice Service 自身生成**（`voice_service/kws_words.py` 按 `wake_word.assistants` 渲染，启动时同步校验）——旧 `hermes-voice-agent/tools/gen_keywords.py` 随退役，词表不再依赖 hermes 配置；
- 内容 ≈ 旧 `config.yaml` 中 `voice:` 段的**语音硬件/云 SDK 部分**（不含 `agents`/会话类——会话归 hermes），再叠一个 `service:` 服务段；
- 服务模块本身**复制**到 `voice_service/voice_service/`（组件平铺于包根，含 vendored SDK 与 `assets/` 提示音），不 import 旧包。

```yaml
# voice_service 目录 config.yaml
voice_service:
  service:
    host: "127.0.0.1"        # 跨机部署改为 0.0.0.0 + token
    port: 8765
    token: ""                # 空=仅限本机（同机默认）
    ping_interval_sec: 20
    heartbeat_timeout_sec: 60
    wait_reply_timeout_sec: 45
  wake_word:
    enabled: true
    assistants:            # 唤醒词↔助手映射（唯一源，KWS 命中返回 @助手名 上行）
      - name: 泡泡          # 助手名 = gateway chat_id "wake:泡泡"
        keywords: ["泡泡管家", "你好泡泡"]
      - name: 小布
        keywords: ["小布小布"]
  vad:         # ...
  asr:         # ...（腾讯云引擎配置；凭据走项目 .env 的 VOICE_SecretId/Key/AppId）
  tts:         # ...
  voiceprint:  # ...（特征库路径 models/voiceprint_lib、speaker_names —— 声纹只在服务侧出现）
  assets:      # prompt/notification/farewell/wait_cue 路径
  music_coordinator:  # 播报避让：每段 TTS 前/后调协调器 hold/release（Unix socket），不直连 MPD（见 §10）
    socket: "/run/user/0/music-coordinator.sock"
    enabled: true
  mic:         # 设备选择/采集失效检测参数（no_data_timeout_sec、fail_cooldown_sec）
```

> 与 hermes 侧配置的边界：hermes 自己的接入参数（WS `url`/`token`、`channel_overrides` 按 chat_id=`wake:<助手名>` 覆盖 prompt/model）仍写在 hermes 的配置文件里（`~/.hermes/config.yaml`），那是 **hermes 平台插件的标准配置位**（类比飞书插件的 bot token 也在 hermes 配置里）——「不放 hermes profile」指的是 Voice Service **自身**的语音/凭据/模型/词表配置，两者不冲突。唤醒词↔助手映射只此一份（Voice Service `wake_word.assistants`），hermes 侧不再配置 `platforms.voice.extra.wakewords`。

### 4.5 组件装配（对照现状 `adapter._init_*`；右列模块均位于 `voice_service/` 包内）

| 现状（adapter 内） | 迁移至 |
|---|---|
| `_init_frontend`（VoiceFrontend 全量配置 + 回调） | `voice_service/inbound.py`：回调改为「发事件」而非「调 hermes」 |
| `_init_asr` + `_on_asr_start/_on_asr_sentence/_on_asr_complete/_on_asr_error` | `inbound.py`：`on_complete` → 上行 `asr_result`（含 voiceprint 拼好的 `[名字 (ID: 编号)]` 前缀）；`on_sentence` → 声纹识别（本地）→ 附 label |
| `_init_playback` + `_playback_consumer/_play_segments/_play_asset/_wait_tone_*` | `voice_service/playback.py`：队列消费者；`speak` 帧即入队元素 |
| `_on_interrupt_request`（打断 TTS/清队） | `inbound.py` 本地打断 + `control/interrupt` 帧处理共用同一清理函数 |
| `_init_voiceprint` | `voice_service` 装配（凭据与模型同在服务侧） |
| `check_requirements`（凭据+模型目录） | Voice Service 启动前置检查（照搬） |
| `player_pause/player_resume`（`music_control`，TTS 播报避让） | `playback.py` 改为调用 **Music Coordinator `hold/release`**（§10），不再直接/经 playerctl 操作 MPD |

---

## 5. Hermes 侧 Voice Gateway 改造（WS 客户端）

### 5.1 适配器职责变化

`VoiceAdapter` 不再装配任何语音组件，改造为：

```
connect()        → client.connect(service_url, token)
                   （失败返回 False → gateway 平台告警禁用，见 §6.3）
WS 上行处理       → asr_result → MessageEvent(wake_chat_id(wake)) → handle_message()
                   wake_word → 记录 _current_wake（路由归属）
WS 下行          → speak{kind,segments}：来源
                     ① post_api_request 钩子（主通道，同步回调）
                        - finish_reason=stop → kind=final + 去重登记
                        - finish_reason=tool_calls → kind=interim
                        - [FINISH] → control{close_window}（不朗读）
                     ② send() 兜底（钩子漏触发/被改写内容，文本语义照搬现状）
                  发送走「线程安全写队列 → asyncio writer」，绝不阻塞 agent 线程
```

### 5.2 保留不动的 hermes 侧语义

- 平台注册契约（`register_platform`、`check_fn`、`validate_config`、`toolsets_for_source`、授权上游）；
- `channel_overrides`（每会话 prompt/model，按 chat_id=`wake:<助手名>` 命中）；mpd 工具**不再**经 `platform_toolsets.voice` 披露（已 MCP 化，见 §10）；
- 系统通知前缀静默、最终回复去重（`_last_final_by_chat`）、`[FINISH]` 关窗、平台提示词（情绪/说话人/简洁规则文字）**原样保留在客户端**；
- **不再注册 mpd 工具**：音乐控制改由 Music Coordinator 的 MCP intent API 提供（§10）；hermes 侧只配置 MCP 接入，`register()` 删除 `register_mpd_tools()` 调用。
- **`platforms.voice.extra.wakewords` 不再配置**：唤醒词↔助手映射的唯一源迁移到 Voice Service 的 `wake_word.assistants`（§4.4），客户端 hello 不再上报词表、`validate_config` 不再校验；会话路由只用上行事件携带的助手名。

### 5.3 重连与生命周期

- 客户端实现独立重连循环（`asyncio`）：断线 → 指数退避重连（1s→30s 封顶）→ 重连成功先 `hello`。
- 与 gateway 平台重连语义配合：`connect(is_reconnect=True)` 时若已有 WS 会话则直接复用/重建；平台断开事件由 gateway 原有 watcher 管理。
- Voice Service 先于 gateway 启动不是硬依赖：gateway 的 voice 平台启动失败只影响语音，其余平台照常（现状已是该行为）。

---

## 6. 配置、部署与运维

### 6.1 配置切分总表

| 配置 | 现在（gateway 进程） | 拆分后归属 |
|---|---|---|
| 唤醒词↔助手映射（keywords→name、session、prompt 提示词） | ✅（`platforms.voice.extra.wakewords`） | **归 Voice Service**：`voice_service/config.yaml` 的 `wake_word.assistants` 唯一维护，启动时生成 KWS `keywords.txt`（`voice_service/kws_words.py`） |
| `platforms.voice.extra.service`（`url`、`token`） | — | **hermes 侧接入参数**（唯一保留的 hermes 侧 voice 配置） |
| 旧 `hermes-voice-agent/config.yaml` 的 `voice:` 段（kws/vad/asr/tts/voiceprint/mic/assets + agents） | ✅（VoiceApp/旧插件读） | 语音硬件部分**复制到voice_service 目录 `config.yaml` 的 `voice_service:` 段**；`agents` 等会话类随 VoiceApp 退役 |
| 腾讯云凭据 env（`VOICE_*`） | ✅（gateway 进程） | **只注入 Voice Service 进程**（voice_service 目录 `.env`）；hermes 侧不再需要 |
| sherpa 模型 / voiceprint_lib | ✅（profile `~/.hermes/models/`） | **单一副本归 Voice Service（voice_service 目录 `models/`）**；gateway 客户端不碰模型 |
| `~/.hermes/voice-agent.yaml`（旧 voice-platform 读的 profile 语音配置） | ✅ | **退役**：服务端改读 voice_service 目录 `config.yaml`，客户端不再需要 |

> 注：Voice Service 的配置/凭据/模型（`voice_service:` 段、`.env`、`models/`、KWS `keywords.txt`）全部落在 **`voice_service/` 子项目目录**（`/root/git/voice-assistant/voice_service/`），旧实现目录已删除（历史见 git）。

### 6.2 systemd

新增 **user 服务**（须访问 per-user 音频栈，沿用现 gateway 用户服务的经验；WorkingDirectory 指向 **`voice_service` 子项目目录**——模型与 `keywords.txt` 按其相对路径解析，cwd 关键）：

```ini
# ~/.config/systemd/user/voice-service.service
[Unit]
Description=Voice Service (mic/kws/asr/tts/playback) — 语音服务（独立进程）
After=pipewire.service pipewire-pulse.service pulseaudio.service
Wants=pipewire-pulse.service

[Service]
Type=simple
ExecStart=/usr/local/lib/hermes-agent/venv/bin/python -u -m voice_service
WorkingDirectory=/root/git/voice-assistant/voice_service
Environment="XDG_RUNTIME_DIR=/run/user/0"   # per-user 音频栈
Environment="HOME=/root"
EnvironmentFile=-/root/git/voice-assistant/voice_service/.env   # VOICE_SecretId/Key/AppId
Restart=always
RestartSec=3
```

- `hermes-gateway.user.service` 的音频 drop-in（`audio.conf`：`After/Wants=pipewire…`）**可移除**，gateway 不再依赖音频栈；
- `hermes-voice-agent.service`（VoiceApp）退役；
- 新增 **`music-coordinator.service`**（§10）：独立常驻进程，`python -m music_coordinator`；不依赖音频栈，但需可连 MPD（localhost:6600），供 hermes（MCP intent）与 Voice Service（hold IPC）双方访问；
- 音频设备运维（健康检测/修复/音量）脚本与 systemd 单元为环境特定内容，已移出本仓库、不随项目分发。

### 6.3 启动顺序与降级

1. `voice-service` 先起（无 gateway 也能跑：无客户端时处于「监听但不识别/识别但不上行」模式，见 §3.5）。
2. `hermes-gateway` 后起：voice 平台 `check_fn` 改为 **WS 连通性探测**（TCP connect + `hello`），连不上 → 平台禁用并告警（不影响其它平台），Voice Service 起来后由 gateway 重连 watcher 恢复。
3. 双方日志带 `client_id/seq`，用 `journalctl -u voice-service` / gateway 日志对照排查。

### 6.4 唤醒词变更流程（涉及两端）

```
改 voice_service 目录 config.yaml 的 voice_service.wake_word.assistants（唯一源：
助手名 name + 触发词 keywords）
→ 重启 voice-service（启动时 kws_words 自动重生成/校验 KWS keywords.txt）
→ gateway 无需改动（hello 不再上报词表；welcome 回显服务端实际助手表可人工对照）
→ 若改了助手名（chat_id），需同步 hermes 的 channel_overrides 键与记忆/会话清理
```

（v1 不做运行期 KWS 热更；如需，后续加 `control{reload_kws}` + 重建 KeywordSpotter，作为可选增强。）

---

## 7. 分阶段实施计划

| 里程碑 | 内容 | 验证 |
|---|---|---|
| **M0 协议与共享模块** | 落 `voice_service/protocol.py`：帧类型/schema/`parse_emotion_segments` 抽取；帧编解码单测 | `python -c "import voice_service.protocol"` 自检 + pytest-less 手工断言脚本 |
| **M1 Voice Service 骨架** | server/hub/session：监听、hello 握手、鉴权、ping、healthz；无语音组件也能跑 | 用 `websockets` 写临时 client 脚本走通 hello/ack/ping |
| **M2 inbound 迁移** | VoiceFrontend + ASR + voiceprint 装配进服务；唤醒/句子/结果 → 上行事件；本地打断；waiting_reply 超时 | 无头可测：向服务端注入合成 PCM（现有 `tests/test_tencent_asr.py --sine` 思路）验证 asr_result 上行；真机验证唤醒 |
| **M3 outbound 迁移** | playback.py：speak 队列/分段/资产音/通知音/对话窗口/wait tone；`control`/`interrupt` 帧 | 打桩 TTS 单测播放序列；真机听感冒烟 |
| **M4 gateway 客户端改造** | `hermes_gateway_plugin/client.py` + `adapter.py` 瘦身；hook/send → speak 帧；去重/静默/[FINISH] 保留；check_fn 改为 WS 探测 | 双进程冒烟：唤醒→问答→播报→连续对话→打断→多唤醒词切换→gateway 重启自动恢复 |
| **M5 部署切换** | systemd 新 unit（voice-service / music-coordinator）、voice_service 目录 `config.yaml` 加 `voice_service:` 段并迁移语音配置与 `.env`/`models`/keywords 生成链、gateway drop-in 移除、VoiceApp 停用、旧 `voice:` 段/`voice-agent.yaml` 清理 | 真机全链路 + 重启/断电恢复演练 |

> 建议顺序：M0→M1→M2→M3 可并行开发但先各自可测；M4 依赖 M1-M3 的协议稳定；M5 最后真机切换，**切换前保留旧 gateway 服务可一键回滚**（旧 adapter 代码留一个 tag/分支）。

---

## 8. 风险与对策

| # | 风险 | 对策 |
|---|---|---|
| R1 | 双进程时序：gateway 重启期间用户说话 → asr_result 无处投递 | Voice Service 无客户端时**只播提示音不识别**（可配），避免隐私录音无主；gateway 恢复即好 |
| R2 | 被 gateway 打断的旧 turn 回复「迟到」到 Voice Service 被播出来 | `speak.turn_seq` + 服务端丢弃旧 seq 队列帧（§3.5）；正常路径 gateway interrupt 已保证 |
| R3 | `post_api_request` 钩子是同步回调，WS 发送异步 → 阻塞 agent 线程 | 客户端线程安全写队列（现状 `_enqueue_playback` 模式照搬），绝不直接 await 发送 |
| R4 | 音频 C 层崩溃（现状已记录 segfault 史） | 崩溃域收敛到 Voice Service；systemd `Restart=always`；gateway 完全免疫 |
| R5 | 播放队列语义与「对话窗口/等待音」跨进程错位 | 全部语音侧状态收敛服务端（M3 单测覆盖状态序列）；客户端只做文本语义 |
| R6 | 配置双份漂移 | **已消除**：唤醒词↔助手映射唯一源 = Voice Service `wake_word.assistants`（§4.4/§6.4），KWS 词表由其生成；hermes 侧不再持有词表 |
| R7 | 跨机部署安全 | token 鉴权 + 建议 WireGuard/TLS；v1 默认 127.0.0.1 零暴露 |
| R8 | 旧 `hermes-voice-agent.service`/`VoiceApp` 退役动作遗漏 | M5 清单化：停用 unit、删除 speech-relay/plugin_hooks 引用、README 更新 |

---

## 9. 已裁决记录

- **A. 无回复等待的兜底（`dismiss_reply` 不进 v1）**：客户端目前没有可靠的「本轮对话正式结束」信号（`post_api_request` 钩子只感知"又来了一段话"；`send()` 是兜底通道、不一定每次都触发），「客户端主动告知无回复」这个机制暂时落不了地。v1 采用 **`waiting_reply` 超时（默认 45s，可配）兜底**：超时 → 停等待音 → 告别语 → 回 IDLE；用户也可随时喊唤醒词打断。`control{dismiss_reply}` 仍保留在协议表，**待 M4 验证 gateway 是否有干净的 turn 完成信号后再启用**（若有，无回复时不必干等 45s，超时降为保险丝）。正文对应：§3.4.4、§4.3。
- **B. `state` 事件不进 v1**：语音状态机变化只写服务本地日志，不上行。客户端做决策所需信息（唤醒/识别结果/播完）已由业务帧携带，`state` 只是"情报"，当前没有消费方；等有真实需求（可视化/调试面板）再加帧。正文对应：§3.3 表后说明、§4.3。
- **C. 配置落点（用户已拍板，随 v3 布局更新）**：Voice Service 定位为独立服务（类比飞书平台，不属于 hermes 的一部分），**不读取 hermes profile**（`~/.hermes/voice-agent.yaml` 退役）；代码在 voice_service 子项目（voice_service/）（自包含、组件复制不 import）；自身配置放在**voice_service 目录 `config.yaml` 的 `voice_service:` 段**，凭据走voice_service 目录 `.env`（`VOICE_*`），模型走voice_service 目录 `models/`；旧 `hermes-voice-agent/config.yaml` 的 `voice:` 段只属旧实现。正文对应：头部决策前提、§4.1、§4.4、§6.1、§6.2。
- **D. turn_seq 丢弃粒度（用户已确认推荐）**：按「整个旧 `turn_seq` 的 speak 帧」整体丢弃，不做逐帧 `id` 精确丢弃。正文对应：§3.5「迟到帧作废」。
- **E. 健康检查形态（用户已确认推荐）**：`/healthz` 挂在 Voice Service 监听端口（`websockets` 同端口应答 HTTP GET）。正文对应：§3.1。

> 补充话题（mpd 工具 MCP 化后，音乐播放状态如何跨进程协作）见 **§10（已定稿）**。

---

*配套改动提示（后续实施时）：hermes-voice-agent/README.md 架构图与部署章节、docs/gateway-voice-plugin-usage.md、AGENTS.md「当前状态」小节、`config.yaml.example` 增加 `voice_service:` 段示例（旧 `voice:` 段标注待清理）。*

---

## 10. 补充设计：音乐播放状态协作（mpd 工具 MCP 化）—— 已定稿

> 状态：**已定稿（用户确认）**。新增组件 **Music Coordinator（§10.3）**：intent API 以 MCP server 形式暴露给 hermes agent（LLM 工具调用），hold API 经本地 Unix socket 提供给 Voice Service，协调器是**唯一写 MPD 的实体**。拍板要点：协调器**独立进程**；MCP 工具披露粒度、playerctl 退役列为**实施期检查项**（§10.6）。相关正文 §2、§4.1、§4.4、§4.5、§5.2、§6.2 已同步修订。
> 背景问题（用户提出）：音乐播放系列工具（`mpd_*`）如果改用 **MCP 方式接入 hermes**，跨进程拆分后，如何让 Voice Service 知道「语音会话/播报结束后，音乐应该保持播放还是暂停」？

### 10.1 现状机制与它的失效点

现状（进程内）已经有一个「预期状态」雏形，分散在两处：

- `voice_agent/music_control.py`：全局变量 `_previous_status` + `player_pause(force=True)`（读当前状态→保存→暂停）+ `player_resume()`（仅当保存值是 Playing 才恢复，一次性）；
- `voice_agent/mpd_tool.py`：每个 mpd handler 执行后调 `set_expected_status()`（`play/resume/next/previous→"Playing"`、`pause→"Paused"`、`stop→None`），**把 AI 指令的意图写进同一个全局变量**，从而影响后续 `player_resume()` 的决策。

它要解决的语义正是用户的问题，但有两个结构性缺陷：

1. **两份语义共用一个全局变量**：`_previous_status` 既存「TTS 播报前我暂停了正在播放的音乐」（瞬时避让），又被 `set_expected_status` 改写成「用户/agent 最新想要的状态」（持久意图）。两者的生命周期不同（一条播报 vs 一个指令），混在一起后靠调用次序碰运气，跨进程后更是彻底失效。
2. **控制通路不统一**：`music_control.py` 用 `playerctl`（MPRIS），`mpd_tool.py` 用 `python-mpd2` 直连 MPD(6600)——实际是两套并存的控制路径。

拆分 + MCP 化后，能操作 MPD 的实体变成三个且互不可见：**hermes agent（经 MCP，意图）**、**Voice Service（TTS 播报避让）**、**用户直接操作**。必须有一个跨进程的归因机制。

### 10.2 关键洞察：把「意图」和「避让」分开，Voice Service 就无须判断

用户问的「会话结束后该播放还是暂停」，本质上不是一个需要 Voice Service 判断的问题。把暂停拆成两种不同语义：

| | 意图 intent | 避让 hold |
|---|---|---|
| 谁表达 | 用户 → agent → MCP 指令（`play/pause/stop/...`） | Voice Service（每段 TTS 播报前/后） |
| 语义 | 「用户想让音乐处于什么状态」，**持久** | 「我要念一段话，音乐暂时让路」，**瞬时** |
| 规则 | 最新指令胜出（last-wins） | 不改意图；释放后回到意图状态 |

于是：

- **Voice Service 永远不用回答「会话结束该播放还是暂停」**——它只做两件事：每段 TTS 前 `hold()`、每段 TTS 播完 `release()`；释放后回到哪个状态由意图决定，天然正确：
  - 对话前在放歌、用户全程没动音乐 → 意图 = 播放中 → 每段 TTS 后都恢复播放（与现状体验一致）；
  - 用户说「暂停音乐」（agent 经 MCP `pause`）→ 意图 = 暂停 → 之后无论多少段 TTS 播报，音乐**保持暂停**，绝不会因为「会话结束自动恢复」而违背用户意图；
  - 用户说「停止」→ 意图 = 停止，同理。
- 会话/对话窗口/`[FINISH]`/告别语等语音侧概念**与音乐状态彻底解耦**：关窗不影响意图，音乐状态只随 MCP 指令变化。

### 10.3 新增组件（已定）：Music Coordinator（音乐状态协调器，唯一写 MPD 的实体）

```
用户 ──语音──► hermes agent ──MCP intent API──┐
                                              ▼
用户 ──直接──────► MPD 播放器 ◄──写── Music Coordinator（协调器）
                                              ▲
Voice Service ──本地 IPC hold/release API─────┘
```

协调器（可视为 mpd-bridge / 正规化后的 `set_expected_status`）持有：

```
intent       ∈ {playing, paused, stopped}    // 最新 agent 指令（含 seq，last-wins）
hold_count   ≥ 0                              // Voice Service 活跃避让数（可嵌套，幂等）
effective    = (hold_count > 0) ? paused : intent    // 唯一落盘到 MPD 的目标态
```

- **intent API（hermes agent，经 MCP 暴露）**：`play/resume/next/previous → intent=playing`；`pause → intent=paused`；`stop → intent=stopped`。注意：即使此刻正在 `hold`（TTS 播报中），也只改 `intent`，**不立刻打断 TTS**；本次 hold 释放后按新意图走（避免「用户让暂停 → 协调器立刻 pause → 与正在播的 TTS 抢声音」的乱序）。
- **hold API（Voice Service，本地 IPC，JSON lines over Unix socket）**：`hold(reason="tts")` / `release(reason="tts")`，计数式，往返目标 <1ms。Voice Service 不再直连 MPD（§2 职责行改为「调协调器 hold/release」）。
- **初始 intent**：协调器启动时读 MPD 实际状态作为基线（或从上次记忆恢复）。
- **可选**：状态变化事件/查询接口，供调试与未来可视化。

由此现状的 `_previous_status`/`set_expected_status` 全局变量即被正规化为协调器内的 `intent` 寄存器，并跨进程可访问。

### 10.4 与拆分/工具披露的关系

- `voice_agent/mpd_tool.py` 的 registry 注册 → 改为 MCP server 暴露（handler 逻辑吸收进协调器、schema 原样复用为 MCP 工具定义）；`hermes_gateway_plugin` 的 `register()` 不再注册 mpd 工具（§5.2 已同步）；
- `platform_toolsets.voice` 条件披露机制**不覆盖 MCP 工具**——MCP 化后工具的可见性取决于 hermes 的 MCP 接入粒度（是否可按平台/会话过滤），列为实施期检查项（§10.6-1）；
- `music_control.py` 的 playerctl/MPRIS 路径确认可退役（真机核对 MPD 是否完全走 6600 协议后），协调器统一用 `python-mpd2` 直连，列为实施期检查项（§10.6-2）。

### 10.5 定稿：进程形态与代码位置（方案 A）

- **协调器独立常驻进程**（已定）：`python -m music_coordinator`，独立 systemd 服务（§6.2）。理由：不依赖音频栈、不与语音故障连坐、可被 hermes 与 Voice Service 两个进程独立访问。备选方案（协调器并入 gateway / 并入 Voice Service / 协调器即 MCP server 并让 Voice Service 当第二个 MCP client）均评估后否决。
- 代码落点：仓库根 `music_coordinator/`（独立包，见 §4.1 目录树）：
  ```
  music_coordinator/
  ├── __main__.py          # python -m music_coordinator
  ├── coordinator.py       # intent 状态机 + effective 规则 + 唯一 MPD 写路径
  ├── mpd_conn.py          # python-mpd2 共享连接管理（吸收旧 mpd_tool._connect）
  ├── tools_mcp.py         # MCP server 面：12 个 mpd_* 工具（schema 复用旧 mpd_tool.SCHEMAS；
  │                        #   handler 改为「更新 intent → 按 effective 应用」）
  ├── hold_ipc.py          # Unix socket 服务：hold/release（计数式、JSON lines）
  └── README               # MPD_HOST/MPD_PORT 等配置说明
  ```
- hermes 侧（`hermes_gateway_plugin`）：配置 MCP 接入（server URL/命令），删除 `register_mpd_tools()`；
- Voice Service 侧（`voice_service/playback.py`）：内置 hold/release 客户端（Unix socket，往返 <1ms），不再 import `music_control`/直连 MPD。

### 10.6 实施期检查项（不阻塞设计）

1. **MCP 工具披露粒度**（M4 验证）：hermes 的 MCP 接入是否支持按平台/会话过滤工具；若支持，则 mpd 工具集可维持"仅语音平台可见"，否则全平台开放（接受——语音场景之外 agent 本也可用 mpd 工具）。
2. **playerctl/MPRIS 退役**（真机核对）：确认现 MPD 播放完全走 6600 协议后，删除 `music_control.py` 的 playerctl 路径。
3. **只读工具语义**：`mpd_get_status` / `mpd_get_current_song` / `mpd_get_playlist` / `mpd_search` 经 MCP 照常提供，**不更新 intent**。
4. 已同步修订的正文：§2 职责表 mpd 行与表后注、§4.1 目录结构（仓库根三子项目布局，代码骨架已落地）、§4.4 `music_coordinator:` 配置注释、§4.5 装配表（music_control → hold/release）、§5.2（mpd 注册移除）、§6.2（新增 coordinator 服务 bullet）。
