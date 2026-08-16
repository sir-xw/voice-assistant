# 语音交互路径迁移为 Hermes Gateway 平台插件 — 设计方案

> 状态：设计稿（待确认后实施）
> 范围：把 `hermes-voice-agent` 的完整语音交互路径（唤醒词 → VAD → 腾讯云 ASR → Hermes Agent → 腾讯云 TTS → 播放）迁移为 hermes gateway 的**平台插件**，由 gateway 统一管理 agent 会话。
> 重点评估：现有 VoiceApp 的**代际（gen）**与**待汇报（pending）**机制在 gateway 模式下是否受影响、如何映射。

---

## 1. 目标与范围

### 1.1 目标

- 新增一个 gateway 平台插件（暂名 **`voice-platform`**），让语音成为 hermes gateway 的一个平台：
  - **inbound**：唤醒词（sherpa KWS）→ VAD → 腾讯云 ASR 文本 → 作为用户消息进入 gateway 会话
  - **outbound**：agent 回复 → 适配器 `send()` → 腾讯云 TTS 流式播放（含 `(情绪)文字` 分段、通知音、对话窗口）
  - **中间轮播报**：插件自带 `post_api_request` 钩子，工具调用轮的文字回复直接播报（替代已废除的 speak 工具）
- 每个唤醒词映射为 gateway 的**独立会话**（用户已确认）。
- 部署从 `python -m voice_agent`（systemd `hermes-voice-agent.service`）改为 `hermes gateway` + `platforms.voice` 配置。

### 1.2 不做的事

- 不修改 hermes-agent 源码（`/root/git/hermes-agent/`，AGENTS.md 约定）。
- 不依赖 hermes 自带的语音栈（`tools.voice_mode`/`hermes_cli/voice.py` 用的是 faster-whisper + 系统 TTS）；本项目继续使用**腾讯云 ASR/TTS + sherpa 唤醒词**（vendored SDK 与组件直接复用）。
- 不保留 VoiceApp 的 agent 管理代码（`_agents` 线程池、`run_conversation`、代际计数器等）——由 gateway 取代。

---

## 2. 现状 vs 目标架构

### 2.1 现状（VoiceApp 嵌入运行）

```
麦克风 → [sherpa KWS 唤醒] → [VAD] → 腾讯云 ASR → 文本
   → VoiceApp._ask_agent() → 线程池 → AIAgent.run_conversation()   ← 自建 agent 管理
   → post_api_request 钩子（speech-relay 插件）→ 播放队列
   → 腾讯云 TTS → 扬声器（中间轮播报 / 最终轮：通知音 + 对话窗口）
自建：多 agent（每唤醒词一个 AIAgent + ThreadPoolExecutor）、代际计数器（打断失效）、
待汇报队列（被取代的 agent 结果排队递送）、连续对话窗口状态机。
```

### 2.2 目标（gateway 平台插件）

```
麦克风 → [sherpa KWS 唤醒] → [VAD] → 腾讯云 ASR → 文本
   → MessageEvent(chat_id="wake:小布") → gateway 会话路由 → agent（gateway 管理）
   → agent 回复 → gateway delivery → 适配器 send() → 腾讯云 TTS → 扬声器
   → post_api_request 钩子（voice-platform 自带）→ 中间轮播报
gateway 提供：会话/历史/上下文、turn 代际（run_generation）、忙时打断/排队
（busy_input_mode）、记忆/cron/多平台并存等。
适配器保留：唤醒词/VAD/ASR/TTS/播放、情绪分段、对话窗口、播报串行化。
```

### 2.3 gateway 平台插件机制（调研结论，均已核对源码）

| 机制 | 位置 | 说明 |
|---|---|---|
| 插件形态 | `plugins/platforms/<name>/`：`plugin.yaml`(kind: platform) + `__init__.py`(导出 register) + `adapter.py` | 参照 `ntfy-platform` |
| 注册 | `ctx.register_platform(name, label, adapter_factory, check_fn, ...)` | `hermes_cli/plugins.py:2774` |
| 钩子 | `ctx.register_hook("post_api_request", fn)` | `hermes_cli/plugins.py:3109`（同一 ctx，平台插件可用） |
| inbound | 构造 `MessageEvent(text, source=...)` → `await self.handle_message(event)` | `gateway/platforms/base.py` |
| outbound | agent 回复经 `delivery.py` → `adapter.send(chat_id, content, metadata)` | `gateway/delivery.py:89` |
| 生命周期 | `async connect(is_reconnect)` / `async disconnect()` | `base.py:3870/3890` |
| 忙时行为 | `busy_input_mode`: `interrupt`(默认) / `queue` / `steer` | `gateway/run.py:6366` |
| 排队槽 | `adapter._pending_messages[session_key]`（单槽 + /queue FIFO） | `base.py:3049` |
| turn 代际 | `run_generation` + `is_still_current`（新 turn 取代旧 turn 检测） | `gateway/run.py` |
| 配置 | `~/.hermes/config.yaml` → `platforms.<name>: {enabled, extra}` | 参照 cli-config.yaml.example |
| 启动 | `hermes gateway`（加载所有 enabled 平台） | `gateway/run.py` |
| 语音基建 | `voice.auto_tts`、流式 TTS（`build_auto_tts_output_path` 等） | 本项目不用，仅提示 |

---

## 3. 目标设计：`voice-platform` 插件

### 3.1 目录结构（项目内）

```
hermes-voice-agent/
└── .hermes/plugins/voice-platform/
    ├── plugin.yaml          # kind: platform, name: voice-platform, label: 语音
    ├── __init__.py          # 导出 register
    └── adapter.py           # VoiceAdapter(BasePlatformAdapter) + register(ctx)
```

插件代码可 `from voice_agent.asr_engine import ...` 复用现有组件（`voice_agent` 是 editable 安装的包，项目内可直接 import）。

### 3.2 plugin.yaml 要点

```yaml
name: voice-platform
kind: platform
version: 1.0.0
label: 语音
description: 语音平台：sherpa 唤醒词 + 腾讯云 ASR/TTS，接入 hermes gateway
requires_env:
  - name: VOICE_SecretId        # 腾讯云凭据（与现有 config.yaml/.env 一致）
  - name: VOICE_SecretKey
  - name: VOICE_AppId
```

### 3.3 register(ctx) 要点

```python
def register(ctx):
    ctx.register_platform(
        name="voice",
        label="语音",
        adapter_factory=lambda cfg: VoiceAdapter(cfg),
        check_fn=check_requirements,        # 校验腾讯云凭据/sherpa 模型存在
        validate_config=validate_config,
        required_env=["VOICE_SecretId", "VOICE_SecretKey", "VOICE_AppId"],
        install_hint="pip install -e .  # hermes-voice-agent",
        emoji="🎙️",
        platform_hint="你通过语音与用户交流……（情绪标记/简洁规则，复用现有提示词）",
    )
    # 中间轮播报：插件自带 post_api_request 钩子（替代 speech-relay）
    ctx.register_hook("post_api_request", on_post_api_request)
```

### 3.4 VoiceAdapter 设计

```
class VoiceAdapter(BasePlatformAdapter):
    # —— 生命周期 ——
    async def connect(self, *, is_reconnect=False):   # 启动麦克风/唤醒词/ASR/TTS/播放器
    async def disconnect(self):                        # 停止全部硬件

    # —— inbound：语音 → gateway 会话 ——
    # 唤醒词命中 / 对话窗口期内 VAD 检测到语音
    #   → 录音 → 腾讯云 ASR → on_final 文本
    #   → MessageEvent(text, source=self.build_source(
    #         chat_id=f"wake:{唤醒词}", chat_name=唤醒词, ...))
    #   → await self.handle_message(event)

    # —— outbound：gateway 回复 → 播放队列 → TTS ——
    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        # content 含 (情绪)文字 分段 → 解析 → 投入全局播报队列（串行）
        # 播完 → 通知音 → 进入对话窗口期（适配器内部状态机）
        # 播放放后台任务，send 快速返回（不阻塞 gateway 事件循环）

    # —— 播报串行化（取代 VoiceApp 待汇报队列的核心）——
    # 全局单播报队列 + 消费任务：所有会话的回复（含中间轮钩子）串行播放，
    # 同一时刻只有一个声音；可选在开头加身份前缀（如"我是小布，"）。
```

### 3.5 中间轮播报（post_api_request 钩子，插件自带）

```python
def on_post_api_request(**kwargs):
    # finish_reason == "tool_calls" 且 assistant_message.content 非空
    #   → 阶段文本 → 播报队列（无通知音、不进对话窗口）
    # finish_reason == "stop" → 忽略（最终回复由 gateway send() 播放，天然去重）
    # content == "[FINISH]" → 通知适配器关闭对话窗口（不播报）
    emit_to_playback_queue(kwargs)
```

- **去重原则**：中间轮只走钩子，最终轮只走 `send()`，两者由 `finish_reason` 严格区分，不会重复播报。
- 播放队列与 `send()` 共用，保证串行。

### 3.6 适配器内部状态机（保留自 VoiceApp）

- `IDLE`（等唤醒词）→ `LISTENING/RECORDING`（唤醒命中，VAD 检测）→ `PROCESSING`（ASR/agent）→ 播报 → `CONVERSATION_WINDOW`（窗口期，VAD 直接听，无需唤醒词）→ 超时回 `IDLE`。
- 物理打断：同会话新唤醒/新语音到达时 → `tts_engine.interrupt()` + `audio_player.clear()` + 清播报队列。

---

## 4. 关键语义映射（重点：代际 / 待汇报）

### 4.1 代际（generation）——✅ gateway 内建覆盖，不受影响

| VoiceApp（现状） | gateway 模式（目标） |
|---|---|
| 每 agent 一个 `gen` 计数器；唤醒/打断时 +1 | 每个 turn 的 `run_generation`，由 gateway 管理 |
| `run_conversation` 返回后对比基线 gen，不等则丢弃结果 | `is_still_current()` 闭包检测旧 turn 是否已被新 turn 取代，取代则丢弃/清理 |
| 同唤醒词再唤醒 → `agent.interrupt()` + gen+1 → 旧结果失效 | 同会话（同 `wake:小布` chat_id）新消息 → `busy_input_mode: interrupt` → gateway 取消进行中 turn |
| 播放队列条目带 gen，播放前校验 | 适配器播报队列由 gateway 的 turn 生命周期保证：被打断的 turn 的回复不再投递（send 不触发或内容作废） |

**结论**：代际语义由 gateway 的 turn 代际 + interrupt busy 模式完整覆盖，**适配器不需要自己实现 gen**。唯一要做的验证项：确认语音场景下"同会话新消息触发 interrupt"的路径（见 §8 验证计划）。

### 4.2 待汇报（pending）——⚠️ 语义转化，需要适配器层实现"播报串行化"

| VoiceApp（现状） | gateway 模式（目标） |
|---|---|
| agent A 被唤醒词 B 取代 → A 的结果进 `_pending` 队列 | **各唤醒词是独立会话**（不同 chat_id），gateway 按会话隔离：A 会话的 turn 不会被 B 会话打断，A 的回复照常送达 A 的 `send()` |
| 等 B 的对话窗口过期（state 回 idle）后递送 | gateway 无"对话窗口"概念；递送时机由适配器播报队列的消费节奏决定 |
| 递送格式：`"{user_title}，我是{keyword}，{回答}"` | 适配器在播报时可选加身份前缀（`我是{唤醒词}，`），解决"多个唤醒词共享一个扬声器时用户不知道谁在说话" |
| 单队列 FIFO | **全局播报队列（单消费者）**：所有会话的回复串行播放——这是"待汇报"在 gateway 模式下的等价物 |

**结论**：**"待汇报"从"结果排队等递送"转化为"播报串行化 + 可选身份前缀"**，由适配器的全局播放队列实现，gateway 本身无需感知。物理上同一扬声器同一时刻只能有一个声音，串行播报天然解决并发冲突；各会话的回复都不丢失（都进队列）。

### 4.3 其他语义映射

| 语义 | VoiceApp | gateway 模式 | 影响 |
|---|---|---|---|
| 情绪分段 `(情绪)文字` | `_parse_emotion_segments` | 适配器 `send()` 内解析（复用同一函数/常量） | 无 |
| 中间轮播报 | speech-relay 插件 + 桥 | 插件自带 `post_api_request` 钩子 | 无（speech-relay 可退役） |
| `[FINISH]` 关窗 | `_ask_agent`/回调识别 | 钩子 + `send()` 识别 `[FINISH]` → 关对话窗口 | 无 |
| 连续对话窗口 | VoiceApp 状态机 + `enter_conversation_window` | 适配器内部状态机（gateway 无感知） | 无 |
| 播放控制工具（mpd_*） | 模型返回 `[FINISH]` | 同左（工具照常注册在 gateway agent 的 registry） | 需确认 mpd 工具在 gateway 进程同样注册 |
| 说话人识别（voiceprint） | `_vp_round` 拼接 `[说话人] 内容` 发 LLM | 适配器 inbound 侧做同样拼接后进 `MessageEvent` | 无 |
| 等待提示音/资产音效 | `_play_asset`/`_wait_tone` | 适配器播放逻辑内保留 | 无 |
| 多 agent 提示词差异 | 每 agent 独立 `system_prompt` | 每会话通过 `platforms.voice.extra.wakewords` 映射 chat_id → 会话提示词（gateway 会话级系统提示词） | 需设计：gateway 每个会话可以带独立 system prompt（会话系统提示词机制） |

---

## 5. 多唤醒词 → 多会话

- 配置 `platforms.voice.extra.wakewords`：
  ```yaml
  platforms:
    voice:
      enabled: true
      extra:
        wakewords:
          小布: { chat_id: "wake:小布", session_id: "hermes-voice-小布" }
          翻译助手: { chat_id: "wake:翻译助手", session_id: "hermes-voice-翻译助手" }
        # 腾讯云凭据走 env（VOICE_SecretId 等，与现有一致）
        conversation_window_sec: 5.0
        voiceprint: { enabled: false, ... }
  ```
- 唤醒词命中 `小布` → `chat_id="wake:小布"` → gateway 独立会话（独立历史/上下文）。✓ 用户已确认此映射。
- **每个唤醒词独立的 system_prompt / model / provider**：用 gateway 原生机制
  `platforms.voice.channel_overrides`（`gateway/run.py::_get_channel_override`，
  按 chat_id 精确匹配）：
  ```yaml
  platforms:
    voice:
      channel_overrides:
        "wake:小布": { system_prompt: "你是家庭助手小布……" }
        "wake:翻译助手": { system_prompt: "你是翻译助手……", model: "openai/gpt-4o-mini" }
  ```
  （已用 `PlatformConfig.from_dict` 验证解析与查询。）
- 关键词文件：`models/sherpa-kws/<model>/keywords.txt` 继续由 `tools/gen_keywords.py` 生成（插件启动时校验存在，缺失则回退自动生成——复用现有逻辑）。

---

## 6. 组件复用清单（voice_agent → voice-platform）

| 组件 | 复用方式 |
|---|---|
| `voice_agent/voice_frontend.py`（KWS+VAD+麦克风+状态机） | 裁剪为适配器的 inbound 引擎（去掉 gateway 不需要的部分，如 `_ask_agent` 回调） |
| `voice_agent/asr_engine.py`（腾讯云 ASR） | 直接复用（`on_final` → 适配器构造 MessageEvent） |
| `voice_agent/tts_engine.py` + `audio_player.py` | 直接复用（流式 TTS → 播放器） |
| `voice_agent/main.py` 的情绪分段/资产音效/等待提示音/对话窗口逻辑 | 抽成适配器内部方法（或抽公共模块 `voice_agent/speech_output.py`） |
| `voice_agent/mpd_tool.py`、`tools/gen_keywords.py` | 不变（工具注册进 gateway registry；关键词生成脚本不变） |
| `voice_agent/speech_bridge.py` + `.hermes/plugins/speech-relay` | **可退役**（钩子移入 voice-platform 自带） |
| `voice_agent/plugin_hooks.py` | 仅 VoiceApp 模式需要；gateway 模式由 hermes 插件系统加载 |

---

## 7. 部署与配置变化

### 7.1 启动方式

- **现在**：`systemctl start hermes-voice-agent`（`python -m voice_agent`）
- **目标**：`hermes gateway`（可用 systemd 新 unit 或复用现有 unit 改 ExecStart）

### 7.2 插件加载（关键约束，需决策）

gateway 进程由 `hermes gateway` 启动，**没有 main.py 那样 patch 白名单的机会**。项目插件（`.hermes/plugins/voice-platform`）要被 gateway 加载，需满足：

1. `HERMES_ENABLE_PROJECT_PLUGINS=1`（systemd unit 的 `Environment=` 或 `hermes gateway` 启动环境）——仅当插件放项目目录；
2. `plugins.enabled` 白名单含 `voice-platform`——**写一次全局 `~/.hermes/config.yaml`**（`hermes plugins enable voice-platform`）。

可选方案：
- **方案 A（推荐）**：插件放 `~/.hermes/plugins/voice-platform/`（用户插件目录，默认扫描）+ `hermes plugins enable voice-platform`。全局配置只多一行 `plugins.enabled`，与 gateway 生态一致（hermes 官方所有平台插件都走全局配置）。
- **方案 B**：插件放项目 `.hermes/plugins/` + systemd unit 加 `Environment=HERMES_ENABLE_PROJECT_PLUGINS=1` + 仍需全局白名单一次。比 A 多一个环境变量、少一层"安装"语义。
- **方案 C**：接受"不改全局配置"原则的替代——用一个启动包装脚本（如 `tools/run_gateway.sh`）先 `export HERMES_ENABLE_PROJECT_PLUGINS=1` 并用 `python -c` 预 patch 白名单再 exec `hermes gateway`。**不建议**：gateway 重启/子进程场景容易失效。

> 注：这是与 VoiceApp 模式最大的部署差异——gateway 生态的插件启用统一走全局 `plugins.enabled`（`hermes plugins enable` 是官方 UX），无法保持"零全局配置"。

### 7.3 与现有部署的关系

- `hermes-voice-agent.service` 停用（或改造为启动 `hermes gateway`）。
- `hermes-voice-agent-volume.service`（USB 音量）可保留，与 gateway 无关。
- 腾讯云凭据（`VOICE_SecretId/SecretKey/AppId`）继续从 `.env`/环境注入（gateway 进程同样读取 `~/.hermes/.env` 与项目 `.env`——需在 unit 中配置 EnvironmentFile）。

---

## 8. 实施步骤

### 阶段 0：可行性验证（无头环境可做）✅ 已完成
- [x] 写最小 `VoiceAdapter` 骨架（connect/send/handle_message 空实现）注册为 platform，`hermes gateway` 能启动、`hermes plugins list`/`gateway status` 能看到 voice 平台（无需真实硬件，check_fn 放行）。
- [x] 确认同会话新消息 → `busy_input_mode: interrupt` 的打断路径（阅读/单测 gateway 行为）。

### 阶段 1：inbound（语音 → 会话）✅ 已完成
- [x] 复用 voice_frontend + asr_engine，唤醒 → ASR → `MessageEvent` → `handle_message`。
- [x] 对话窗口期 VAD 直接进 ASR（适配器状态机，复用 VoiceFrontend）。
- [ ] 说话人识别拼接（可选，voiceprint 未接入适配器 inbound——待真机需求确认）。

### 阶段 2：outbound（回复 → 播放）✅ 已完成
- [x] `send()`：情绪分段解析 → 全局播报队列 → TTS 流式播放 → 通知音 → 对话窗口。
- [x] `post_api_request` 钩子：中间轮播报 + `[FINISH]` 关窗。
- [x] 播报串行化 + 身份前缀（取代待汇报）。
- [x] 打断：物理打断（TTS/播放器）+ gateway interrupt 协同。

### 阶段 3：多唤醒词与会话映射 ✅ 配置层已完成
- [x] `wakewords` 配置 → chat_id → 会话绑定；per-session system_prompt/model
      用 gateway 原生 `channel_overrides` 实现（已验证解析）。
- [x] `keywords.txt` 生成/校验（复用 VoiceFrontend 现有逻辑）。

### 阶段 4：部署切换 📋 文档已就绪，待真机执行
- [x] 部署/使用文档：`docs/gateway-voice-plugin-usage.md`
      （pip 安装 + `hermes plugins enable voice-platform` + platforms.voice 配置 +
      `platform_toolsets.voice` + systemd unit 改造 + 真机冒烟清单）。
- [ ] 真机执行：停用 `hermes-voice-agent.service` → 启用 gateway 服务 → 冒烟。

---

## 9. 风险与验证

### 9.1 风险

| 风险 | 说明 | 缓解 |
|---|---|---|
| gateway 打断语义与语音直觉不符 | 语音"打断"= 同会话新消息；需确认 interrupt 模式下旧 turn 的 TTS 是否立即停 | 阶段 0 验证 + 适配器物理打断兜底 |
| 播报阻塞 gateway 事件循环 | `send()` 里做阻塞 TTS 会卡住其他平台 | 播放丢后台任务/线程，`send()` 快速返回 `SendResult` |
| 双平台并发（voice + telegram 同进程） | 语音播报与 telegram 消息互不干扰；但同一 agent 会话可能被两平台共用（不同 chat_id 则是不同会话） | 明确 chat_id 隔离策略 |
| 插件加载依赖全局配置 | 与"不改全局配置"原则冲突 | §7.2 三方案取舍，需用户拍板 |
| mpd/工具注册 | gateway 进程需注册 voice_agent 工具集 | 工具注册逻辑搬进插件 `register()` 或 gateway 侧工具配置 |
| 无头环境无法真机验证 | 麦克风/扬声器链路需真机 | 分阶段：逻辑/单测先行，真机冒烟清单 |

### 9.2 验证计划

- **无头环境（本会话可做）**：插件注册/加载、`hermes gateway` 启动、`gateway status` 显示 voice 平台、`post_api_request` 钩子触发链、`send()` 播报队列逻辑（打桩 TTS）。
- **真机（用户执行）**：唤醒→问答→播报全链路；打断；连续对话窗口；双唤醒词独立会话；`[FINISH]` 关窗；音乐播放控制；说话人识别（如启用）。

---

## 10. 待确认/开放项

1. **插件加载方案** ✅ 已定：作为正式插件，接受全局 `plugins.enabled`
   （`hermes plugins enable voice-platform`），项目开放给所有 hermes-agent 用户
   （entry-point 分发，pip 安装即注册）。
2. **mpd 工具注册** ✅ 已实现：register() 注册 mpd_* 到 `voice_agent` toolset，
   披露由 `platform_toolsets.voice: [voice_agent]` 控制（只对语音平台开放）。
3. **身份前缀** ✅ 已实现：`extra.identity_prefix` 可配置（默认 true）。
4. **speech-relay / plugin_hooks**：VoiceApp 模式保留（未删除）；gateway 模式
   不需要（钩子已移入 voice-platform 自带）。VoiceApp 整体下线后再清理。
5. **多唤醒词提示词差异** ✅ 已定：gateway 原生 `channel_overrides`
   （per-chat system_prompt/model/provider），已验证配置解析。
6. **说话人识别接入适配器**：voiceprint 未接入 gateway inbound（VoiceApp 有）。
   真机需求确认后再接（`_vp_round` 拼接逻辑搬入适配器 `_on_asr_complete`）。
