# 家庭语音助手（Hermes Agent 语音前端）

基于 [Hermes Agent](https://github.com/NousResearch/hermes-agent) 框架的家庭语音助手，为 AI 代理增加完整的语音交互能力：**唤醒词唤醒 → 语音识别 → AI 对话 → 语音合成**。

## 项目构成

语音能力运行在**独立服务进程**（Voice Service）中，与 Hermes Agent 网关经 WebSocket 通信——语音链路故障不影响网关。仓库由三个相互独立的 Python 子项目构成：

```
麦克风/扬声器 ─► voice_service（语音服务，WS 服务端）
                     ▲        │
                     │  WS    ▼  语音识别结果 / 待朗读文本
        hermes_gateway_plugin（hermes voice 平台，WS 客户端）──► Hermes Agent 会话
                     ▲
                     │ MCP（意图）   音乐状态协调器（music_coordinator）──► MPD
```

- **[`voice_service/`](voice_service/)**：独立语音服务进程，持有麦克风/扬声器、sherpa-onnx 唤醒词、WebRTC VAD、腾讯云 ASR/TTS、播放队列与连续对话窗口；对外暴露 WS 接入点。**零 hermes 依赖**。唤醒词 ↔ 助手映射的唯一源也在这里（`config.yaml` 的 `wake_word.assistants`）。
- **[`hermes_gateway_plugin/`](hermes_gateway_plugin/)**：hermes 的 voice 平台插件，作为 Voice Service 的 **WS 客户端**，把语音识别结果变成 gateway 会话消息（`chat_id=wake:<助手名>`）、把 LLM 回复文本下发朗读。**唯一与 hermes 绑定的子包**。
- **[`music_coordinator/`](music_coordinator/)**：MPD 的唯一写入口（intent 意图 + hold 避让状态机），经 **MCP（streamable HTTP）** 向 Agent 暴露 `mpd_*` 音乐控制工具。零 hermes 依赖。

架构、协议与部署设计详见 [`docs/voice-service-websocket-architecture.md`](docs/voice-service-websocket-architecture.md)。

## 当前功能

### 功能特点

- **全链路语音交互**：sherpa-onnx 唤醒词 → WebRTC VAD → 腾讯云实时 ASR → Hermes Agent → 腾讯云流式 TTS；本地唤醒，云端识别与合成，全程低延迟
- **多助手 / 多唤醒词**：多个唤醒词（泡泡、小布…）各自唤起独立助手会话（`chat_id=wake:<助手名>`，会话历史隔离）；唤醒词↔助手映射在 Voice Service 配置中统一维护
- **说话人识别**：腾讯云说话人分离引擎（句子级 speaker_id）+ 本地 CAM++ 声纹特征库，识别每句话是谁说的；新说话人自动注册（spk_100 起），上行给 agent 的文本按句带 `[名字 (ID: 编号)]` 前缀（如 `[爸爸 (ID: 100)] 打开客厅灯`）
- **说话人身份绑定**：agent 可通过 `voice_speaker_bind` 工具把声纹编号记成名字（用户说"我是辰辰"→ 之后就是 `[辰辰 (ID: 101)]`，认错人时带 `overwrite` 更正）；绑定写回语音服务并即时生效
- **对话历史日志**：语音服务把每轮对话写成按天的 JSONL（时间/说话人/助手/完整内容），方便人工核对并维护说话人身份
- **多段情绪语音**：大模型可在一条回复中使用多个 `(情绪)` 标记，各段按对应情绪流式合成播放
- **随时待命提示**：等待 LLM 回复期间循环播放柔和提示音，让用户知道当前不再收听
- **AI 回复自动播报**：Agent 每轮回复（工具轮的阶段性文字与最终回答）由 gateway 的 `post_api_request` 钩子实时捕获、按 `(情绪)` 分段下发语音服务流式合成播放——无需模型调用任何播报工具
- **唤醒体验优化**：唤醒词后 VAD 静音保护期，避免提示音回声导致 ASR 过早结束
- **音乐播报避让**：每段 TTS 播报前经 Music Coordinator 暂停音乐、播完恢复，不打扰听歌

## 未来设想

当前助手是"唤醒后应答"模式。下一步希望它成为一个**随时在听、主动参与**的家庭成员：

### 1. 全天候记录家庭对话

- 通过**本地 ASR 模型**持续识别家庭成员的对话（音频不上云，保护隐私）
- 同时用 **CAM++ 声纹识别**标注每句话的说话人

### 2. 两级思考，按需唤醒

- 用**本地零样本分类模型**（zero-shot text classification，非 LLM）对对话做浅层思考：判断对话内容是否提到自己、是否需要回应
  - 选择分类模型而非本地 LLM：浅层判断只需轻量的相关性分类，分类模型常驻成本低、延迟可控，且同样全程本地推理、不上云
- 与自己无关 → 保持静默，继续监听
- 与自己有关 → 唤醒**远程大规模 LLM** 深度思考：参与对话、调用工具解决问题

### 3. 掌握上下文

- 必要时自行决策，从**数据库调取家庭对话历史**，以便掌握问题的完整上下文

### 4. 异步结果送达

- 执行长时间任务后，提问者可能已经不在附近
- 大模型调用工具**记录触发器**（提问者身份）
- 当系统再次通过 **CAM++ 识别到该提问者在家中说话**时，主动插话播报执行结果
