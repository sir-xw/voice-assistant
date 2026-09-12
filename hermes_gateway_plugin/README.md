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
# 预期：voice-platform 已注册（含工具 voice_speaker_bind / 工具集 voice_speaker）
#      → [voice client] 已连接 …（服务端助手表: [...]）→ ✓ voice connected
```

> - gateway 正常运行时若重启 voice-service，voice 平台会自动重连，无需操作 gateway。
> - Voice Service 未启动时 voice 平台会持续退避重连（日志见 warning），起服务后自动恢复。

## 说话人身份绑定工具（`voice_speaker_bind`）

用户在语音里说清身份（"我是辰辰"）后，agent 用这个工具把**当前说话人的声纹编号**绑定到名字；
之后该说话人的消息前缀就从「未知 (ID: 101)」变成「辰辰 (ID: 101)」，不必再问。

- 工具集名 `voice_speaker`，**必须列进 hermes 的 voice 平台工具集**，否则工具不会进入 agent
  的工具表（`platform_toolsets.voice` 是显式白名单）：

```yaml
platform_toolsets:
  voice:
    - browser
    - file
    - kanban
    - memory
    - session_search
    - skills
    - voice_speaker      # ← 本插件提供的工具集
    - web
```

- 参数：`name`（必填，用户说出的身份名）、`spk_id`（可选，消息前缀里 `ID:` 后面的数字；
  一轮里有多位说话人时必须指定）、`overwrite`（该编号已绑别的名字时，**先向用户确认**再置 true；
  用户说"你认错了，我是xx"属于这种情况）。
- 数据来源：`asr_result.data.speakers`（插件记下每个会话最近一轮的说话人与编号），**不解析
  `text` 前缀**。只对语音会话生效（`session_id → wake` 映射仅由 `platform=voice` 的钩子学习），
  CLI 等会话调用会被拒绝。
- 写路径：插件发 WS `speaker_alias` 帧 → Voice Service 校验并写
  `voice_service/models/voiceprint_lib/names.json` → **立即生效**（无需重启）。协议细节见
  `voice_service/PROTOCOL.md` §4.1。
- 名字与编号是**多对多**：声纹识别偏严，同一个人可能被注册成新编号，允许同一名字绑多个编号。
- 绑定结果只用于称呼与上下文，**不是身份认证**；敏感操作仍需按业务规则确认。

对应提示词规则（插件 `platform_hint`）：前缀已是具体名字 → 直接用、不再确认；前缀是「未知」
且任务需要身份 → 先问清再绑定；只有用户亲口说明才绑定，不要猜；认错人时用 `overwrite` 更正。

## 每助手模型 / 提示词（channel_overrides，待决策）

旧架构在 `hermes-voice-agent/config.yaml` 的 `agents[]` 里为每个唤醒词单独配置
`model` / `provider` / `api_mode` / `base_url` / `max_iterations` / `session_id` /
`system_prompt`。新架构把「每助手差异」交给 hermes 原生 `channel_overrides`
（按 `chat_id = wake:<助手名>` 命中），**当前部署未配置**（泡泡/小布共用默认模型与
提示词），需要时按下面方式补：

```yaml
platforms:
  voice:
    channel_overrides:
      "wake:小布":
        model: "deepseek-v4-flash"        # 对应旧 agents[].model
        provider: "deepseek"              # 对应旧 agents[].provider
        system_prompt: "本轮会话中，你不能执行任何会影响系统稳定运行或改变系统关键设置的指令。"
```

已知差异（后续决策）：

- `ChannelOverride` 仅支持 `model` / `provider` / `system_prompt`（见 hermes
  `gateway/config.py`）；`api_mode` / `base_url` 由 provider 与凭据自动路由，
  `max_iterations` 无 per-channel 项（全局设置）。
- 旧的 `session_id` 无需迁移：每个助手名对应独立 `chat_id`，会话历史天然隔离。

### 回复归属（session_id → 助手）与"被打断会话"

`post_api_request` 钩子只带 `session_id`。适配器用**惰性映射** `_wake_for_hook()`：某会话
首次出现时，当前活跃助手即其归属方，之后该会话的中间轮/最终回复都按它标记 `speak.wake`
（`_session_wake`）。这样用户在助手 A 思考/调工具期间唤醒助手 B 时，A 迟到的回复仍标记为 A，
不会因为全局 `_current_wake` 变成 B 而被错记。

Voice Service 据此处理被打断会话（详见 `voice_service/PROTOCOL.md` §6）：

- A 的 `interim` → **丢弃**（不打扰正在进行的 B 对话）；
- A 的 `final` → **排队**到当前播放结束后播出，并在文本前加 `我是A，` 前缀。

> 注意：前缀只加给"被其他对话打断"的最终回复，正常回复不加。旧 VoiceApp 的
> `{user_title}，我是{助手名}，…`（每会话首条都加）已不再沿用。

## 实现进度

- [x] `client.py`：可独立运行的 WS 客户端（hello/welcome、ping、线程安全 send/request）
- [x] `adapter.py`：接入 hermes gateway 平台（WS 客户端 + `post_api_request` 钩子）
- [x] `voice_speaker_bind` 工具（工具集 `voice_speaker`）：身份绑定 → `speaker_alias` 帧
- [x] entry point（voice-platform）；旧 `voice_agent.gateway_plugin` 已退役

## 测试

无 pytest，用可执行脚本（在 hermes venv 中运行，不需要 gateway/语音设备）：

```bash
/usr/local/lib/hermes-agent/venv/bin/python -u hermes_gateway_plugin/tests/test_speaker_bind.py
```

覆盖：非语音会话/记录过旧/多说话人/编号不属于本轮等拒绝分支，正常帧内容与 ack 处理，
冲突时的 overwrite 提示，以及工具注册（工具集名、schema、check_fn）。
