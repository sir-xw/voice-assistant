# 交互要求
- 思考（Thinking）过程使用中文
- 回复也要使用中文

# AGENTS.md

## 仓库结构

语音能力已从旧的单体内嵌实现（`hermes-voice-agent/`）拆分为**三个平级、自包含的子项目**
（各有独立 `pyproject.toml`，可分别安装到不同 venv）：

| 目录 | 角色 | 依赖 hermes |
|---|---|---|
| `voice_service/` | 独立语音服务（WS **服务端**）：麦克风 / 唤醒词 / VAD / 腾讯云 ASR-TTS / 播放队列 / 连续对话窗口；**唤醒词↔助手映射的唯一源** | ❌ |
| `hermes_gateway_plugin/` | hermes gateway 的 voice 平台插件（WS **客户端**）：识别结果 → 会话；回复文本 → 下发朗读 | ✅（唯一） |
| `music_coordinator/` | MPD 唯一写入口（intent/hold 状态机）：hold IPC 供 Voice Service 播报避让，MCP(web) 供 agent 音乐工具 | ❌ |

- 其他目录：`docs/`（架构设计：`voice-service-websocket-architecture.md`）、
  `deploy/systemd/`（hermes-gateway 单元模板）。旧单体实现 `hermes-voice-agent/`
  已于 2026-09 删除：可复用的组件/工具已迁入 `voice_service/`，历史见 git log。
- 接入协议见 `voice_service/PROTOCOL.md`；协议常量以 `voice_service/voice_service/protocol.py`
  为准（两端共用、零第三方依赖，改动需同步 `hermes_gateway_plugin`）。
- 注释、文档、代码字符串均为中文 —— 请保持中文。
- **USB 音频设备运维属本机环境特定内容，已移出仓库**（`/root/usb-audio/`：修复脚本、健康
  检测、音量服务与其 systemd 单元，见该目录 README）。不要把它们加回本仓库。

## 运行环境（关键）

- **仓库内没有 venv。** 依赖 hermes 自带 venv：`/usr/local/lib/hermes-agent/venv/bin/python3`
  （Python 3.11）；`pip install` 也装到那里。
- hermes-agent 源码在 `/root/git/hermes-agent`（已 editable 安装）—— **不要修改**；本仓库只提供
  语音侧代码与 gateway 插件装配。
- 三个子包均为可编辑安装（改代码即生效）：
  `pip install -e ./voice_service -e ./hermes_gateway_plugin -e "./music_coordinator[mcp]"`
  （`hermes_gateway_plugin` 依赖 `voice_service` 的 `protocol.py`，须同 venv）。
- 腾讯云语音 SDK **vendored** 在 `voice_service/voice_service/tencentcloud_speech/`，不是 pip 依赖。
- 运行 Voice Service：在 `voice_service/` 目录下 `python -u -m voice_service --config config.yaml
  [--audio] [--out]`。**cwd 很关键**：模型与 KWS 词表按 `voice_service/` 相对路径解析
  （`models/sherpa-kws/<model_name>/`）。
- 部署为 systemd **用户服务**（root + `XDG_RUNTIME_DIR=/run/user/0`）：
  `voice-service.service`（`--audio --out --config config.yaml`）、`music-coordinator.service`
  （`--enable-mcp`）、`hermes-gateway.service`（hermes 自身）。单元定义见各子项目 README
  （仓库 `deploy/systemd/` 只含 hermes-gateway 模板）。
  改代码后 `systemctl --user restart voice-service`；日志 `journalctl --user -u voice-service -f`
  （gateway 另有 `~/.hermes/logs/{gateway,agent}.log`）。
- Voice Service 先于/后于 gateway 启动都可以：gateway 只是 WS 客户端，断线自动重连；
  **重启 voice-service 不需要重启 gateway**。

## 配置

- Voice Service 自身配置：`voice_service/config.yaml`（**gitignore，不入库**；模板
  `config.yaml.example` 入库，部署时 `cp config.yaml.example config.yaml`）；凭据
  `voice_service/.env`（`VOICE_SecretId/Key/AppId`，gitignore）；模型 `voice_service/models/`
  （gitignore）。
- hermes 侧只配连接参数：`~/.hermes/config.yaml` 的 `platforms.voice.extra.service`
  （`url`/`token`）。**不再配置 wakewords**；如需按助手覆盖 prompt/model，用 hermes 原生
  `channel_overrides`（键 `wake:<助手名>`）。
- 音乐 MCP：`~/.hermes/config.yaml` 的 `mcp_servers.music`（`url: http://127.0.0.1:8766/mcp`）。
- gitignore 覆盖 `*.log`/`*.wav`/`.env`/`models/`/`__pycache__`/`.venvs/` 等；包内提示音
  `voice_service/voice_service/assets/*.wav` 需入库（已有豁免规则）。

## 唤醒词与助手

- 唤醒词↔助手映射的**唯一源**：`voice_service/config.yaml` 的 `wake_word.assistants`
  （`name` = 助手名，`keywords` = 触发短语）。Voice Service 启动时由 `kws_words.py`
  生成/校验 `models/sherpa-kws/<model>/keywords.txt`；命中后上行事件的 `wake`（KWS `@` 后缀）
  即助手名，gateway 用它拼 `chat_id = wake:<助手名>`。
- 修改后重启 `voice-service` 即生效；hermes 侧无需改动（`hello` 不再上报词表，
  `welcome.my_wakewords` 回显服务端实际助手表）。
- 旧 `hermes-voice-agent/tools/gen_keywords.py` 已退役，勿再使用。

## AI 输出契约

- 服务端按 `(情绪)文字` 解析分段并流式合成播放；情绪集合与 `voice_service/protocol.py` 的
  `VALID_EMOTIONS`、`parse_emotion_segments` 必须一致，勿孤立修改。
- **没有 `speak` 工具**：模型正常返回文本即可。gateway 插件用 `post_api_request` 钩子捕获每轮
  回复 → `speak` 帧（`finish_reason=stop` → `final`，其余 → `interim`），`send()` 仅兜底。
- 播后行为：`final` → 通知音 + 进入连续对话窗口；`interim` → 回到等待（不通知音、不进窗口）。
  「不需要朗读」时客户端发 `control{close_window}`（hermes 插件里 `[FINISH]` 即此语义）。
- **被打断会话的回复**：用户在助手 A 思考/调工具期间唤醒 B 时，插件按 `session_id` 惰性归属回复
  （`_wake_for_hook`），保证 A 的迟到回复仍标记为 A；服务端据此**丢弃 A 的 interim**、把 A 的
  final **排队并加「我是A，」前缀**。改动归属/播报语义时需同时改插件与 playback，并同步
  `voice_service/PROTOCOL.md`。
- `post_api_request` 是 hermes **全局 observer 钩子**，必须按 `platform == "voice"` 过滤来源
  （已实现，勿移除），否则 CLI 等会话的回复也会被朗读。

## 语音输入契约（说话人前缀）

- voiceprint 启用时，**Voice Service** 在 `asr_result.text` 里按句拼好
  `[说话人名字 (ID: 编号)] 内容`（多句换行），例如 `[爸爸 (ID: 100)] 打开客厅灯`、
  `[未知 (ID: 101)] 你好`；是否加前缀与 `voiceprint.enabled` 绑定，格式与规则见
  `voice_service/PROTOCOL.md` §5。
- 名字取 `voiceprint.speaker_names`（`spk_100` → 真实姓名），未映射/未识别显示「未知」；
  编号为声纹库 id 的数字部分（`spk_100` → `100`），同一人跨轮次稳定 —— agent 据此区分说话人。
- 拼接只在服务端（`voiceprint.speaker_label` + `inbound.py`）完成；gateway 插件与 agent
  侧**原样透传、不解析不重排**。改动格式需同步插件 `platform_hint` 与 `PROTOCOL.md`。

## 测试

未配置 pytest / linter / typecheck，均为手工脚本：

- 协议自检：`python -m voice_service --selfcheck`（等价 `protocol._selftest()`）
- 音乐协调器：`python -m music_coordinator --dummy-mpd`（Dummy MPD，避免影响真实 MPD）；
  `python -u music_coordinator/tests/mcp_smoke.py`
- 语音调试工具（`voice_service/tests/`，在 `voice_service/` 下运行）：
  `test_sherpa_asr.py`、`test_sensevoice_asr.py`（本地 ASR）、`test_sherpa_kws.py`、
  `test_sherpa_tts.py`、`test_tencent_asr.py`（云端 ASR）、`test_speaker_identify.py`、
  `test_voiceprint_live.py`（说话人/声纹；`tests/4spk.wav` 用于对比模型分离效果）
- 语音工具（`voice_service/tools/`）：`download_models.py`（按配置下载模型到 `models/`）、
  `tts_gen.py`（腾讯云 TTS 生成提示音资产）
- 握手/联调：临时起 `voice_service`（覆盖端口、不带 `--audio`）用 WS 客户端验证
  `hello/welcome`、`speak/ack`；参考 `voice_service/PROTOCOL.md` 的最小示例

## 提交规范（约定式提交）

格式：`<type>(<scope>): <中文简述>`（首行 ≤ 72 字符，末尾不加句号）；空行后用中文正文说明
**为什么改**与影响、取舍。

- **type**：`feat` 新功能 / `fix` 修缺陷 / `refactor` 重构（行为不变）/ `docs` 文档 /
  `test` 测试 / `chore` 杂项（构建、依赖、配置）/ `perf` 性能。
- **scope**：受影响对象，如 `voice-service`、`voice`（hermes 插件）、`music`、`protocol`、
  `docs`、`deploy`。
- **粒度**：一个提交只做一件事；按内容拆分为多个独立提交；三个子包的**同一主题**改动可合并为
  一次提交（例：三子包拆分）。
- **兼容性**：改动协议/配置格式等破坏性变更时，在正文显式标注。
- **不要提交**：`voice_service/config.yaml`、`.env`、`models/`、日志与运行产物、本机绝对路径类
  配置（模板/示例除外）。
- 示例：
  - `feat(voice-service): 恢复 speak kind 区分——interim 播完回到等待`
  - `fix(voice): post_api_request 钩子按 platform=voice 过滤来源`
  - `docs(voice-service): 新增面向其他 agent 的 WebSocket 协议说明`
  - `refactor: 新架构文件归位（文档→docs/、gateway 单元→deploy/）`

## 当前状态

- 新架构已部署运行：`voice-service` / `music-coordinator` / `hermes-gateway` 均为 systemd
  用户服务，语音全链路（唤醒 → ASR → agent → TTS 播报 → 连续对话 → 音乐避让）可用。
- 迁移里程碑 M0–M4 已完成；M1 的 hub/session 多连接仍为 v1 单客户端（既定取舍）。
