# 交互要求
- 思考（Thinking）过程使用中文
- 回复也要使用中文

# AGENTS.md

## 仓库结构

- 真正的项目代码位于 `hermes-voice-agent/` 目录下。仓库根目录的文件（`API-DOCS/`、`*.wav`、`hermes-voice-agent-deploy.txt`、根目录 `.env`）是部署笔记/产物，不是应用代码。请在 `hermes-voice-agent/` 内工作。
- 这是 **Hermes Agent 框架的语音前端**：唤醒词 → VAD → 腾讯云 ASR → Hermes Agent → 腾讯云 TTS。
- 注释、文档、代码字符串均为中文 —— 请保持中文。

## 运行环境（关键）

- **仓库内没有 venv。** 所有运行都依赖 Hermes Agent 安装自带的虚拟环境：
  `/usr/local/lib/hermes-agent/venv/bin/python3`（Python 3.11）。`pip install` 也装到那里。
- `AIAgent`（`from run_agent import AIAgent`）和 `from tools.registry import registry` **都从 `/usr/local/lib/hermes-agent/` 导入，不在本仓库内**。本仓库只提供工具处理函数和应用装配。不要试图在仓库内改动它们。
- 腾讯云语音 SDK 是 **vendored 在 `voice_agent/tencentcloud_speech/`** 的 —— 直接改那里的代码；它不是 pip 依赖。
- 在 `hermes-voice-agent/` 目录下运行：`python -u -m voice_agent`（入口在 `voice_agent/__main__.py`）。**cwd 很关键**：sherpa 模型和 `keywords.txt` 都按项目根目录的相对路径解析。
- **`voice_agent` 是可编辑安装的包**：首次使用需 `python -m pip install -e .`（`pyproject.toml`，装进 hermes venv）。之后 `from voice_agent.xxx import ...` 全项目可用（vendored SDK 写作 `from voice_agent.tencentcloud_speech...`），**任何脚本都无需 sys.path hack**。改代码即生效，无需重装；`run_agent` / `tools.registry` 来自已安装的 hermes-agent，保持顶层导入。
- 通过 systemd 部署：`services/hermes-voice-agent.service`（用户 `root`，`WorkingDirectory=/root/git/voice-assistant/hermes-voice-agent`）。改代码后：`systemctl restart hermes-voice-agent`；日志：`journalctl -u hermes-voice-agent -f`。还有一个配套的 `hermes-voice-agent-volume.service` 负责 USB 音量。

## 配置

- `config.yaml` + `.env`（腾讯云凭据 `VOICE_SecretId`、`VOICE_SecretKey`、`VOICE_AppId`；已 gitignore）。优先级：环境变量 > yaml > 默认值（见 `voice_agent/config.py`；环境变量落入 `tencent` 键下）。
- `models/`、`.env`、`*.log`、`*.wav`、`*.pyc` 已被 gitignore —— 全新克隆需要先有 KWS 模型和 `.env` 才能跑起来。

## 多 Agent 与唤醒词

- `config.yaml` 中每个 `agents[].name` 都是一个唤醒词；它必须与 `models/sherpa-kws/<model>/raw_keywords.txt` 里的 `@original` 标记一致。修改唤醒词后执行 `python -u tools/gen_keywords.py` 重新生成 `keywords.txt`。
- 每个 agent 可配置 `model`（如 `deepseek-v4-flash`）与 `provider`（如 `deepseek`，否则 hermes 日志显示 provider=unknown；base_url 仍会按凭据自动路由）。`api_mode` / `base_url` 直接写入本仓库 `config.yaml` 的 `agents[]`（如 `codex_responses` + `https://api.deepseek.com`）。
- 每个 Agent 拥有独立的 `ThreadPoolExecutor`、`session_id` 和代际计数器，代际用于打断时使进行中的结果失效（见 `voice_agent/main.py` 的 `_on_wake_word`、`_ask_agent`）。

## AI 输出契约（由 `voice_agent/main.py` 中的系统提示词强制）

- 阶段性进展只能用 `speak` 工具播报。最终回答必须以 `(情绪)文字` 格式返回，情绪取自固定集合（neutral/sad/happy/angry/.../jieshuo）。**一段回复可以包含多个情绪标记分段**：各段按对应情绪**流式实时合成播放**，前一段播完自然续播后一段，全部播完后才播通知音并进入对话窗口。播放控制类工具（`mpd_*`）直接返回字面量 `[FINISH]`。
- 不要孤立地改动这个契约 —— `main.py`（`_parse_emotion_segments` 的情绪集合）与 `speak` 的 schema 必须保持一致。

## 测试

- 未配置 pytest / linter / typecheck。测试是手工脚本，从 `hermes-voice-agent/` 目录直接运行（`voice_agent` 已 `pip install -e .`，无需 sys.path hack）：
  - `python -u tests/test_sherpa_asr.py --help`
  - `python -u tests/test_sensevoice_asr.py --help`
  - `python -u tests/test_tencent_asr.py --help`
  - `python -u tests/test_agent_tools.py`
  - `python -u tests/test_speaker_identify.py --help`  # 说话人分离 + 声纹库对照（--segment tencent 默认）
  - `python -u tests/test_voiceprint_live.py`          # 说话人识别集成模拟（空库首次运行，自动注册新说话人）
  - `python -u tests/test_voiceprint_live.py --reuse`  # 复用特征库，验证持久化加载与命中
- 大部分测试需要真实的腾讯云凭据 + 可用的麦克风；sherpa 测试会自动下载模型。`python -u tests/test_tencent_asr.py --sine 3` 可无麦克风验证 ASR。`python -u tests/test_agent_tools.py` 通过真实的 hermes-agent registry 测试工具。

## 当前状态

