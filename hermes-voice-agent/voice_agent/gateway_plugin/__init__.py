"""
voice-platform — hermes gateway 语音平台插件。

把 hermes-voice-agent 的语音交互路径（唤醒词 → VAD → 腾讯云 ASR → agent →
腾讯云 TTS → 播放）接入 hermes gateway，作为 gateway 的一个平台：

- **inbound**：sherpa 唤醒词 / 连续对话窗口期 VAD → 腾讯云 ASR 文本 →
  `MessageEvent` → gateway 会话（每个唤醒词一个独立会话，chat_id=`wake:<名>`）
- **outbound**：gateway 回复 → 适配器 `send()` → 情绪分段解析 → 全局播报队列
  （串行）→ 腾讯云 TTS 流式播放 → 通知音 → 对话窗口期
- **中间轮播报**：插件自带 `post_api_request` 钩子，工具轮文字回复直接播报
  （替代已废除的 speak 工具）
- **工具披露**：通过 `toolsets_for_source()` / `platform_toolsets.voice` 条件
  披露 mpd_* 等工具集，仅对语音平台开放

分发方式：pyproject.toml 的 `hermes_agent.plugins` entry point 注册
（pip 安装即成为 hermes 插件），亦可目录部署（plugin.yaml + 本模块）。

注意：**本模块不得在顶层导入 `gateway.*`** —— providers/__init__.py 会遍历
整个 `hermes_agent.plugins` entry-point 组并 `ep.load()` 每个目标模块，若顶层
导入 gateway.config（部分初始化期）会触发循环导入告警。因此 register() 内
延迟导入 adapter（真实加载发生在 PluginManager 路径，此时 gateway 已就绪）。
"""


def register(ctx):
    """插件入口：延迟导入 adapter 并转发（避免 entry-point 扫描期循环导入）。"""
    from .adapter import register as _register

    return _register(ctx)


__all__ = ["register"]
