#!/usr/bin/env python3

import os

# 项目插件目录（./.hermes/plugins）默认不参与插件发现，必须显式开启
# 该环境变量需要在导入 hermes_cli.plugins / 触发 discover 之前设置
os.environ["HERMES_ENABLE_PROJECT_PLUGINS"] = "1"

# 注意：PluginManager 没有 enable_plugin() 方法（hermes 0.20.x 已无此 API）。
# standalone 插件默认 opt-in：必须出现在 ~/.hermes/config.yaml 的
# plugins.enabled 白名单中才会加载。本项目插件不写全局配置，改为在
# 进程内把"白名单"替换为项目插件集合 —— 等价于临时 enable，只影响
# 当前进程（正式环境用 `hermes plugins enable <name>`）。
#
# 这个 patch 必须在构造 AIAgent 之前完成：AIAgent 初始化（agent/agent_init.py）
# 内部会调用 discover_plugins() 触发插件发现并缓存状态，晚了就不生效。
import hermes_cli.plugins as _plugins_mod
_plugins_mod._get_enabled_plugins = lambda: {"my-logger", "speech-relay"}

from pathlib import Path

from voice_agent.config import load_config
from run_agent import AIAgent


# --- Handler ---

def test_callback(message: str) -> str:
    """print a message to stdout. Returns message string."""
    print("中间消息：", message)
    return True

# 1. 创建 Agent 实例
speak_prompt = (
            '\n\n【语音助手通用规则】\n'
            '你是一个语音助手。\n'
            '1. 用户的输入来自语音识别（ASR），可能存在同音字、漏字、多字等错误。\n'
            '   如果问题听起来不合逻辑，结合上下文做合理推断，而不是逐字照搬。\n'
            '2. 回答要简洁，控制在 3 句话以内。\n'
            '   需要列举时用「第一、第二、第三」代替长段落。\n'
            '3. 回答中自然融入确认——不是生硬复述，而是把确认编织在回答里。\n'
            '   例如用户说「今天天气怎么样」，不要说「你是问今天天气吗？」\n'
            '   直接说「今天晴天，25度，适合出门。」\n'
            '   如果确实听清了，不需要额外确认。\n'
            '4. 如果实在听不懂，直接说「不好意思没听清，能再说一遍吗？」\n'
            '\n'
            '【回复规则】\n'
            '你的文字回复消息只会通过语音播报给用户，因此：\n'
            '- 每次tool_calls响应需要同步包含文字回复，让用户了解正在进行的工作\n'
            '- 中间与最终回复消息请直接使用纯文本格式，不需要任何标记文本，格式为：(情绪)你要说的话\n'
            '  例如：(happy)你好，有什么可以帮助你的？\n'
            '  情绪可选值：neutral(中性) sad(悲伤) happy(高兴) angry(生气) fear(恐惧) '
            '- 如果你没有用到情绪，则使用newtral，例如：(neutral)这是回答\n'
            '\n'
        )

agent = AIAgent(
    model='deepseek-v4-flash',
    skip_context_files=True,
    skip_memory=True,
    reasoning_config={'enabled': True, "effort": "low"},
    ephemeral_system_prompt=speak_prompt,
    # interim_assistant_callback=test_callback,
)

# 2. 获取插件管理器并加载项目插件目录
plugin_dir = Path(".hermes/plugins")  # 你的插件目录
from hermes_cli.plugins import get_plugin_manager

manager = get_plugin_manager()
manager.discover_and_load()
for p in manager.list_plugins():
    if p.get("name") in ("my-logger", "speech-relay"):
        print("插件状态：", p)

# 3. 注册播报桥测试接收方：验证 speech-relay 插件转发链路
# （真实运行时 VoiceApp 用 set_sink 注册自己；这里用打印函数演示）
from voice_agent.speech_bridge import set_sink, clear_sink

def _print_reply(payload: dict):
    fr = payload.get("finish_reason")
    am = payload.get("assistant_message")
    if isinstance(am, dict):
        content = am.get("content") or ""
    else:
        content = getattr(am, "content", None) or ""
    print(f"📣 [hook] finish_reason={fr} content={str(content)[:80]!r}")

set_sink(_print_reply)

result = agent.run_conversation("帮我统计一下/root/git/voice-assistant/hermes-voice-agent目录中文本文件的数量，然后找出子文件最多的文件夹是哪个")

clear_sink()

print('最终回答：', result)
