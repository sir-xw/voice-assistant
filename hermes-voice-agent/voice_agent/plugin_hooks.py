"""
项目级插件引导：让 hermes 在本进程加载项目插件目录（./.hermes/plugins）下的插件。

为什么需要它：
- hermes 默认只扫描 `~/.hermes/plugins`（用户插件）与内置插件目录；项目插件
  （`./.hermes/plugins`）必须设置环境变量 HERMES_ENABLE_PROJECT_PLUGINS=1 才参与发现；
- standalone 插件默认 opt-in：必须出现在 hermes 配置（~/.hermes/config.yaml）
  的 plugins.enabled 白名单中才会加载。本项目插件不写全局配置，改为在进程内
  把白名单替换为项目插件集合（等价于临时 enable，只影响当前进程）。

注意：必须在 AIAgent 构造之前调用 —— agent/agent_init.py 内部会调用
discover_plugins() 触发插件发现并缓存状态，晚了白名单替换就不生效。
"""

import os

# 本项目插件集合（与 .hermes/plugins/ 下的目录对应）
PROJECT_PLUGINS = {"speech-relay"}


def ensure_project_plugins_loaded():
    """开启项目插件发现 + 进程内白名单 + 触发首次发现，返回插件管理器。"""
    os.environ["HERMES_ENABLE_PROJECT_PLUGINS"] = "1"

    import hermes_cli.plugins as _plugins_mod

    # 进程内白名单：不修改全局 ~/.hermes/config.yaml
    _plugins_mod._get_enabled_plugins = lambda: set(PROJECT_PLUGINS)

    _plugins_mod.discover_plugins()
    return _plugins_mod.get_plugin_manager()
