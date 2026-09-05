"""
hermes_gateway_plugin — Hermes voice gateway 客户端（voice-platform 插件）。

插件入口 register()。与 hermes 绑定：加载于 hermes gateway 进程
（entry point：hermes_agent.plugins → voice-platform）。

注意：本模块顶层**不得** import gateway.* 或 voice_service 之外的 hermes 模块
——插件扫描期（entry point ep.load()）会加载本模块，过早 import 可能触发
循环导入/拖慢扫描；真正的接入在 register() 内延迟导入 adapter。
"""

__version__ = "0.1.0"


def register(ctx):
    """插件入口：延迟导入 adapter 并转发注册。"""
    from . import adapter
    return adapter.register(ctx)
