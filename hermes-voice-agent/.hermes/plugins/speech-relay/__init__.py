from voice_agent.speech_bridge import emit


def on_post_api_request(**kwargs):
    """API 调用完成后触发：把回复载荷转发给 voice_agent 播放队列。

    注意：
    - hermes 的 hook 回调是同步调用的（invoke_hook 不会 await），不能用 async def；
    - 回调以关键字参数方式被调用（task_id/model/finish_reason/assistant_message/...），
      用 **kwargs 接收完整载荷后整体交给 speech_bridge 转发；
    - VoiceApp 启动时已通过 speech_bridge.set_sink() 注册接收方；未注册时转发为空操作。
    """
    emit(kwargs)


def register(ctx):
    # 核心：将转发函数注册到 post_api_request 钩子上
    ctx.register_hook("post_api_request", on_post_api_request)
    print("插件 speech-relay 已激活！")
