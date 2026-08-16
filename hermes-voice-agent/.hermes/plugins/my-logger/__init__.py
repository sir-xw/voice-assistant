def on_post_api_request(**kwargs):
    """API调用完成后触发，kwargs里包含请求和响应的详细信息

    注意：
    - hermes 的 hook 回调是同步调用的（invoke_hook 不会 await），
      所以这里不能用 async def，否则回调永远不会执行；
    - 回调以关键字参数方式被调用（task_id/turn_id/model/response/...），
      用 **kwargs 接收完整载荷。
    """
    response = kwargs.get("response") or kwargs.get("assistant_message") or {}
    print("LLM响应：", response)
    print("元信息：model=", kwargs.get("model"), "provider=", kwargs.get("provider"),
          "finish_reason=", kwargs.get("finish_reason"), "usage=", kwargs.get("usage"))


def register(ctx):
    # 核心：将你的函数注册到 post_api_request 钩子上
    ctx.register_hook("post_api_request", on_post_api_request)
    print("插件 my_logger 已激活！")
