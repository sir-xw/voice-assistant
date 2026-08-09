#!/usr/bin/env python3

from voice_agent.config import load_config
from run_agent import AIAgent

from tools.registry import registry


# --- Handler ---

def test_echo(message: str) -> str:
    """print a message to stdout. Returns message string."""
    print("调用自定义工具：", message)
    return message


# --- Schema ---

SCHEMA = {
    "name": "test_echo",
    "description": "测试自定义tool，它会将传入的消息打印到终端，并返回原本的消息.",
    "parameters": {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "a message want to print to stdout"
            },
        },
        "required": ["message"]
    }
}


# --- Registration ---

from tools.registry import registry

registry.register(
    name="test_echo",
    toolset="voice_agent",
    schema=SCHEMA,
    handler=lambda args, **kw: test_echo(
        message=args.get('message')),
)


agent = AIAgent(
    model='deepseek-v4-flash',
    skip_context_files=True,
    skip_memory=True,
    reasoning_config={'enabled': False},
)

result = agent.run_conversation("现在，你想回答的内容都通过`test_echo`工具输出，文字消息只需要返回一个“FINISH”，因为我不会看回复内容")

print(result)
