#!/usr/bin/env python3
"""
voice_agent 启动入口。

用法:
    python -u -m voice_agent        # 在 hermes-voice-agent/ 目录下运行
"""
import asyncio
import logging

from .main import VoiceApp

logger = logging.getLogger("main")


async def main():
    app = VoiceApp()
    try:
        await app.start()
        while app._running:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("\nShutting down...")
    finally:
        await app.stop()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    asyncio.run(main())
