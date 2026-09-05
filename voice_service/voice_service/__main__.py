"""
Voice Service 进程入口。

用法：
    python -m voice_service [--config config.yaml] [--host H] [--port P]
                            [--env-file .env] [--audio] [--selfcheck]

- 默认：仅启动 WS 服务端（/healthz、hello、ping/pong；不动音频设备）；
- --audio：启用语音输入（唤醒词监听；打开麦克风与腾讯云 ASR）。**注意**：
  同一机器上同时只能有一个进程占用麦克风 —— 现役 hermes-voice-agent /
  voice-platform 运行期间不要加本参数启动；
- --selfcheck：先跑协议自检。

语音侧播放（TTS 播报/提示音/连续对话窗口）为 M3，尚未装配。
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys


def _parse(argv=None):
    parser = argparse.ArgumentParser(prog="voice_service", description="Voice Service（语音服务）")
    parser.add_argument("--config", default=None, help="配置文件（含 voice_service: 段）")
    parser.add_argument("--host", default=None, help="覆盖监听地址")
    parser.add_argument("--port", type=int, default=None, help="覆盖监听端口")
    parser.add_argument("--env-file", default=".env", help=".env 路径（腾讯云凭据）")
    parser.add_argument("--audio", action="store_true",
                        help="启用语音输入（打开麦克风；勿与现役语音服务同时占用设备）")
    parser.add_argument("--out", action="store_true",
                        help="启用语音输出（TTS→扬声器/提示音/等待音，会真实发声；默认静默）")
    parser.add_argument("--selfcheck", action="store_true", help="运行协议自检后退出")
    parser.add_argument("--verbose", action="store_true", help="DEBUG 日志")
    return parser.parse_args(argv)


async def _amain(args) -> int:
    if args.selfcheck:
        from .protocol import _selftest
        _selftest()
        return 0

    from .service import VoiceServiceApp
    from .service_config import load_voice_service_config

    cfg = load_voice_service_config(args.config, env_file=args.env_file)
    if args.host:
        cfg.service.host = args.host
    if args.port:
        cfg.service.port = args.port

    app = VoiceServiceApp(cfg)
    logging.getLogger("voice_service").info(
        "Voice Service 启动（audio=%s, out=%s, config=%s）",
        args.audio, args.out, cfg.config_path)
    await app.start(audio=args.audio, audio_out=args.out)
    try:
        await asyncio.Event().wait()
    finally:
        await app.stop()
    return 0


def main(argv=None) -> int:
    args = _parse(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        return asyncio.run(_amain(args))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
