#!/usr/bin/env python

"""
腾讯云 TTS 文本转语音 — 生成提示音频文件。

用法:
    python -u tools/tts_gen.py --text "你好" --output prompt.wav
    python -u tools/tts_gen.py --text "我在听" --voice 101001 --speed 1.0 --output listening.wav

依赖:
    pip install websocket-client
"""

import argparse
import json
import logging
import os
import sys
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tencentcloud_speech.common.credential import Credential
from tencentcloud_speech.tts.flowing_speech_synthesizer import (
    FlowingSpeechSynthesizer,
    FlowingSpeechSynthesisListener,
    FlowingSpeechSynthesizer_ACTION_SYNTHESIS,
    FlowingSpeechSynthesizer_ACTION_COMPLETE,
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("tts_gen")


class _Collector(FlowingSpeechSynthesisListener):
    """收集 PCM 音频块。"""

    def __init__(self):
        super().__init__()
        self.chunks: list[bytes] = []
        self.error: str | None = None

    def on_audio_result(self, audio_bytes: bytes):
        self.chunks.append(audio_bytes)

    def on_synthesis_fail(self, response):
        self.error = f"[{response.get('code')}] {response.get('message')}"
        logger.error(f"TTS fail: {self.error}")


def tts_to_file(
    text: str,
    output: str,
    voice_type: int = 101001,
    speed: float = 0.0,
    volume: float = 0.0,
    sample_rate: int = 16000,
    codec: str = "pcm",
    emotion: str = "",
    timeout: float = 20.0,
):
    """
    使用腾讯云流式 TTS 合成文本并保存为 WAV 文件。

    Args:
        text: 待合成文本
        output: 输出文件路径（自动根据 codec 确定后缀）
        voice_type: 音色 ID（101001=晓晓）
        speed: 语速 -2~6
        volume: 音量 -10~10
        sample_rate: 采样率 8000/16000/24000
        codec: "pcm" 或 "mp3"
        timeout: 超时秒数
    """
    from dotenv import load_dotenv

    dotenv_path = Path(__file__).resolve().parent.parent.parent / ".env"
    load_dotenv(dotenv_path)

    secret_id = os.getenv("VOICE_SecretId", "")
    secret_key = os.getenv("VOICE_SecretKey", "")
    app_id = os.getenv("VOICE_AppId", "")

    if not secret_id or not secret_key or not app_id:
        print("❌ 请在 .env 中设置 VOICE_SecretId / VOICE_SecretKey / VOICE_AppId")
        sys.exit(1)

    print(f"🔊 正在合成...")
    print(f"   文本: {text[:60]}{'...' if len(text) > 60 else ''}")
    print(f"   音色: {voice_type}  语速: {speed}  音量: {volume}")
    print(f"   采样率: {sample_rate}  格式: {codec}")

    credential = Credential(secret_id, secret_key)
    collector = _Collector()
    synthesizer = FlowingSpeechSynthesizer(app_id, credential, collector)

    # 配置
    synthesizer.set_voice_type(voice_type)
    synthesizer.set_codec(codec)
    synthesizer.set_sample_rate(sample_rate)
    synthesizer.set_speed(speed)
    synthesizer.set_volume(volume)
    synthesizer.set_enable_subtitle(0)
    if emotion:
        synthesizer.set_emotion_category(emotion)

    # 启动
    synthesizer.start()
    if not synthesizer.wait_ready(int(timeout * 1000)):
        print("❌ TTS 连接超时")
        synthesizer.wait()
        sys.exit(1)

    # 合成
    synthesizer.process(text, FlowingSpeechSynthesizer_ACTION_SYNTHESIS)
    synthesizer.complete(FlowingSpeechSynthesizer_ACTION_COMPLETE)
    synthesizer.wait()

    if collector.error:
        print(f"❌ 合成失败: {collector.error}")
        sys.exit(1)

    if not collector.chunks:
        print("❌ 未收到音频数据")
        sys.exit(1)

    # 合并音频块
    audio_data = b"".join(collector.chunks)

    # 确定输出路径
    output_path = Path(output)
    if output_path.suffix == "":
        ext = ".mp3" if codec == "mp3" else ".wav"
        output_path = output_path.with_suffix(ext)

    # PCM → WAV
    if codec == "pcm":
        import struct

        with wave.open(str(output_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data)

        duration = len(audio_data) / (sample_rate * 2)
        print(f"✅ 已保存: {output_path}")
        print(f"   时长: {duration:.1f}s  大小: {len(audio_data)/1024:.1f} KB")

    elif codec == "mp3":
        output_path.write_bytes(audio_data)
        print(f"✅ 已保存: {output_path}")
        print(f"   大小: {len(audio_data)/1024:.1f} KB")

    else:
        output_path.write_bytes(audio_data)
        print(f"✅ 已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="腾讯云 TTS 文本转语音 — 生成提示音频文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -u tools/tts_gen.py --text "我在" --output prompt.wav
  python -u tools/tts_gen.py --text "请说话" --voice 101002 --speed 1.0 --output speak.wav
  python -u tools/tts_gen.py --text "你好" --codec mp3 --output hello.mp3
        """,
    )

    parser.add_argument("--text", required=True, help="待合成文本")
    parser.add_argument("--output", "-o", required=True, help="输出文件路径")
    parser.add_argument("--voice", type=int, default=101001,
                        help="音色 ID（默认 101001=晓晓）")
    parser.add_argument("--speed", type=float, default=0.0,
                        help="语速 -2~6（默认 0）")
    parser.add_argument("--volume", type=float, default=0.0,
                        help="音量 -10~10（默认 0）")
    parser.add_argument("--sample-rate", type=int, default=16000,
                        choices=[8000, 16000, 24000],
                        help="采样率（默认 16000）")
    parser.add_argument("--codec", default="pcm", choices=["pcm", "mp3"],
                        help="输出格式（默认 pcm=wav）")
    parser.add_argument("--emotion",
                        help="语音情绪/风格，可选：neutral sad happy angry fear "
                             "story poetry sajiao disgusted amaze exciting aojiao jieshuo")

    args = parser.parse_args()

    tts_to_file(
        text=args.text,
        output=args.output,
        voice_type=args.voice,
        speed=args.speed,
        volume=args.volume,
        sample_rate=args.sample_rate,
        codec=args.codec,
        emotion=args.emotion or "",
    )


if __name__ == "__main__":
    main()
