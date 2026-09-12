#!/usr/bin/env python3
"""
说话人识别集成模拟测试（无需麦克风）。

把多说话人测试音频（tests/4spk.wav）分块 feed 进腾讯云实时识别引擎，
触发 on_sentence 回调，走与 voice_service/inbound.py 相同的处理逻辑：
  句子音频截取（get_sentence_audio）→ CAM++ 声纹识别 → 新说话人自动注册
  → speaker_id 缓存 → "[名字 (ID: 编号)] 内容" 消息拼装。

用法（在 voice_service/ 目录下运行）:
    python -u tests/test_voiceprint_live.py            # 空库首次运行（注册新说话人）
    python -u tests/test_voiceprint_live.py --reuse    # 复用上次库（全部命中）
"""

import argparse
import shutil
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from voice_service.asr_engine import TencentCloudASREngine, TencentASRConfig, ASRState
from voice_service.voiceprint import VoiceprintManager, PROJECT_ROOT

SAMPLE_RATE = 16000
CHUNK = 6400  # 0.2s
LIB_DIR = PROJECT_ROOT / "tests" / "voiceprint_sim_lib"
SRC_WAV = PROJECT_ROOT / "tests" / "4spk.wav"


def make_engine():
    """创建腾讯云 ASR 引擎（说话人分离引擎 + 开启说话人上下文）。"""
    from dotenv import load_dotenv
    import os

    load_dotenv(PROJECT_ROOT / ".env")
    sid = os.getenv("VOICE_SecretId", "")
    skey = os.getenv("VOICE_SecretKey", "")
    appid = os.getenv("VOICE_AppId", "")
    if not sid or not skey or not appid:
        print("❌ 缺少 .env 凭据")
        sys.exit(1)
    cfg = TencentASRConfig(
        secret_id=sid, secret_key=skey, app_id=appid,
        engine_model="16k_zh_en_speaker_2.0",
        needvad=False, voice_format=1,
        enable_speaker_context=1,
    )
    return TencentCloudASREngine(cfg)


def main():
    p = argparse.ArgumentParser(description="说话人识别集成模拟测试")
    p.add_argument("--reuse", action="store_true",
                   help="复用上次特征库（默认清空重建，验证新说话人注册）")
    args = p.parse_args()

    if args.reuse:
        if not (LIB_DIR / "embeddings.npz").is_file():
            print("❌ 无既有特征库，先不带 --reuse 运行一次")
            sys.exit(1)
        print(f"📚 复用特征库: {LIB_DIR}")
    else:
        shutil.rmtree(LIB_DIR, ignore_errors=True)
        print(f"🧹 清空特征库并重建: {LIB_DIR}")

    # 说话人识别管理器（与 main.py 相同的配置项）
    vp = VoiceprintManager(
        lib_dir=LIB_DIR,
        threshold=0.6,
        speaker_names={"spk_100": "爸爸"},  # 模拟管理员已把 spk_100 映射为"爸爸"
    )
    auto_register = True     # 服务端始终自动注册（auto_register 开关已取消）
    min_register_sec = 1.5
    use_cache = True
    vp_round: list[tuple[str, str]] = []
    vp_id_cache: dict[int, tuple[str, float]] = {}

    # 与 main.py _on_asr_sentence 相同的处理逻辑
    def on_sentence(info):
        nonlocal vp_round, vp_id_cache
        text = info.get("text", "").strip()
        if not text:
            return
        tx_spk = info.get("speaker_id", 0)
        if use_cache and tx_spk in vp_id_cache:
            spk_id, score = vp_id_cache[tx_spk]
            label = vp.speaker_label(spk_id)
            print(f"🗣️ [{label}] {text}（相似度 {score:.2f}，缓存）")
            vp_round.append((label, text))
            return
        samples = engine.get_sentence_audio(info)
        if samples is None or len(samples) == 0:
            print(f"⚠️  无法获取句子音频: {text[:20]}")
            return
        spk_id, score = vp.identify(samples)
        if spk_id is None:
            if auto_register and len(samples) / SAMPLE_RATE >= min_register_sec:
                new_id = vp.register(samples)
                if new_id:
                    spk_id, score = new_id, 1.0
                    print(f"🆕 [新说话人] {text} → 已注册为 {spk_id}"
                          f"（{vp.speaker_label(spk_id)}）")
                else:
                    print(f"❔ [未知] {text}（注册失败，相似度 {score:.2f}）")
            else:
                print(f"❔ [未知] {text}（相似度 {score:.2f}）")
        else:
            print(f"🗣️ [{vp.speaker_label(spk_id)}] {text}（相似度 {score:.2f}）")
        vp_round.append((vp.speaker_label(spk_id), text))
        if use_cache and spk_id:
            vp_id_cache[tx_spk] = (spk_id, score)

    # 引擎 + 回调
    engine = make_engine()
    done = threading.Event()
    engine.on_complete = done.set
    engine.on_error = lambda m: (print("❌", m), done.set())
    engine.on_sentence = on_sentence

    # 分块 feed 整段音频（模拟流式）
    print(f"\n🔊 分块 feed: {SRC_WAV.name}...")
    engine.start_recognition()
    for _ in range(50):
        if engine.state in (ASRState.RECORDING, ASRState.ERROR):
            break
        time.sleep(0.1)
    if engine.state == ASRState.ERROR:
        sys.exit(1)

    from scipy.io import wavfile
    sr, data = wavfile.read(str(SRC_WAV))
    if data.ndim > 1:
        data = data.mean(axis=1)
    pcm = data.astype(np.int16).tobytes()
    for i in range(0, len(pcm), CHUNK):
        c = pcm[i:i + CHUNK]
        if len(c) < CHUNK:
            c += b"\x00" * (CHUNK - len(c))
        engine.feed_audio(c)
        time.sleep(CHUNK / SAMPLE_RATE / 2)

    engine.stop_recognition()
    done.wait(15)

    # 消息拼装（与 inbound.py _on_asr_complete 相同）
    print(f"\n{'='*56}")
    print("📤 发送给 LLM 的聊天消息:")
    print(f"{'='*56}")
    if vp_round:
        msg = "\n".join(f"[{spk}] {t}" for spk, t in vp_round)
    else:
        msg = engine.last_text
    print(msg)
    print(f"{'='*56}")
    print(f"\n📊 特征库: {len(vp.speakers)} 人 {vp.speakers}")
    print(f"   speaker_id 缓存: {vp_id_cache}")


if __name__ == "__main__":
    main()
