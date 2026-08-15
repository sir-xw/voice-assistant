#!/usr/bin/env python3
"""
说话人分离 + 声纹特征库对照 + ASR 关联测试（仿照 test_tencent_asr.py 的结构）。

两种切分方式（--segment 选择）:

  1. --segment tencent（默认，推荐）——腾讯云句子段落 + 声纹识别:
     整段音频一次性交给腾讯云实时识别（引擎 16k_zh_en_speaker_2.0，
     开启说话人分离），保持完整语境识别；腾讯云返回的每条最终句子自带
     speaker_id 和起止毫秒。随后按句子时间戳截取音频片段，调用 CAM++
     提取声纹并与特征库对照，识别出该段落的具体说话人身份。
     切分粒度与说话人编号由腾讯云给出（比本地 pyannote 更准），本地
     只负责"这段声音是谁"。

  2. --segment local ——本地 pyannote 切分 + 逐段识别（对照）:
     用 sherpa-onnx OfflineSpeakerDiarization（pyannote segmentation 3.0
     + CAM++）本地切分说话人区间，再对每个区间做 ASR（--asr 选择
     tencent/sherpa/none）并提取声纹对照特征库。

腾讯云句子偶尔会把紧邻的两个说话人拼进同一句（如 4spk.wav 的
"明明暗暗"与"和你们一起走过…"），脚本会对每个句子做滑窗声纹一致性
检测：发现句内出现多个身份时自动细分并标注 ⚠️。

声纹特征库:
  仓库目前没有现成特征库 → 先用已有测试音频创建:
      python -u tests/test_speaker_identify.py --build-lib
  默认从 tests/4spk.wav（56.9s 多说话人测试音频）分离出各说话人，
  为每人注册一段或多段 CAM++ embedding，保存到 tests/voiceprint_lib/
  （embeddings.npz + meta.json），识别时直接加载复用，无需重新提取。

用法（在 hermes-voice-agent/ 目录下运行，voice_agent 已 pip install -e .）:
    # 1) 从已有测试音频创建声纹特征库
    python -u tests/test_speaker_identify.py --build-lib
    #    自定义说话人名字（逗号分隔，按首次出现顺序对应各说话人）
    python -u tests/test_speaker_identify.py --build-lib \
        --speaker-names "张三,李四,王五,赵六"

    # 2) 腾讯云句子段落 → 声纹对照 → 说话人身份（默认）
    python -u tests/test_speaker_identify.py --file tests/4spk.wav

    # 3) 本地 pyannote 切分 + 逐段识别（对照；--asr 可选 tencent/sherpa/none）
    python -u tests/test_speaker_identify.py --file tests/4spk.wav --segment local
    python -u tests/test_speaker_identify.py --file tests/4spk.wav --segment local --asr sherpa

依赖:
    sherpa-onnx（hermes venv 已安装）、numpy、scipy
    腾讯云 ASR 模式需要 .env 中的 VOICE_SecretId / VOICE_SecretKey / VOICE_AppId
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

# 腾讯云 SDK / ASR 引擎的内部日志较吵，测试输出只需 WARNING 及以上
logging.getLogger("voice_agent.asr_engine").setLevel(logging.WARNING)
logging.getLogger("tencent_speech.log").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# 常量（模型/音频路径基于仓库根目录解析，任何 cwd 下都能跑）
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_RATE = 16000

# 说话人分离模型（pyannote segmentation 3.0，划分语音段 + 说话人变化点）
SEG_MODEL = PROJECT_ROOT / "models" / "sherpa-spk" / "sherpa-onnx-pyannote-segmentation-3-0" / "model.int8.onnx"
# CAM++ 声纹提取模型（192 维说话人 embedding）
EMB_MODEL = PROJECT_ROOT / "models" / "sherpa-spk" / "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx"
# 本地备选 ASR 模型（streaming zipformer，transducer 结构）
SHERPA_ASR_DIR = PROJECT_ROOT / "models" / "sherpa-asr" / "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20"

# 声纹特征库默认位置与构建来源
DEFAULT_LIB_DIR = PROJECT_ROOT / "tests" / "voiceprint_lib"
DEFAULT_LIB_SRC = PROJECT_ROOT / "tests" / "4spk.wav"

# 腾讯云实时识别"说话人分离"专用引擎（句子自带 speaker_id + 起止毫秒）
TENCENT_SPEAKER_MODEL = "16k_zh_en_speaker_2.0"

# 声纹匹配阈值（CAM++ 余弦相似度，高于此值判定同一人；与 test_sensevoice_asr.py 一致）
SPEAKER_THRESHOLD = 0.6


# ---------------------------------------------------------------------------
# 音频工具
# ---------------------------------------------------------------------------

def load_wav_float32(path: str):
    """
    读取 WAV 为 float32 样本（[-1, 1]），返回 (samples, sample_rate)。
    多声道取均值；非 16k 由调用方重采样。
    """
    from scipy.io import wavfile

    sr, data = wavfile.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1).astype(data.dtype)
    if data.dtype == np.int16:
        samples = data.astype(np.float32) / 32768
    else:
        samples = data.astype(np.float32)
        if samples.max() > 1.0:
            samples = samples / 32768
    return samples, sr


def to_16k(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """重采样到 16kHz（说话人分离与声纹模型均要求 16k 单声道）。"""
    if sample_rate == SAMPLE_RATE:
        return samples
    from scipy import signal
    n = round(len(samples) * SAMPLE_RATE / sample_rate)
    return signal.resample(samples, n).astype(np.float32)


def samples_to_pcm(samples: np.ndarray) -> bytes:
    """float32 [-1, 1] → 16-bit PCM bytes（腾讯云 ASR 输入）。"""
    return (samples * 32768).astype(np.int16).tobytes()


# ---------------------------------------------------------------------------
# 说话人分离（pyannote segmentation + CAM++ embedding + 聚类）
# ---------------------------------------------------------------------------

def create_diarizer(num_threads: int = 2, cluster_threshold: float = 0.6):
    """
    创建 sherpa-onnx 说话人分离器（segmentation + embedding + 聚类）。

    cluster_threshold: 聚类相似度阈值，高于此值的段合并为同一说话人。
    默认 0.6：过低会把同一人（相似度 0.6~0.8 的两次发言）拆成多人；
    过高会把音色接近的不同人误合并。4spk.wav 实测同一人最低 0.758、
    不同人最高 0.443，0.6 是安全取值。
    """
    import sherpa_onnx

    for f in (SEG_MODEL, EMB_MODEL):
        if not f.is_file():
            raise FileNotFoundError(f"模型文件缺失: {f}")

    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=str(SEG_MODEL),
            ),
            num_threads=num_threads,
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(EMB_MODEL),
            num_threads=num_threads,
            provider="cpu",
        ),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=-1, threshold=cluster_threshold,
        ),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    if not config.validate():
        raise ValueError(f"无效的说话人分离配置: {config}")
    diarizer = sherpa_onnx.OfflineSpeakerDiarization(config)
    return diarizer


def diarize(diarizer, samples: np.ndarray):
    """
    对音频做说话人分离，返回按时间排序的段列表 [(start, end, speaker)]。
    speaker 是分离器内部的全局说话人编号（不连续）。
    """
    result = diarizer.process(samples)
    return [(seg.start, seg.end, seg.speaker) for seg in result.sort_by_start_time()]


def exclude_overlap(segments):
    """
    把相互重叠的说话人段裁剪为互不重叠的"独占"段（重叠部分丢弃）。

    多人同时说话的重叠区会同时混入两个人的声音，既污染声纹 embedding，
    也会让 ASR 文本张冠李戴；测试音频（如 4spk.wav）存在这种重叠，默认裁剪。
    """
    out = []
    for i, (s, e, spk) in enumerate(segments):
        # 收集本段被其他段覆盖的交集区间
        cuts = []
        for j, (s2, e2, _) in enumerate(segments):
            if i == j:
                continue
            lo, hi = max(s, s2), min(e, e2)
            if hi > lo:
                cuts.append((lo, hi))
        # 从 [s, e] 中减去所有交集，保留独占部分
        cur = s
        for lo, hi in sorted(cuts):
            if lo > cur:
                out.append((cur, min(lo, e), spk))
            cur = max(cur, hi)
            if cur >= e:
                break
        if cur < e:
            out.append((cur, e, spk))
    return [(s, e, spk) for s, e, spk in out if e - s >= 0.05]


# ---------------------------------------------------------------------------
# CAM++ 声纹提取
# ---------------------------------------------------------------------------

def create_speaker_extractor(num_threads: int = 2):
    """创建 sherpa-onnx CAM++ 说话人 embedding 提取器。"""
    import sherpa_onnx

    if not EMB_MODEL.is_file():
        raise FileNotFoundError(f"模型文件缺失: {EMB_MODEL}")
    config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=str(EMB_MODEL),
        num_threads=num_threads,
        debug=False,
        provider="cpu",
    )
    if not config.validate():
        raise ValueError(f"无效的说话人模型配置: {config}")
    extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
    return extractor


def extract_embedding(extractor, samples: np.ndarray):
    """
    对一段音频提取 CAM++ 声纹 embedding，返回 1-D float32 数组。
    音频太短（不足模型最短输入）时返回 None。
    """
    if len(samples) < int(0.2 * SAMPLE_RATE):
        return None
    stream = extractor.create_stream()
    stream.accept_waveform(sample_rate=SAMPLE_RATE, waveform=samples)
    stream.input_finished()
    if not extractor.is_ready(stream):
        return None
    return np.array(extractor.compute(stream))


# ---------------------------------------------------------------------------
# 声纹特征库（构建 / 加载 / 比对）
# ---------------------------------------------------------------------------

def parse_speaker_names(raw: str):
    """解析 --speaker-names "张三,李四" → ["张三", "李四"]。"""
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def build_voiceprint_lib(lib_dir: Path, src_wav: Path, speaker_names_raw: str,
                         min_sec: float, num_threads: int,
                         cluster_threshold: float):
    """
    从一段多说话人音频构建声纹特征库：
      说话人分离 → 每说话人逐段提取 CAM++ embedding → 注册 → 保存 npz + meta。

    保存格式:
      lib_dir/embeddings.npz   每个说话人一个数组（N 段 × 192 维）
      lib_dir/meta.json        来源音频、模型、维度、说话人名字列表等
    """
    import sherpa_onnx

    if not src_wav.is_file():
        print(f"❌ 特征库来源音频不存在: {src_wav}")
        sys.exit(1)

    lib_dir = Path(lib_dir)
    lib_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🔊 加载来源音频: {src_wav}")
    samples, sr = load_wav_float32(str(src_wav))
    samples = to_16k(samples, sr)
    print(f"   时长: {len(samples) / SAMPLE_RATE:.1f}s @ {SAMPLE_RATE}Hz")

    print("\n🗣️ 说话人分离中（pyannote segmentation + CAM++）...")
    diarizer = create_diarizer(num_threads, cluster_threshold)
    segments = diarize(diarizer, samples)
    segments = exclude_overlap(segments)
    segments = [(s, e, spk) for s, e, spk in segments if e - s >= min_sec]
    if not segments:
        print("❌ 未分离出任何说话人区间，无法构建特征库")
        sys.exit(1)

    # 按首次出现顺序给说话人编号 0..N-1（分离器内部编号可能不连续）
    order = {}
    for s, e, spk in segments:
        order.setdefault(spk, len(order))
    n_spk = len(order)
    print(f"   分离出 {n_spk} 个说话人，{len(segments)} 个语音段")

    names = parse_speaker_names(speaker_names_raw)
    if len(names) > n_spk:
        print(f"⚠️  提供了 {len(names)} 个名字，但只分离出 {n_spk} 个说话人，多余名字忽略")
        names = names[:n_spk]
    for i in range(n_spk):
        if i >= len(names):
            names.append(f"spk_{i}")

    extractor = create_speaker_extractor(num_threads)
    manager = sherpa_onnx.SpeakerEmbeddingManager(extractor.dim)
    lib_data = {}
    print("\n🎙️ 提取各说话人声纹并注册:")
    for spk in sorted(order, key=order.get):
        name = names[order[spk]]
        embs, speech_time = [], 0.0
        for s, e, spk2 in segments:
            if spk2 != spk:
                continue
            audio = samples[int(s * SAMPLE_RATE):int(e * SAMPLE_RATE)]
            emb = extract_embedding(extractor, audio)
            if emb is not None:
                embs.append(emb)
                speech_time += e - s
        if not embs:
            print(f"  ⚠️  说话人 {name}: 无可用语音段（过短），跳过注册")
            continue
        if not manager.add(name, embs):
            print(f"  ❌ 说话人 {name}: 注册失败")
            continue
        lib_data[name] = np.array(embs)
        print(f"  ✅ {name}: {len(embs)} 段 / {speech_time:.1f}s 语音")

    if not lib_data:
        print("❌ 没有任何说话人注册成功")
        sys.exit(1)

    # 持久化
    np.savez(lib_dir / "embeddings.npz", **lib_data)
    meta = {
        "source": str(src_wav),
        "embedding_model": EMB_MODEL.name,
        "dim": int(extractor.dim),
        "threshold": SPEAKER_THRESHOLD,
        "cluster_threshold": cluster_threshold,
        "speakers": list(lib_data.keys()),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (lib_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'='*50}")
    print(f"✅ 声纹特征库构建完成: {lib_dir}")
    print(f"   embeddings.npz（{len(lib_data)} 人 × {meta['dim']} 维）")
    print(f"   meta.json")
    for name in meta["speakers"]:
        print(f"   - {name}")
    print(f"{'='*50}")


def load_voiceprint_lib(lib_dir: Path):
    """
    从磁盘加载声纹特征库，重建 SpeakerEmbeddingManager。
    返回 (manager, meta)。
    """
    import sherpa_onnx

    lib_dir = Path(lib_dir)
    meta_path = lib_dir / "meta.json"
    npz_path = lib_dir / "embeddings.npz"
    if not meta_path.is_file() or not npz_path.is_file():
        print(f"❌ 声纹特征库不存在: {lib_dir}")
        print(f"   请先构建: python -u tests/test_speaker_identify.py --build-lib")
        sys.exit(1)

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    data = np.load(npz_path)
    manager = sherpa_onnx.SpeakerEmbeddingManager(int(meta["dim"]))
    for name in meta["speakers"]:
        embs = np.asarray(data[name])  # (N, dim)
        manager.add(name, [embs[i] for i in range(len(embs))])

    print(f"📚 加载声纹特征库: {lib_dir}")
    print(f"   来源: {meta.get('source')} | 模型: {meta.get('embedding_model')}")
    print(f"   已注册 {len(meta['speakers'])} 人: {', '.join(meta['speakers'])}")
    return manager, meta


def identify_speaker(manager, emb: np.ndarray, threshold: float):
    """
    与特征库比对，返回 (说话人名字, 相似度)。
    最高相似度低于 threshold 时返回 (None, 最高相似度) —— 视为未知说话人。
    """
    best_name, best_score = None, -1.0
    for name in manager.all_speakers:
        score = manager.score(name, emb)
        if score > best_score:
            best_name, best_score = name, score
    if best_score < threshold:
        return None, best_score
    return best_name, best_score


def sub_segment_identify(extractor, manager, samples: np.ndarray,
                         start_s: float, end_s: float, threshold: float,
                         win_sec: float = 0.6, step_sec: float = 0.3):
    """
    对片段做滑窗声纹识别，检测片段内是否存在说话人切换。

    腾讯云句子是按语义断句的，跨说话人时可能把两个人的话拼进一句
    （如 4spk.wav 中"明明暗暗"与"和你们一起走过…"被切在同一句），
    此时整句提取的声纹是混合的。这里用 0.6s 窗 / 0.3s 步滑窗逐个
    提取声纹并与特征库比对，把连续同一身份的窗口合并为子段。

    返回 (sub_segments, changed):
      sub_segments: [(start, end, name, score), ...] 按时间排序
      changed: 片段内是否检测到 2 个及以上身份（说话人切换）
    """
    sr = SAMPLE_RATE
    n_win, n_step = int(win_sec * sr), int(step_sec * sr)
    start, end = int(start_s * sr), int(end_s * sr)

    labels = []  # (window_start, window_end, name, score)
    pos = start
    while pos + n_win <= end:
        window = samples[pos:pos + n_win]
        emb = extract_embedding(extractor, window)
        if emb is not None:
            name, score = identify_speaker(manager, emb, threshold)
            if name is not None:  # 丢弃"未知"窗口，避免静音/噪声干扰
                labels.append((pos / sr, (pos + n_win) / sr, name, score))
        pos += n_step

    if not labels:
        return [], False

    # 合并连续同一身份的窗口（间隙 < 0.15s 视为连续）
    merged = []
    for st, en, nm, sc in labels:
        if merged and merged[-1][2] == nm and st - merged[-1][1] < 0.15:
            merged[-1] = (merged[-1][0], en, nm, max(merged[-1][3], sc))
        else:
            merged.append((st, en, nm, sc))

    names = {m[2] for m in merged}
    return merged, len(names) > 1


# ---------------------------------------------------------------------------
# ASR（腾讯云实时识别，仿 test_tencent_asr.py 的引擎用法）
# ---------------------------------------------------------------------------

def make_tencent_engine(engine_model: str = "16k_zh",
                        enable_speaker_context: int = 0):
    """
    从 .env + config.yaml 创建腾讯云 ASR 引擎（与 test_tencent_asr.py 一致）。

    enable_speaker_context=1 时开启说话人分离（句子带 speaker_id），
    此时 engine_model 必须用支持分离的引擎（如 16k_zh_en_speaker_2.0）。
    """
    from dotenv import load_dotenv
    import os

    # SDK import 时会把自己 logger 级别重置为 INFO，须在此之后再压静
    import voice_agent.asr_engine  # noqa: F401
    logging.getLogger("tencent_speech.log").setLevel(logging.WARNING)
    logging.getLogger("voice_agent.asr_engine").setLevel(logging.WARNING)

    from voice_agent.asr_engine import TencentCloudASREngine, TencentASRConfig

    dotenv_path = PROJECT_ROOT / ".env"
    load_dotenv(dotenv_path)

    sid = os.getenv("VOICE_SecretId", "")
    skey = os.getenv("VOICE_SecretKey", "")
    appid = os.getenv("VOICE_AppId", "")

    if not sid or not skey:
        print("❌ 缺少 .env 中的 VOICE_SecretId / VOICE_SecretKey")
        sys.exit(1)
    if not appid:
        print("❌ 缺少 .env 中的 VOICE_AppId")
        sys.exit(1)

    cfg = TencentASRConfig(
        secret_id=sid, secret_key=skey, app_id=appid,
        engine_model=engine_model or "16k_zh", needvad=False, voice_format=1,
        enable_speaker_context=enable_speaker_context,
    )
    return TencentCloudASREngine(cfg)


def asr_tencent_segment(engine, samples: np.ndarray, chunk_size: int = 6400):
    """
    对一段音频（float32, 16k）调用腾讯云实时 ASR，返回识别文本。
    发送节奏模拟实时率（与 test_tencent_asr.py 的 run_file 一致）。
    """
    from voice_agent.asr_engine import ASRState

    pcm = samples_to_pcm(samples)
    finals: list[str] = []
    engine.on_final = lambda text: finals.append(text)

    engine.start_recognition()
    # 等 WS 连接建立
    for _ in range(50):
        if engine.state in (ASRState.RECORDING, ASRState.ERROR):
            break
        time.sleep(0.1)
    if engine.state == ASRState.ERROR:
        return ""

    for i in range(0, len(pcm), chunk_size):
        chunk = pcm[i:i + chunk_size]
        if len(chunk) < chunk_size:
            chunk = chunk + b"\x00" * (chunk_size - len(chunk))
        engine.feed_audio(chunk)
        time.sleep(chunk_size / SAMPLE_RATE / 2)  # 模拟实时率

    engine.stop_recognition()
    if not engine.wait_for_complete(10):
        print("  ⚠️  ASR 未在 10s 内回调 complete")
    return "".join(finals) or engine.last_text


def asr_tencent_full(engine, samples: np.ndarray, chunk_size: int = 6400):
    """
    整段音频一次性腾讯云实时识别（保持完整语境），返回最终句子列表。

    每句: {text, speaker_id, start_ms, end_ms, sentence_id}
    需使用支持说话人分离的引擎（16k_zh_en_speaker_2.0）并开启
    enable_speaker_context，否则 speaker_id 恒为 0、无分离意义。
    """
    from voice_agent.asr_engine import ASRState

    pcm = samples_to_pcm(samples)
    engine.start_recognition()
    # 等 WS 连接建立
    for _ in range(50):
        if engine.state in (ASRState.RECORDING, ASRState.ERROR):
            break
        time.sleep(0.1)
    if engine.state == ASRState.ERROR:
        print("  ❌ ASR 连接失败")
        return []

    for i in range(0, len(pcm), chunk_size):
        chunk = pcm[i:i + chunk_size]
        if len(chunk) < chunk_size:
            chunk = chunk + b"\x00" * (chunk_size - len(chunk))
        engine.feed_audio(chunk)
        time.sleep(chunk_size / SAMPLE_RATE / 2)  # 模拟实时率

    engine.stop_recognition()
    if not engine.wait_for_complete(15):
        print("  ⚠️  ASR 未在 15s 内回调 complete")
    return engine.sentences


def create_sherpa_recognizer(num_threads: int = 2):
    """创建 sherpa-onnx 本地流式识别器（streaming zipformer）。"""
    import sherpa_onnx

    model_dir = SHERPA_ASR_DIR
    if not model_dir.is_dir():
        raise FileNotFoundError(f"sherpa ASR 模型目录缺失: {model_dir}")

    enc = model_dir / "encoder-epoch-99-avg-1.onnx"
    if (model_dir / "encoder-epoch-99-avg-1.int8.onnx").is_file():
        enc = model_dir / "encoder-epoch-99-avg-1.int8.onnx"
    dec = model_dir / "decoder-epoch-99-avg-1.onnx"
    if (model_dir / "decoder-epoch-99-avg-1.int8.onnx").is_file():
        dec = model_dir / "decoder-epoch-99-avg-1.int8.onnx"
    joi = model_dir / "joiner-epoch-99-avg-1.onnx"
    if (model_dir / "joiner-epoch-99-avg-1.int8.onnx").is_file():
        joi = model_dir / "joiner-epoch-99-avg-1.int8.onnx"
    tok = model_dir / "tokens.txt"
    for f in (enc, dec, joi, tok):
        if not f.is_file():
            raise FileNotFoundError(f"模型文件缺失: {f}")

    recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=str(tok),
        encoder=str(enc),
        decoder=str(dec),
        joiner=str(joi),
        num_threads=num_threads,
        provider="cpu",
        sample_rate=SAMPLE_RATE,
        feature_dim=80,
        decoding_method="greedy_search",
    )
    return recognizer


def asr_sherpa_segment(recognizer, samples: np.ndarray):
    """对一段音频用本地 sherpa ASR 识别，返回文本（尾部补 0.5s 静音）。"""
    stream = recognizer.create_stream()
    stream.accept_waveform(SAMPLE_RATE, samples)
    tail = np.zeros(int(0.5 * SAMPLE_RATE), dtype=np.float32)
    stream.accept_waveform(SAMPLE_RATE, tail)
    stream.input_finished()
    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)
    return recognizer.get_result(stream)


# ---------------------------------------------------------------------------
# 识别主流程
# ---------------------------------------------------------------------------

def fmt_time(t: float) -> str:
    m, s = divmod(int(t), 60)
    return f"{m:02d}:{s:02d}"


def _load_audio_and_lib(file_path: Path, lib_dir: Path):
    """
    加载音频（重采样到 16k）与声纹特征库，返回 (manager, meta, samples)。
    """
    if not file_path.is_file():
        print(f"❌ 音频文件不存在: {file_path}")
        sys.exit(1)

    manager, meta = load_voiceprint_lib(lib_dir)
    # 特征库模型与当前模型不一致时提示（维度不同会直接出错）
    if meta.get("embedding_model") != EMB_MODEL.name:
        print(f"⚠️  特征库由 {meta.get('embedding_model')} 构建，当前模型为 "
              f"{EMB_MODEL.name}，比对结果可能不准确")

    print(f"\n🔊 加载音频: {file_path}")
    samples, sr = load_wav_float32(str(file_path))
    samples = to_16k(samples, sr)
    print(f"   时长: {len(samples) / SAMPLE_RATE:.1f}s @ {SAMPLE_RATE}Hz")
    return manager, meta, samples


def identify_by_tencent_sentences(file_path: Path, lib_dir: Path,
                                  threshold: float, min_sec: float,
                                  num_threads: int, engine_model: str):
    """
    腾讯云句子段落模式（推荐）:
      整段一次实时识别（说话人分离引擎）→ 腾讯云句子段落（speaker_id + 时间戳）
      → 按句子时间戳截取音频 → CAM++ 声纹 → 特征库对照 → 说话人身份。
    """
    manager, meta, samples = _load_audio_and_lib(file_path, lib_dir)
    extractor = create_speaker_extractor(num_threads)

    print(f"\n🎙️ 腾讯云实时识别（{engine_model}，开启说话人分离）...")
    engine = make_tencent_engine(engine_model, enable_speaker_context=1)
    t0 = time.time()
    sentences = asr_tencent_full(engine, samples)
    elapsed = time.time() - t0
    if not sentences:
        print("❌ 未获得腾讯云句子结果")
        return
    n_spk = len({s["speaker_id"] for s in sentences})
    print(f"   识别完成: {len(sentences)} 句 / {n_spk} 个说话人"
          f"（腾讯云 speaker_id: {sorted({s['speaker_id'] for s in sentences})}），"
          f"耗时 {elapsed:.0f}s")

    # 逐句：按时间戳截取 → 声纹对照 → 说话人身份
    print(f"\n{'='*72}")
    print(f"🎙️  句子段落  |  腾讯云spk → 声纹身份（相似度）  |  文本")
    print(f"{'='*72}")
    results = []
    for s in sentences:
        start_s, end_s = s["start_ms"] / 1000.0, s["end_ms"] / 1000.0
        if end_s - start_s < min_sec:
            continue
        audio = samples[int(start_s * SAMPLE_RATE):int(end_s * SAMPLE_RATE)]
        if len(audio) == 0:
            continue

        emb = extract_embedding(extractor, audio)
        name, score = (identify_speaker(manager, emb, threshold)
                       if emb is not None else (None, 0.0))
        hit = f"{name}（{score:.3f}）" if name else f"未知（{score:.3f}）"
        results.append({
            "start": start_s, "end": end_s,
            "tx_spk": s["speaker_id"], "name": name, "score": score,
            "text": s["text"],
        })
        print(f"[{fmt_time(start_s)}-{fmt_time(end_s)}] 腾讯云spk{s['speaker_id']} → {hit}")
        print(f"      📝 {s['text']}")

        # 句内声纹一致性检测：腾讯云断句可能把两人拼进一句（如 4spk 的
        # "明明暗暗"与"和你们一起走过…"），滑窗识别可发现并细分
        subs, changed = sub_segment_identify(
            extractor, manager, samples, start_s, end_s, threshold)
        if changed:
            print(f"      ⚠️  句内检测到 {len({m[2] for m in subs})} 个说话人:")
            for st, en, nm, sc in subs:
                hit_s = f"{nm}（{sc:.3f}）" if nm else f"未知（{sc:.3f}）"
                print(f"        └ [{st:6.2f}-{en:6.2f}s] {hit_s}")
    print(f"{'='*72}")

    # 按腾讯云 speaker_id 汇总（说话人身份由声纹对照给出）
    print(f"\n📊 按腾讯云说话人（speaker_id）汇总:")
    by_spk = {}
    for r in results:
        by_spk.setdefault(r["tx_spk"], []).append(r)
    for spk, rs in sorted(by_spk.items()):
        names = {r["name"] or "未知" for r in rs}
        # 该 speaker_id 下声纹身份一致时直接显示；不一致则逐句列出
        if len(names) == 1:
            ident = names.pop()
        else:
            ident = "/".join(sorted(names))
        print(f"\n  [腾讯云spk{spk}] {len(rs)} 句 → 声纹身份: {ident}")
        for r in rs:
            print(f"    [{fmt_time(r['start'])}-{fmt_time(r['end'])}] {r['text']}")


def identify_by_local(file_path: Path, lib_dir: Path, asr_backend: str,
                      threshold: float, min_sec: float, keep_overlap: bool,
                      num_threads: int, engine_model: str):
    """
    本地 pyannote 切分模式（对照）:
      本地说话人分离 → 每区间截取 → 声纹对照特征库 → 逐段 ASR 关联文本。
    """
    manager, meta, samples = _load_audio_and_lib(file_path, lib_dir)
    duration = len(samples) / SAMPLE_RATE

    # 1) 说话人分离
    print("\n🗣️ 说话人分离中（pyannote segmentation + CAM++）...")
    t0 = time.time()
    diarizer = create_diarizer(num_threads)
    segments = diarize(diarizer, samples)
    d_elapsed = time.time() - t0
    print(f"   分离耗时 {d_elapsed:.1f}s（RTF {d_elapsed / duration:.2f}），"
          f"{len(segments)} 段 / {len(set(s for _, _, s in segments))} 个说话人")

    if keep_overlap:
        segs = list(segments)
    else:
        segs = exclude_overlap(segments)
        if len(segs) != len(segments):
            print(f"   排除重叠区间后剩余 {len(segs)} 段")
    segs = [(s, e, spk) for s, e, spk in segs if e - s >= min_sec]
    if not segs:
        print("❌ 没有可用说话人区间（全部过短）")
        return

    # 2) 声纹提取器 + 特征库
    extractor = create_speaker_extractor(num_threads)

    # 3) ASR 按需创建
    recognizer = engine = None
    if asr_backend == "tencent":
        print("\n🔊 创建腾讯云 ASR 引擎...")
        engine = make_tencent_engine(engine_model)
    elif asr_backend == "sherpa":
        print("\n🔊 创建本地 sherpa ASR 识别器...")
        recognizer = create_sherpa_recognizer(num_threads)

    print(f"\n{'='*70}")
    print(f"🎙️  区间  |  说话人（相似度）  |  识别文本")
    print(f"{'='*70}")
    results = []
    for i, (s, e, spk) in enumerate(segs, 1):
        audio = samples[int(s * SAMPLE_RATE):int(e * SAMPLE_RATE)]
        if len(audio) == 0:
            continue

        # 声纹对照
        emb = extract_embedding(extractor, audio)
        name, score = identify_speaker(manager, emb, threshold) if emb is not None else (None, 0.0)
        hit = f"{name}（{score:.3f}）" if name else f"未知（{score:.3f}）"

        # ASR
        if asr_backend == "tencent":
            text = asr_tencent_segment(engine, audio)
        elif asr_backend == "sherpa":
            text = asr_sherpa_segment(recognizer, audio)
        else:
            text = ""

        results.append({
            "start": s, "end": e, "diar_spk": spk, "name": name, "score": score,
            "text": text,
        })
        print(f"[{fmt_time(s)}-{fmt_time(e)}] {hit}")
        if text:
            print(f"      📝 {text}")
        else:
            print(f"      📝 <无识别结果>")
    print(f"{'='*70}")

    # 4) 按说话人汇总
    print(f"\n📊 按说话人汇总:")
    by_name = {}
    for r in results:
        key = r["name"] or "未知"
        by_name.setdefault(key, []).append(r)
    for name, rs in by_name.items():
        total = sum(r["end"] - r["start"] for r in rs)
        print(f"\n  [{name}] {len(rs)} 段 / {total:.1f}s")
        for r in rs:
            text = r["text"] or "<无识别结果>"
            print(f"    [{fmt_time(r['start'])}-{fmt_time(r['end'])}] {text}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    from voice_agent.config import load_config

    cfg = load_config().get("voice", {})
    df_model = cfg.get("asr", {}).get("engine_model", "16k_zh")

    p = argparse.ArgumentParser(
        description="说话人分离 + 声纹特征库对照 + ASR 关联测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument("--build-lib", action="store_true",
                   help=f"构建声纹特征库（默认来源 {DEFAULT_LIB_SRC.name}）")
    g.add_argument("--file", type=str, metavar="PATH",
                   help="要识别的 WAV（默认 tests/4spk.wav）")

    p.add_argument("--lib-dir", type=str, default=str(DEFAULT_LIB_DIR),
                   help="声纹特征库目录（默认 %(default)s）")
    p.add_argument("--lib-src", type=str, default=str(DEFAULT_LIB_SRC),
                   help="--build-lib 的来源音频（默认 %(default)s）")
    p.add_argument("--speaker-names", type=str, default="",
                   help="--build-lib 时自定义说话人名字，逗号分隔，按首次出现顺序对应")
    p.add_argument("--segment", type=str, default="tencent",
                   choices=["tencent", "local"],
                   help="切分方式: tencent=腾讯云句子段落（默认，推荐），"
                        "local=本地 pyannote 切分")
    p.add_argument("--asr", type=str, default="tencent",
                   choices=["tencent", "sherpa", "none"],
                   help="--segment local 模式下的逐段 ASR 后端（默认 %(default)s）")
    p.add_argument("--threshold", type=float, default=SPEAKER_THRESHOLD,
                   help=f"声纹匹配阈值，余弦相似度高于此值判定同一人（默认 {SPEAKER_THRESHOLD}）")
    p.add_argument("--min-sec", type=float, default=0.5,
                   help="最短语音区间（秒），更短的段跳过（默认 %(default)s）")
    p.add_argument("--keep-overlap", action="store_true",
                   help="--segment local 模式保留重叠区间（默认排除，避免声纹/文本混入他人声音）")
    p.add_argument("--num-threads", type=int, default=2, help="推理线程数（默认 %(default)s）")
    p.add_argument("--cluster-threshold", type=float, default=0.6,
                   help="--segment local 模式说话人聚类阈值（默认 %(default)s，"
                        "低于此值会把同一人拆成多人）")
    p.add_argument("--model", type=str, default=None,
                   help="腾讯云 ASR 引擎模型；tencent 模式默认 "
                        f"{TENCENT_SPEAKER_MODEL}（说话人分离引擎，句子带 speaker_id），"
                        f"local 模式默认 {df_model}（来自 config）")

    args = p.parse_args()

    if args.build_lib:
        build_voiceprint_lib(
            lib_dir=args.lib_dir,
            src_wav=Path(args.lib_src),
            speaker_names_raw=args.speaker_names,
            min_sec=args.min_sec,
            num_threads=args.num_threads,
            cluster_threshold=args.cluster_threshold,
        )
        return

    # 引擎模型默认值：tencent 段落模式必须用说话人分离引擎
    if args.model is None:
        engine_model = (TENCENT_SPEAKER_MODEL if args.segment == "tencent"
                        else df_model)
    else:
        engine_model = args.model

    # 默认识别 tests/4spk.wav（与特征库同源，可直观验证声纹对照正确性）
    file_path = Path(args.file) if args.file else DEFAULT_LIB_SRC
    if args.segment == "tencent":
        identify_by_tencent_sentences(
            file_path=file_path,
            lib_dir=Path(args.lib_dir),
            threshold=args.threshold,
            min_sec=args.min_sec,
            num_threads=args.num_threads,
            engine_model=engine_model,
        )
    else:
        identify_by_local(
            file_path=file_path,
            lib_dir=Path(args.lib_dir),
            asr_backend=args.asr,
            threshold=args.threshold,
            min_sec=args.min_sec,
            keep_overlap=args.keep_overlap,
            num_threads=args.num_threads,
            engine_model=engine_model,
        )


if __name__ == "__main__":
    main()
