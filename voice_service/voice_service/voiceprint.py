"""
说话人识别模块 — VoiceprintManager。

职责（作为语音助手主程序的并行旁路，不阻塞 ASR/Agent 主链路）:
  - 加载 CAM++ 说话人 embedding 模型（sherpa-onnx SpeakerEmbeddingExtractor）
  - 加载/维护声纹特征库（embeddings.npz + meta.json，格式与
    tests/test_speaker_identify.py 一致，两处可互换）
  - identify(samples): 提取声纹并与特征库比对，返回 (说话人id, 相似度)
  - register(samples): 新说话人自动分配 spk_<N>（从 100 起）注册并持久化
  - display_name(id): 通过 config 的 speaker_names 映射显示真实身份
    （管理员在 config.yaml 把默认 id 改为真实身份，无需重建特征库）

特征库格式:
  lib_dir/embeddings.npz   每个说话人一个数组（N 段 × 192 维）
  lib_dir/meta.json        来源、模型、维度、阈值、说话人列表、创建时间
  写入采用"临时文件 + rename"原子替换，避免写坏库。
"""

import json
import logging
import threading
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_RATE = 16000

# CAM++ 声纹提取模型（192 维说话人 embedding）
EMB_MODEL = PROJECT_ROOT / "models" / "sherpa-spk" / "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx"

# 自动注册的默认 id 起始编号（与测试库 spk_0~N 区分）
DEFAULT_ID_START = 100


def extract_embedding(extractor, samples: np.ndarray):
    """
    对一段音频（16k float32）提取 CAM++ 声纹 embedding，返回 1-D float32 数组。
    音频太短（不足 0.2s）或模型不可用时返回 None。
    """
    if len(samples) < int(0.2 * SAMPLE_RATE):
        return None
    stream = extractor.create_stream()
    stream.accept_waveform(sample_rate=SAMPLE_RATE, waveform=samples)
    stream.input_finished()
    if not extractor.is_ready(stream):
        return None
    return np.array(extractor.compute(stream))


class VoiceprintManager:
    """声纹特征库管理与说话人识别。线程安全（识别/注册可并发调用）。"""

    def __init__(self, lib_dir: str | Path, threshold: float = 0.6,
                 speaker_names: dict | None = None):
        self.lib_dir = Path(lib_dir)
        self.threshold = float(threshold)
        self.speaker_names = dict(speaker_names or {})
        self._lock = threading.Lock()

        # CAM++ 提取器
        self.extractor = self._create_extractor()

        # 特征库（SpeakerEmbeddingManager + 内存数据镜像用于持久化）
        import sherpa_onnx
        self._manager = sherpa_onnx.SpeakerEmbeddingManager(self.extractor.dim)
        self._data: dict[str, np.ndarray] = {}  # name -> (N, dim)
        self._load_lib()

        logger.info("VoiceprintManager 就绪: lib=%s, threshold=%.2f, "
                    "已注册 %d 人 (%s)", self.lib_dir, self.threshold,
                    len(self._data), ", ".join(self._data.keys()) or "空库")

    # ─── 模型与特征库加载 ────────────────────────────────

    def _create_extractor(self):
        """创建 CAM++ SpeakerEmbeddingExtractor；模型缺失时返回 None（功能降级）。"""
        import sherpa_onnx

        if not EMB_MODEL.is_file():
            logger.error("CAM++ 声纹模型缺失: %s（说话人识别不可用）", EMB_MODEL)
            return None
        config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(EMB_MODEL), num_threads=2, debug=False, provider="cpu",
        )
        if not config.validate():
            logger.error("CAM++ 声纹模型配置无效（说话人识别不可用）")
            return None
        extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
        logger.info("CAM++ 声纹模型加载成功（embedding 维度 %d）", extractor.dim)
        return extractor

    def _load_lib(self):
        """从磁盘加载特征库；目录不存在或为空时按空库处理（首次运行自动建库）。"""
        self.lib_dir.mkdir(parents=True, exist_ok=True)
        meta_path = self.lib_dir / "meta.json"
        npz_path = self.lib_dir / "embeddings.npz"
        if not meta_path.is_file() or not npz_path.is_file():
            logger.info("声纹特征库不存在（%s），从空库开始（新说话人将自动注册）",
                        self.lib_dir)
            return
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            data = np.load(npz_path)
            for name in meta.get("speakers", []):
                embs = np.asarray(data[name])
                if embs.ndim == 1:
                    embs = embs.reshape(1, -1)
                if self._manager.add(name, [embs[i] for i in range(len(embs))]):
                    self._data[name] = embs
        except Exception as e:
            logger.error("加载声纹特征库失败（%s）: %s", self.lib_dir, e)

    # ─── 识别 ────────────────────────────────────────────

    @property
    def num_speakers(self) -> int:
        return len(self._data)

    @property
    def speakers(self) -> list[str]:
        return list(self._data.keys())

    def identify(self, samples: np.ndarray) -> tuple[str | None, float]:
        """
        对一段音频做声纹识别，返回 (说话人id, 相似度)。
        未匹配（相似度低于 threshold）或无法提取声纹时 id 为 None。
        """
        if self.extractor is None:
            return None, 0.0
        emb = extract_embedding(self.extractor, samples)
        if emb is None:
            return None, 0.0
        with self._lock:
            best_name, best_score = None, -1.0
            for name in self._manager.all_speakers:
                score = self._manager.score(name, emb)
                if score > best_score:
                    best_name, best_score = name, score
        if best_score < self.threshold:
            return None, float(best_score)
        return best_name, float(best_score)

    # ─── 注册 ────────────────────────────────────────────

    def register(self, samples: np.ndarray) -> str | None:
        """
        注册新说话人：自动分配 spk_<N>（从 100 起递增），用该段音频的声纹
        入库并持久化。成功返回分配的 id；失败（音频过短/模型不可用）返回 None。
        """
        if self.extractor is None:
            return None
        emb = extract_embedding(self.extractor, samples)
        if emb is None:
            logger.warning("声纹注册失败: 音频过短（%.2fs）",
                           len(samples) / SAMPLE_RATE)
            return None
        with self._lock:
            new_id = self._next_id()
            if not self._manager.add(new_id, emb):
                logger.warning("声纹注册失败: %s 加入特征库失败", new_id)
                return None
            self._data[new_id] = np.array([emb])
            self._save()
        logger.info("✅ 已注册新说话人 %s", new_id)
        return new_id

    def _next_id(self) -> str:
        """分配下一个默认 id（spk_<N>，N ≥ DEFAULT_ID_START 且递增）。"""
        max_n = DEFAULT_ID_START - 1
        for name in self._data:
            if name.startswith("spk_") and name[4:].isdigit():
                max_n = max(max_n, int(name[4:]))
        return f"spk_{max_n + 1}"

    def _save(self):
        """原子持久化特征库（临时文件 + rename，避免写坏库）。"""
        try:
            self.lib_dir.mkdir(parents=True, exist_ok=True)
            # 注意 np.savez 会自动追加 .npz 后缀，临时文件名需直接以 .npz 结尾
            npz_tmp = self.lib_dir / "embeddings_tmp.npz"
            np.savez(npz_tmp, **{k: np.asarray(v) for k, v in self._data.items()})
            npz_tmp.replace(self.lib_dir / "embeddings.npz")

            meta = {
                "embedding_model": EMB_MODEL.name,
                "dim": int(self.extractor.dim),
                "threshold": self.threshold,
                "speakers": list(self._data.keys()),
                "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            meta_tmp = self.lib_dir / "meta.json.tmp"
            meta_tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                encoding="utf-8")
            meta_tmp.replace(self.lib_dir / "meta.json")
        except Exception as e:
            logger.error("声纹特征库持久化失败: %s", e)

    # ─── 显示 ────────────────────────────────────────────

    def display_name(self, spk_id: str) -> str:
        """管理员配置映射（speaker_names）优先，无映射时显示默认 id。"""
        return self.speaker_names.get(spk_id, spk_id)
