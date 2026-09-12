"""
说话人识别模块 — VoiceprintManager。

职责（作为语音助手主程序的并行旁路，不阻塞 ASR/Agent 主链路）:
  - 加载 CAM++ 说话人 embedding 模型（sherpa-onnx SpeakerEmbeddingExtractor）
  - 加载/维护声纹特征库（embeddings.npz + meta.json，格式与
    tests/test_speaker_identify.py 一致，两处可互换）
  - identify(samples): 提取声纹并与特征库比对，返回 (说话人id, 相似度)
  - register(samples): 新说话人自动分配 spk_<N>（从 100 起）注册并持久化
  - display_name(id): 说话人显示名 —— config 的 speaker_names（人工 seed）
    与运行时绑定 names.json（agent 工具写入）合并，后者优先
  - speaker_label(id): 拼上行给 agent 的说话人标签
    （`爸爸 (ID: 100)` / `未知 (ID: 101)` / `未知`，见 PROTOCOL.md §5）

特征库格式:
  lib_dir/embeddings.npz   每个说话人一个数组（N 段 × 192 维）
  lib_dir/meta.json        来源、模型、维度、阈值、说话人列表、创建时间
  lib_dir/names.json       运行时身份绑定（agent 写入；人工也可手改）
  写入采用"临时文件 + rename"原子替换，避免写坏库。
"""

import json
import logging
import threading
import time
from pathlib import Path

import numpy as np

from . import protocol as P

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_RATE = 16000

# CAM++ 声纹提取模型（192 维说话人 embedding）
EMB_MODEL = PROJECT_ROOT / "models" / "sherpa-spk" / "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx"

# 自动注册的默认 id 起始编号（与测试库 spk_0~N 区分）
DEFAULT_ID_START = 100

# 未识别 / 未映射说话人的显示名
UNKNOWN_SPEAKER = "未知"

# 绑定名字长度上限（防模型塞长句当名字）
ALIAS_NAME_MAX = 16


def is_default_id(spk_id: str) -> bool:
    """是否为自动注册的默认 id（``spk_<N>``）。非默认 id 视为"库内直接用了名字"。"""
    return P.speaker_number(spk_id) != spk_id


class SpeakerAliases:
    """说话人身份绑定（`lib_dir/names.json`）。

    两层映射，**运行时优先**：
    - seed：config ``voiceprint.speaker_names``，人工维护的初始值（只读）；
    - runtime：本文件，agent 通过 ``speaker_alias`` 帧写入（可覆盖 seed）。

    这样 agent 记住的身份与人工维护的映射共存：names.json 是"最近一次用户亲口确认"，
    人工要改就直接编辑同一文件或 config，重启即生效。线程安全、原子写。
    """

    def __init__(self, lib_dir: str | Path, seed_names: dict | None = None,
                 known_ids=None):
        self.lib_dir = Path(lib_dir)
        self.path = self.lib_dir / "names.json"
        self._seed = {str(k): str(v) for k, v in (seed_names or {}).items()}
        # 返回"声纹库里实际存在的编号"的可调用对象（VoiceprintManager 注入）；
        # 用于挡住模型编造的编号。None = 不校验（单测/无库场景）。
        self._known_ids = known_ids
        self._lock = threading.Lock()
        self._names: dict[str, str] = {}
        self._load()

    # ─── 读写 ────────────────────────────────────────────

    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            names = raw.get("names") if isinstance(raw, dict) else None
            if isinstance(names, dict):
                self._names = {str(k): str(v) for k, v in names.items() if v}
        except Exception as exc:
            logger.error("说话人绑定文件损坏（%s）：%s", self.path, exc)

    def _save(self) -> None:
        """原子持久化（临时文件 + rename）。"""
        try:
            self.lib_dir.mkdir(parents=True, exist_ok=True)
            payload = {"names": dict(self._names),
                       "updated": time.strftime("%Y-%m-%d %H:%M:%S")}
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                           encoding="utf-8")
            tmp.replace(self.path)
        except Exception as exc:
            logger.error("说话人绑定持久化失败: %s", exc)

    @property
    def names(self) -> dict[str, str]:
        """合并视图（seed + runtime），运行时优先。"""
        return {**self._seed, **self._names}

    def name_of(self, spk_id: str | None) -> str | None:
        if not spk_id:
            return None
        return self.names.get(str(spk_id))

    def is_bound(self, spk_id: str | None) -> bool:
        return self.name_of(spk_id) is not None

    # ─── 绑定操作（返回结果 dict，不抛业务异常）───────────

    def set_alias(self, spk_id: str, name: str,
                  *, overwrite: bool = False) -> dict:
        """绑定 ``spk_id`` → ``name``。

        - 未绑定 → 直接绑定；
        - 已绑定同名 → 幂等成功（``unchanged=True``）；
        - 已绑定异名且 ``overwrite=False`` → ``ok=False``，提示确认后带 overwrite 重试；
        - ``overwrite=True`` → 覆盖并回报 ``previous``。
        同一个人允许多个编号（``also_bound`` 列出该名字的其它编号）。
        """
        sid = P.normalize_spk_id(spk_id)
        if not sid:
            return {"ok": False, "error": f"非法说话人编号: {spk_id!r}"}
        if self._known_ids is not None:
            try:
                known = set(self._known_ids())
            except Exception:
                known = set()
            if known and sid not in known:
                return {"ok": False, "spk_id": sid,
                        "error": f"声纹库中没有编号 {sid}（未注册或仍在识别中）"}
        clean = str(name or "").strip()
        if not clean:
            return {"ok": False, "error": "名字不能为空"}
        if len(clean) > ALIAS_NAME_MAX:
            return {"ok": False,
                    "error": f"名字过长（最多 {ALIAS_NAME_MAX} 字）: {clean!r}"}
        if any(c in clean for c in "\r\n[]") or clean == UNKNOWN_SPEAKER:
            return {"ok": False, "error": f"名字不合法: {clean!r}"}

        with self._lock:
            current = self.name_of(sid)
            also_bound = sorted(k for k, v in self.names.items()
                                if v == clean and k != sid)
            if current == clean:
                return {"ok": True, "unchanged": True, "spk_id": sid,
                        "name": clean, "also_bound": also_bound}
            if current and not overwrite:
                return {"ok": False, "spk_id": sid, "previous": current,
                        "error": (f"{sid} 当前绑定为「{current}」；"
                                  f"如已向用户确认要更正，请带 overwrite=true 重试")}
            self._names[sid] = clean
            self._save()
        logger.info("说话人绑定: %s → %s（previous=%s, also_bound=%s）",
                    sid, clean, current, also_bound)
        return {"ok": True, "spk_id": sid, "name": clean,
                "previous": current, "also_bound": also_bound}

    def unset_alias(self, spk_id: str) -> dict:
        """解除运行时绑定；来自 config seed 的绑定无法在此删除（提示去 config）。"""
        sid = P.normalize_spk_id(spk_id)
        if not sid:
            return {"ok": False, "error": f"非法说话人编号: {spk_id!r}"}
        with self._lock:
            removed = self._names.pop(sid, None)
            if removed:
                self._save()
            seed_hit = sid in self._seed
        logger.info("说话人解绑: %s（runtime=%s, seed=%s）", sid, removed, seed_hit)
        if not removed and not seed_hit:
            return {"ok": True, "spk_id": sid, "unset": False, "removed": None,
                    "seed": False, "note": "该编号没有绑定"}
        return {"ok": True, "spk_id": sid, "unset": bool(removed),
                "removed": removed, "seed": seed_hit,
                "note": ("该绑定来自 config.yaml 的 speaker_names，"
                         "需在配置里删除" if seed_hit and not removed else "")}


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
                 speaker_names: dict | None = None,
                 aliases: "SpeakerAliases | None" = None):
        self.lib_dir = Path(lib_dir)
        self.threshold = float(threshold)
        self.speaker_names = dict(speaker_names or {})
        # 身份绑定（config seed + 运行时 names.json），启动时先建好目录；
        # known_ids 让绑定只能落在声纹库真实存在的编号上
        self.lib_dir.mkdir(parents=True, exist_ok=True)
        self.aliases = aliases or SpeakerAliases(
            self.lib_dir, self.speaker_names,
            known_ids=lambda: list(self._data.keys()))
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
        """说话人展示名：绑定映射（names.json > config）优先。

        未绑定的自动注册 id（``spk_<N>``）→ ``未知``（编号另行拼在标签里）；
        调试库把名字直接当 id 用时原样显示。
        """
        if not spk_id:
            return UNKNOWN_SPEAKER
        bound = self.aliases.name_of(spk_id)
        if bound:
            return bound
        return UNKNOWN_SPEAKER if is_default_id(spk_id) else str(spk_id)

    def speaker_label(self, spk_id: str | None) -> str:
        """上行给 agent 的说话人标签（PROTOCOL.md §5）。

        格式：``爸爸 (ID: 100)`` / ``未知 (ID: 101)`` / ``未知``（无编号时）。
        编号是声纹库 id 的数字部分 —— 管理员后来在 config/names.json 里把 ``spk_101``
        映射成真实姓名，编号也不变，agent 因此能跨轮次稳定地区分同一个人。
        """
        if not spk_id:
            return UNKNOWN_SPEAKER
        name = self.display_name(spk_id)
        if not is_default_id(spk_id):
            return name            # 非 spk_<N>（调试库用名字当 id）：只有名字
        return f"{name} (ID: {P.speaker_number(spk_id)})"
