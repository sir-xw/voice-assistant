#!/usr/bin/env python3
"""
说话人身份绑定（SpeakerAliases）单测 —— 不依赖声纹模型与音频。

覆盖：绑定/幂等/冲突/覆盖/一人多编号/解绑/入参校验/持久化/seed 与运行时优先级，
以及 `VoiceprintManager.speaker_label` 的标签格式（绕过模型加载）。

用法（在 voice_service/ 目录下运行）:
    python -u tests/test_speaker_alias.py
"""

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voice_service.voiceprint import (  # noqa: E402
    UNKNOWN_SPEAKER, SpeakerAliases, VoiceprintManager,
)


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="vp-alias-"))
    try:
        # ── seed（config speaker_names）──
        a = SpeakerAliases(tmp, seed_names={"spk_100": "爸爸", "spk_50": "奶奶"})
        assert a.name_of("spk_100") == "爸爸"
        assert a.name_of("spk_101") is None
        assert not a.path.exists(), "尚未绑定不应创建 names.json"

        # ── 绑定新编号（工具传来的可能是 "101" / "spk_101"）──
        r = a.set_alias("101", "辰辰")
        assert r["ok"] and r["spk_id"] == "spk_101" and r["previous"] is None, r
        assert a.path.is_file()
        # 同名幂等
        r = a.set_alias("spk_101", "辰辰")
        assert r["ok"] and r.get("unchanged"), r
        # 异名冲突：不覆盖，提示带 overwrite 重试
        r = a.set_alias("spk_101", "淘淘")
        assert not r["ok"] and r["previous"] == "辰辰" and "overwrite" in r["error"], r
        # 覆盖（用户更正身份）
        r = a.set_alias("spk_101", "淘淘", overwrite=True)
        assert r["ok"] and r["previous"] == "辰辰", r
        # 运行时覆盖 config seed
        r = a.set_alias("spk_100", "妈妈", overwrite=True)
        assert r["ok"] and r["previous"] == "爸爸", r
        assert a.name_of("spk_100") == "妈妈"
        # 一人多编号
        r = a.set_alias("102", "淘淘")
        assert r["ok"] and r["also_bound"] == ["spk_101"], r

        # ── 入参校验 ──
        for spk, name in [("", "x"), ("spk_x", "x"), (None, "x"), ("101", ""),
                          ("101", "x" * 17), ("101", "未知"), ("101", "[a]"),
                          ("101", "a\nb")]:
            assert not a.set_alias(spk, name)["ok"], (spk, name)

        # ── 持久化 + 重载（运行时优先于 seed）──
        b = SpeakerAliases(tmp, seed_names={"spk_100": "爸爸"})
        assert b.name_of("spk_101") == "淘淘", b.names
        assert b.name_of("spk_100") == "妈妈", b.names

        # ── 解绑 ──
        r = a.unset_alias("102")
        assert r["ok"] and r["unset"] and a.name_of("spk_102") is None, r
        # 覆盖过 seed 的编号：删掉运行时覆盖 → 回落到 seed 的名字
        r = a.unset_alias("100")
        assert r["ok"] and r["unset"] and r["seed"] and r["removed"] == "妈妈", r
        assert a.name_of("spk_100") == "爸爸"
        # 只存在于 seed 的编号：此处删不掉，需改 config
        r = a.unset_alias("spk_50")
        assert r["ok"] and not r["unset"] and r["seed"], r
        # 从未绑定的编号
        r = a.unset_alias("spk_77")
        assert r["ok"] and not r["unset"] and not r["seed"], r

        # ── 文件损坏时降级（不丢 seed，不崩）──
        (tmp / "names.json").write_text("{ 坏 json", encoding="utf-8")
        c = SpeakerAliases(tmp, seed_names={"spk_100": "爸爸"})
        assert c.name_of("spk_100") == "爸爸" and c.name_of("spk_101") is None

        # ── 编号必须真实存在于声纹库（VoiceprintManager 注入 known_ids）──
        guard = SpeakerAliases(tmp / "guard",
                               known_ids=lambda: ["spk_100", "spk_101"])
        r = guard.set_alias("109", "辰辰")
        assert not r["ok"] and "声纹库中没有编号" in r["error"], r
        assert guard.set_alias("101", "辰辰")["ok"], "库中已有的编号应可绑定"

        # ── 标签格式（不加载 CAM++ 模型）──
        vp = VoiceprintManager.__new__(VoiceprintManager)
        vp.aliases = SpeakerAliases(tmp, seed_names={"spk_100": "爸爸"})
        assert vp.speaker_label("spk_100") == "爸爸 (ID: 100)"
        assert vp.speaker_label("spk_101") == f"{UNKNOWN_SPEAKER} (ID: 101)"
        assert vp.speaker_label(None) == "未知"
        assert vp.speaker_label("") == "未知"
        # 调试库把名字当 id：只有名字、不带编号
        assert vp.display_name("custom-name") == "custom-name"
        assert vp.speaker_label("custom-name") == "custom-name"

        print("说话人身份绑定单测通过 ✅")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
