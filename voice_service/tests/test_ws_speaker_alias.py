#!/usr/bin/env python3
"""
speaker_alias 端到端联调（不需要麦克风/声纹模型/腾讯云凭据）。

用真实的 `VoiceServer` + `VoiceGatewayClient` 验证工具链路的请求/应答：
seq 关联（并发不串号）、ack 带数据、绑定落盘 names.json、冲突需 overwrite、
解绑、未知 action、未启用声纹时的报错。

用法（在 voice_service/ 目录下运行）:
    python -u tests/test_ws_speaker_alias.py
"""

import asyncio
import json
import shutil
import socket
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hermes_gateway_plugin.client import VoiceGatewayClient  # noqa: E402
from voice_service import protocol as P  # noqa: E402
from voice_service.server import VoiceServer  # noqa: E402
from voice_service.service_config import ServiceConfig  # noqa: E402
from voice_service.voiceprint import SpeakerAliases  # noqa: E402


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


async def main_async() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="ws-alias-"))
    port = _free_port()
    server = VoiceServer(ServiceConfig(host="127.0.0.1", port=port))
    admin = SpeakerAliases(tmp, seed_names={"spk_100": "爸爸"})
    server.speaker_admin = admin
    await server.start()

    client = VoiceGatewayClient(f"ws://127.0.0.1:{port}", client_id="test-client")
    run_task = None
    try:
        ok = await asyncio.wait_for(client.connect(handler=lambda f: None), timeout=10)
        assert ok, "hello 握手失败"
        run_task = asyncio.create_task(
            client.reconnect_forever(handler=lambda f: None))

        def call(data: dict):
            # request() 只能在非事件循环线程调用（工具 handler 的真实情形）
            return client.request(P.CMD_SPEAKER_ALIAS, data, timeout=5.0)

        # ── 绑定（编号用消息前缀里的 "101"）──
        ack = await asyncio.to_thread(
            call, {"action": "set", "spk_id": "101", "name": "辰辰"})
        assert ack and ack["ok"], ack
        assert ack["spk_id"] == "spk_101" and ack["name"] == "辰辰", ack
        saved = json.loads((tmp / "names.json").read_text(encoding="utf-8"))
        assert saved["names"]["spk_101"] == "辰辰", saved
        assert admin.name_of("spk_101") == "辰辰"

        # ── 冲突（不带 overwrite）与覆盖 ──
        ack = await asyncio.to_thread(
            call, {"action": "set", "spk_id": "101", "name": "淘淘"})
        assert ack and ack["ok"] is False and "overwrite" in ack["error"], ack
        ack = await asyncio.to_thread(
            call, {"action": "set", "spk_id": "spk_101", "name": "淘淘",
                   "overwrite": True})
        assert ack and ack["ok"] and ack["previous"] == "辰辰", ack

        # ── 一人多编号（also_bound 回给 agent）──
        ack = await asyncio.to_thread(
            call, {"action": "set", "spk_id": "102", "name": "淘淘"})
        assert ack and ack["ok"] and ack["also_bound"] == ["spk_101"], ack

        # ── 解绑 ──
        ack = await asyncio.to_thread(call, {"action": "unset", "spk_id": "102"})
        assert ack and ack["ok"] and ack["unset"], ack
        assert admin.name_of("spk_102") is None

        # ── 并发请求：seq 不串号 ──
        running_loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=2) as pool:
            f1 = running_loop.run_in_executor(pool, call, {
                "action": "set", "spk_id": "101", "name": "小布"})
            f2 = running_loop.run_in_executor(pool, call, {
                "action": "set", "spk_id": "103", "name": "辰辰"})
            r1, r2 = await asyncio.gather(f1, f2)
        assert r1["ok"] is False and r1["previous"] == "淘淘", r1
        assert r2["ok"] and r2["spk_id"] == "spk_103" and r2["name"] == "辰辰", r2

        # ── 未知 action / 未启用声纹 ──
        ack = await asyncio.to_thread(call, {"action": "boom"})
        assert ack and ack["ok"] is False and "未知 action" in ack["error"], ack
        server.speaker_admin = None
        ack = await asyncio.to_thread(
            call, {"action": "set", "spk_id": "101", "name": "辰辰"})
        assert ack and ack["ok"] is False and "未启用" in ack["error"], ack

        print("speaker_alias WS 联调通过 ✅")
    finally:
        if run_task is not None:
            run_task.cancel()
        await client.close()
        await server.stop()
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(main_async())
