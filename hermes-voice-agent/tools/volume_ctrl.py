#!/usr/bin/env python3
"""
USB 音箱音量控制（触摸按键松手后执行）
监听按键事件，在连续触摸停止 50ms 后触发一次音量调节
"""

import os
import sys
import time
import threading
import subprocess
from evdev import InputDevice, ecodes, list_devices

# ========== 配置 ==========
DEVICE_NAME_SUBSTRING = "AIMIC-M4"   # 设备名称关键字
VOLUME_STEP = 0.02                    # 每次调节数值, 1为满音量
DELAY_SEC = 0.05                     # 松手后等待秒数（50ms）
AUDIO_SERVER = "pipewire"            # "pulse" 或 "pipewire"

# ========== 音量控制命令 ==========
if AUDIO_SERVER == "pulse":
    VOL_UP_CMD = ["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"+{VOLUME_STEP}%"]
    VOL_DOWN_CMD = ["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"-{VOLUME_STEP}%"]
elif AUDIO_SERVER == "pipewire":
    VOL_UP_CMD = ["wpctl", "set-volume", "@DEFAULT_SINK@", f"{VOLUME_STEP}+"]
    VOL_DOWN_CMD = ["wpctl", "set-volume", "@DEFAULT_SINK@", f"{VOLUME_STEP}-"]
else:
    raise ValueError("AUDIO_SERVER must be 'pulse' or 'pipewire'")

# ========== 查找设备 ==========
def find_device(partial_name):
    for path in list_devices():
        try:
            dev = InputDevice(path)
            if partial_name.lower() in dev.name.lower():
                return dev
        except (PermissionError, OSError):
            continue
    return None

# ========== 音量控制器 ==========
class VolumeController:
    def __init__(self, delay_sec):
        self.delay = delay_sec
        self.timers = {}  # key_code -> threading.Timer
        self.lock = threading.Lock()

    def _execute(self, key_code, cmd):
        """实际执行音量调节"""
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"触发: {cmd}")
        except Exception as e:
            print(f"执行命令失败: {e}", file=sys.stderr)

    def _schedule(self, key_code, cmd):
        """取消旧定时器，创建新定时器"""
        with self.lock:
            # 取消已存在的定时器
            if key_code in self.timers:
                self.timers[key_code].cancel()
            # 创建新定时器
            t = threading.Timer(self.delay, self._execute, args=(key_code, cmd))
            self.timers[key_code] = t
            t.start()

    def on_event(self, key_code, cmd):
        """当收到按键事件时调用，重置定时器"""
        # 只处理音量加、减、静音
        self._schedule(key_code, cmd)

    def cancel_all(self):
        """退出时取消所有定时器"""
        with self.lock:
            for t in self.timers.values():
                t.cancel()
            self.timers.clear()

# ========== 主循环 ==========
def main():
    dev = find_device(DEVICE_NAME_SUBSTRING)
    if dev is None:
        print(f"未找到包含 '{DEVICE_NAME_SUBSTRING}' 的输入设备", file=sys.stderr)
        print("可用设备列表：")
        for path in list_devices():
            try:
                d = InputDevice(path)
                print(f"  {d.path}: {d.name}")
            except:
                pass
        sys.exit(1)

    print(f"监听设备: {dev.path} ({dev.name})")
    controller = VolumeController(DELAY_SEC)

    # 按键映射
    KEY_MAP = {
        ecodes.KEY_VOLUMEUP: VOL_UP_CMD,
        ecodes.KEY_VOLUMEDOWN: VOL_DOWN_CMD,
    }

    try:
        for event in dev.read_loop():
            if event.type == ecodes.EV_KEY and event.value == 1:  # 按下事件
                key_code = event.code
                if key_code in KEY_MAP:
                    controller.on_event(key_code, KEY_MAP[key_code])
    except KeyboardInterrupt:
        print("\n退出")
    finally:
        controller.cancel_all()

if __name__ == "__main__":
    main()
