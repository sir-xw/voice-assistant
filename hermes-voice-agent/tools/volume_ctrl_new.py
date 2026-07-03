#!/usr/bin/env python3
"""
USB 音箱音量控制（触摸按键松手后执行）
监听按键事件，在连续触摸停止 50ms 后触发一次音量调节
"""

import sys
import threading
import subprocess
import re
from evdev import InputDevice, ecodes, list_devices

# ========== 配置 ==========
DEVICE_NAME_SUBSTRING = "AIMIC-M4"   # 设备名称关键字
DELAY_SEC = 0.05                     # 松手后等待秒数（50ms）


# 人工测试出的设备亮灯音量阈值
# 5灯：pcm 90 wpctl 0.94
# 4灯：pcm 81 wpctl 0.88
# 3灯：pcm 69， wpctl 0.78
# 2灯：pcm 47, wpctl 0.6
# 1灯：pcm 9 wpctl 0.1


# ========== 音量控制命令 ==========
VOLUME_LEVELS = [0, 0.1, 0.4, 0.6, 0.69, 0.78, 0.83, 0.88, 0.91, 0.94, 1]

def get_current_volume():
    """读取当前默认 sink 的音量 (0.0~1.0)"""
    try:
        output = subprocess.check_output(
            ["wpctl", "get-volume", "@DEFAULT_SINK@"],
            stderr=subprocess.DEVNULL,
            text=True
        )
        match = re.search(r"Volume:\s*([\d.]+)", output)
        if match:
            return float(match.group(1))
    except Exception:
        pass
    # 如果读取失败，返回第5个档位的音量值
    return VOLUME_LEVELS[5]

def find_closest_index(volume, levels):
    """在 levels 列表中找到最接近 volume 的值的索引"""
    if not levels:
        return 0
    # 使用 abs 和 min 找到最小差值对应的索引
    closest_idx = min(range(len(levels)), key=lambda i: abs(levels[i] - volume))
    return closest_idx

# 在程序启动时调用
def init_volume_index():
    current = get_current_volume()
    idx = find_closest_index(current, VOLUME_LEVELS)
    print(f"当前音量: {current:.2f}, 匹配到档位 {idx}: {VOLUME_LEVELS[idx]:.2f}")
    return idx

# 全局变量
CURRENT_IDX = init_volume_index()


def set_volume_from_idx(idx):
    val = VOLUME_LEVELS[idx]
    try:
        subprocess.run(["wpctl", "set-volume", "@DEFAULT_SINK@", str(val)], check=True, capture_output=True)
        print(f"set vol: {val}")
    except Exception as e:
        print(f"执行命令失败: {e}", file=sys.stderr)


# 按键处理
def change_volume(direction):
    global CURRENT_IDX
    if direction == "up" and CURRENT_IDX < len(VOLUME_LEVELS) - 1:
        CURRENT_IDX += 1
        print(f'触发音量提高至 {CURRENT_IDX} 档')
    elif direction == "down" and CURRENT_IDX > 0:
        CURRENT_IDX -= 1
        print(f'触发音量降低至 {CURRENT_IDX} 档')
    else:
        return
    set_volume_from_idx(CURRENT_IDX)


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

    def _schedule(self, key_code, direction):
        """取消旧定时器，创建新定时器"""
        with self.lock:
            # 取消已存在的定时器
            if key_code in self.timers:
                self.timers[key_code].cancel()
            # 创建新定时器
            t = threading.Timer(self.delay, change_volume, args=(direction,))
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
        ecodes.KEY_VOLUMEUP: 'up',
        ecodes.KEY_VOLUMEDOWN: 'down',
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
