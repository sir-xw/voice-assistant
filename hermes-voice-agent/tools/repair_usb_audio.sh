#!/bin/bash
# ============================================================
# repair_usb_audio.sh —— 修复 QEMU usb-host 透传 USB 音箱的音频链路
#
# 背景：本机是 PVE/KVM 虚拟机，USB 音箱（AIMIC-M4）通过 QEMU
#       usb-host 透传进入。长时间运行后 QEMU 虚拟 USB 控制器的
#       isochronous（等时）传输状态会积累异常，表现为：
#       「播放时完全听不到音乐音调，只有持续低噪 + 偶尔爆音」。
#       重启系统有效，是因为虚拟 USB 控制器被完全重建。
#
# 本脚本模拟"重启系统"中 USB 相关的重建环节（不重启系统）：
#   卸载并重载 xhci_pci 驱动 → 虚拟 USB 控制器重新初始化 →
#   usb-host 后端重建 isoc 传输通道。已验证有效。
#
# 经实测（2026-08-15）：重建时【无需停止 PipeWire】——
#   - pipewire/wireplumber 全程存活，sink 由 wireplumber 自动重建
#   - 正在播放的 MPD 音乐会自动续播
#   因此脚本不再操作 pipewire 服务，修复更快速、影响更小。
#
# 副作用：USB 音箱和虚拟 USB 鼠标会短暂断开重连（约 5 秒），
#         正在播放的音频会有几秒中断后自动恢复。
# ============================================================

log() { echo "[$(date '+%F %T')] $*"; }

log "=== 开始修复：重建虚拟 USB 控制器（xhci_pci）==="

# 1. 重建控制器驱动（xhci_pci_renesas 无需手动处理，卸载时自动释放引用）
if ! modprobe -r xhci_pci 2>/tmp/repair_unload.err; then
    log "错误：卸载 xhci_pci 失败：$(cat /tmp/repair_unload.err)"
    exit 1
fi
log "xhci_pci 已卸载"

if ! modprobe xhci_pci 2>/tmp/repair_load.err; then
    log "错误：加载 xhci_pci 失败：$(cat /tmp/repair_load.err)"
    exit 1
fi
log "xhci_pci 已重新加载，等待设备重新枚举..."
sleep 3

# 2. 确认声卡恢复
if grep -q "AIMICM4" /proc/asound/cards 2>/dev/null; then
    log "声卡已恢复：$(grep AIMICM4 /proc/asound/cards)"
else
    log "警告：未检测到 AIMIC-M4 声卡，可能修复失败"
    exit 2
fi

# 3. 确认 sink 恢复（wireplumber 自动重建，最多等 10 秒）
for i in $(seq 1 10); do
    if pactl list sinks short 2>/dev/null | grep -q "usb.*AIMIC"; then
        break
    fi
    sleep 1
done
if pactl list sinks short 2>/dev/null | grep -q "usb.*AIMIC"; then
    log "sink 已恢复：$(pactl get-default-sink 2>/dev/null)"
else
    log "警告：sink 未恢复，请检查 wireplumber"
fi

# 4. 播放测试音验证
if timeout 3 pw-play /tmp/usb_audio_health_tone.wav >/dev/null 2>&1; then
    log "测试音播放正常，修复完成 ✅"
    exit 0
else
    log "警告：测试音播放异常（可能设备仍在初始化），请人工确认"
    exit 2
fi
