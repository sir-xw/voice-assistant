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
# 实测记录（2026-09-02/03）：
#   1. 仅重载 xhci_pci 后 sink（播放）会自动恢复，但 source（麦克风采集）
#      会进入"假 RUNNING 无数据"状态 —— ALSA capture DMA 冻结、PulseAudio
#      客户端收不到任何 PCM。重启 wireplumber 强制重新配置设备节点后采集
#      才恢复（节点激活约需 20-45 秒，不稳定，偶尔需重启两次）。
#   2. voice-service（语音服务进程）的采集流会占用 ALSA 麦克风，导致
#      wireplumber 重启时节点无法激活（pending linkable）。因此脚本改为
#      【先停止 voice-service.service 让出设备】：进程退出后声卡句柄全部
#      释放，wireplumber 可正常重建节点；待音频栈重建与采集验证全部通过
#      后，脚本末尾再重新启动 voice-service。相比旧方案（运行中写暂停标记
#      /run/user/<uid>/hermes_voice_mic.pause 通知其主动关流）更干净：
#      无暂停标记过期/遗留竞态，进程本身也以全新状态重建音频流。
#   3. wireplumber 重启后 pipewire-pulse 偶发挂起，pactl/parec 会无限阻塞
#      （实测脚本卡死 12 分钟）。因此本脚本对全部 pulse 命令加 timeout，
#      并在 pactl 无响应时自动重启 pipewire-pulse —— 保证脚本总时长有界
#      （约 2-3 分钟）。
#   4. voice-service 停止/启动期间 hermes-gateway 无需任何处理：其 voice
#      平台是 Voice Service 的 WS 客户端，语音服务恢复后会自动重连
#      （voice_service 重启会短暂踢掉 gateway 的旧连接，属预期）。
#
# 副作用：USB 音箱和虚拟 USB 鼠标会短暂断开重连（约 5 秒），
#         正在播放的音频会有几秒中断后自动恢复。
# ============================================================

log() { echo "[$(date '+%F %T')] $*"; }

export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"

# --- PulseAudio 命令统一加 timeout 防卡（见顶部实测记录 3）---
PACTL_BIN=$(command -v pactl)
PAREC_BIN=$(command -v parec)
pactl() { timeout 5 "$PACTL_BIN" "$@"; }
parec() { timeout 4 "$PAREC_BIN" "$@"; }

# pipewire-pulse 挂起自愈：pactl 不响应 → 重启 pipewire-pulse
ensure_pulse_alive() {
    if ! pactl info >/dev/null 2>&1; then
        log "pactl 无响应（pipewire-pulse 可能挂起），重启 pipewire-pulse..."
        if systemctl --user restart pipewire-pulse >/dev/null 2>&1; then
            log "pipewire-pulse 已重启"
        else
            log "警告：pipewire-pulse 重启失败"
        fi
        sleep 3
    fi
}

# 定位 AIMIC-M4 的 USB 设备路径（如 9-1），未找到返回空
find_aimic_usb_dev() {
    for d in /sys/bus/usb/devices/*/; do
        if [ "$(cat "$d/idVendor" 2>/dev/null)" = "e2b7" ]; then
            basename "$d"
            return 0
        fi
    done
    return 1
}

# 采集假死兜底：USB 设备 unbind/rebind 复位 + 重启 wireplumber。
# 实测（2026-09-03）：xhci_pci 全重载后采集经常"假 RUNNING 无数据"且激活
# 极不稳定（20s~数分钟），而设备级 unbind/rebind 约 6 秒即可恢复采集，
# 且不影响其它 USB 设备，作为采集轮询失败后的二次恢复手段。
rebind_aimic() {
    local dev
    dev=$(find_aimic_usb_dev)
    if [ -z "$dev" ]; then
        log "警告：未找到 AIMIC USB 设备，跳过 unbind/rebind"
        return 1
    fi
    log "USB 设备级复位 $dev（unbind/rebind）..."
    if echo "$dev" > /sys/bus/usb/drivers/usb/unbind 2>/dev/null; then
        log "$dev 已 unbind"
    else
        log "警告：$dev unbind 失败（可能已断开）"
    fi
    sleep 3
    echo "$dev" > /sys/bus/usb/drivers/usb/bind 2>/dev/null && log "$dev 已 rebind"
    sleep 5
    systemctl --user restart wireplumber >/dev/null 2>&1 || true
    log "wireplumber 已重启（设备复位后重建节点）"
    sleep 3
    ensure_pulse_alive
}

log "=== 开始修复：停止 voice-service → 重建虚拟 USB 控制器（xhci_pci）+ 重启 wireplumber → 启动 voice-service ==="

# 0. 停止 voice-service，让出 ALSA 麦克风：进程退出后声卡句柄全部释放，
#    wireplumber 重启时节点才能正常激活。
#    hermes-gateway 无需处理：其 voice 平台是 Voice Service 的 WS 客户端，
#    语音服务停止/恢复期间会自动断线重连。
VS_RUNNING=0
if systemctl --user is-active voice-service.service >/dev/null 2>&1; then
    VS_RUNNING=1
    log "停止 voice-service（voice-service.service）以释放采集设备..."
    if ! timeout 120 systemctl --user stop voice-service.service 2>/tmp/repair_vs_stop.err; then
        log "警告：voice-service 停止失败或超时：$(cat /tmp/repair_vs_stop.err)"
    fi
    # 等待采集流从 pulse 消失（最多 15 秒），确认设备已让出
    for i in $(seq 1 15); do
        ensure_pulse_alive
        if ! pactl list source-outputs 2>/dev/null | grep -q 'application.name = "ALSA plug-in'; then
            log "voice-service 采集流已释放 ✅"
            break
        fi
        sleep 1
    done
else
    log "提示：voice-service.service 未运行，保持停止（脚本末尾不启动）"
fi

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

# 3. 重启 wireplumber：强制重建全部设备节点与采集流。
#    仅重载 xhci_pci 时采集端点会假死（见顶部注释），此步骤是采集恢复的关键。
if systemctl --user restart wireplumber 2>/tmp/repair_wp.err; then
    log "wireplumber 已重启，等待设备节点重建..."
else
    log "警告：wireplumber 重启失败：$(cat /tmp/repair_wp.err)"
fi
sleep 3
ensure_pulse_alive

# 4. 确认 sink 与 source 都恢复（设备激活约需 20-45 秒，轮询最多 60 秒）。
#    若首次长时间未恢复，额外重启一次 wireplumber（实测部分场景需要两次）。
sink_ok=0
src_ok=0
restarted_wp=0
for i in $(seq 1 30); do
    ensure_pulse_alive
    sink_ok=$(pactl list sinks short 2>/dev/null | grep -c "usb.*AIMIC")
    src_ok=$(pactl list sources short 2>/dev/null | grep -c "usb.*AIMIC")
    [ "$sink_ok" -ge 1 ] && [ "$src_ok" -ge 1 ] && break
    # 15s 内未恢复 → 再重启一次 wireplumber（共两次，覆盖不稳定场景）
    if [ "$restarted_wp" -eq 0 ] && [ "$i" -eq 8 ]; then
        log "sink/source 15s 内未恢复，再重启一次 wireplumber..."
        systemctl --user restart wireplumber >/dev/null 2>&1 || true
        sleep 3
        restarted_wp=1
    fi
    sleep 2
done
if [ "$sink_ok" -ge 1 ] && [ "$src_ok" -ge 1 ]; then
    log "sink/source 已恢复（sink=$(pactl get-default-sink 2>/dev/null)）"
else
    log "警告：sink 或 source 未完全恢复（sink=$sink_ok source=$src_ok），请检查 wireplumber"
fi

# 5. 播放测试音验证播放链路（先确保 pulse 存活）
ensure_pulse_alive
if timeout 5 pw-play /tmp/usb_audio_health_tone.wav >/dev/null 2>&1; then
    log "测试音播放正常 ✅"
    PLAY_OK=1
else
    log "警告：测试音播放异常（可能设备仍在初始化），请人工确认"
    PLAY_OK=0
fi

# 6. 采集链路自检：录制 3 秒，有 PCM 数据即链路通（静音环境也会收到零帧）。
#    采集激活 ~20-45 秒且不稳定，轮询最多 ~90 秒（10 次）；每次探测前确保
#    pulse 存活（挂起时自动重启，避免 parec 空等拖垮整个流程）。
#    第一轮失败后执行设备 unbind/rebind 复位（实测 6s 恢复）并再轮询一轮。
probe_capture() {
    local round="$1"
    for i in $(seq 1 10); do
        ensure_pulse_alive
        CAP_BYTES=$(parec --format=s16le --rate=16000 --channels=1 --raw 2>/dev/null | wc -c)
        if [ "$CAP_BYTES" -gt 0 ]; then
            log "采集链路正常（${CAP_BYTES} bytes/4s）✅"
            return 0
        fi
        log "采集探测（第 ${round} 轮 ${i}/10 次）无数据，继续等待..."
        sleep 4
    done
    return 1
}
CAP_OK=0
if probe_capture 1; then
    CAP_OK=1
else
    log "警告：第一轮 ~90s 采集无数据，执行设备 unbind/rebind 复位后重试..."
    if rebind_aimic && probe_capture 2; then
        CAP_OK=1
    else
        log "警告：复位后采集仍无数据（~3min 内无 PCM），麦克风可能未恢复，请人工确认"
    fi
fi

# 7. 启动 voice-service（仅当步骤 0 停止了它）。此刻音频栈已重建并验证，
#    语音服务以干净进程重新打开音频流；hermes-gateway 无需重启：voice
#    平台是 Voice Service 的 WS 客户端，语音服务恢复后会自动重连
#    （voice_service 恢复会短暂踢掉 gateway 的旧连接，属预期）。
VS_SVC_OK=0
VS_MIC_OK=0
if [ "$VS_RUNNING" -eq 1 ]; then
    log "启动 voice-service（voice-service.service）..."
    if timeout 120 systemctl --user start voice-service.service 2>/tmp/repair_vs.err; then
        VS_SVC_OK=1
        log "voice-service 已启动，等待其恢复采集..."
    else
        log "警告：voice-service 启动失败或超时：$(cat /tmp/repair_vs.err)"
    fi
    # 等待 voice-service 重新建立采集流（最多 90 秒）
    for i in $(seq 1 45); do
        ensure_pulse_alive
        if pactl list source-outputs 2>/dev/null | grep -q 'application.name = "ALSA plug-in'; then
            VS_MIC_OK=1
            log "voice-service 采集流已恢复 ✅"
            break
        fi
        sleep 2
    done
    if [ "$VS_MIC_OK" -eq 0 ]; then
        log "警告：voice-service 在 90s 内未恢复采集流（可能未启用 --audio 或启动慢），请检查"
    fi
else
    log "提示：voice-service 最初未运行，保持停止（跳过启动）"
    VS_SVC_OK=1  # 服务本就不在，不视为失败
fi

# 8. 汇总
if [ "$PLAY_OK" -eq 1 ] && [ "$CAP_OK" -eq 1 ] && [ "$VS_SVC_OK" -eq 1 ]; then
    log "修复完成：播放 ✅ 采集 ✅ voice-service ✅"
    exit 0
else
    log "修复部分完成（播放=$PLAY_OK 采集=$CAP_OK voice-service=$VS_SVC_OK mic=${VS_MIC_OK:-0}），请人工确认"
    exit 2
fi
