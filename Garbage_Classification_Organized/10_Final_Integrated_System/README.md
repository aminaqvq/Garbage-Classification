# ★ 10 — 最终整合系统

## 1. 这是唯一最终推荐部署版本

整合了树莓派 AI + 单片机超声波 + 满载检测 + 舵机分拣 + 电机推出/回位 + LED 指示。

## 2. 包含的脚本

| 文件名 | 类型 | 作用 | 是否最终推荐 |
|--------|------|------|-------------|
| `rpi_final_ai_sorting_with_full_load.py` | RPi Python ★ | 最终树莓派上位机 | ★ 是 |
| `mcu_final_ultrasonic_full_load_rkho.c` | MCU C ★ | 最终单片机固件 | ★ 是 |

## 3. 最终协议：ASCII RKHO + T/F/N/D/E

| 方向 | 字符 | 含义 |
|------|------|------|
| MCU→RPi | T | 超声波检测到垃圾 |
| MCU→RPi | F | 满载，暂停 |
| MCU→RPi | N | 满载解除 |
| MCU→RPi | D | 分拣完成 |
| MCU→RPi | E | 错误/超时 |
| RPi→MCU | R | 可回收 |
| RPi→MCU | K | 厨余 |
| RPi→MCU | H | 有害 |
| RPi→MCU | O | 其他 |

旧的 AA xx 55 协议不再作为最终部署协议。

## 4. 工作流程
MCU 超声波检测物体 → 发送 T → 树莓派 AI 识别 → 发送 R/K/H/O → MCU 舵机+电机分拣 → 发送 D
满载：MCU 发送 F → 树莓派暂停 → 清除后 MCU 发送 N → 恢复

## 5. 运行
```bash
# 烧录 MCU: mcu_final_ultrasonic_full_load_rkho.c
# 树莓派:
python3 rpi_final_ai_sorting_with_full_load.py --serial-port /dev/ttyAMA0
```

详见 `run_final_system.md`, `wiring_final.md`, `protocol_final.md`。
