# 08 — 旧 AA55 协议参考

## 1. 用途
保留旧的 AA xx 55 字节帧协议文件作为历史参考。**不是最终部署协议**。

## 2. 包含的脚本

| 文件名 | 类型 | 作用 | 是否最终推荐 |
|--------|------|------|-------------|
| `rpi_aa55_serial_handshake_test_old.py` | Legacy | 旧 AA55 手动串口测试 | ❌ |
| `mcu_ultrasonic_led_aa55_test_old.c` | Legacy | 旧 AA55 LED 测试固件 | ❌ |
| `rpi_final_sorting_aa55_old.py` | Legacy | 旧 final_sorting_system.py | ❌ |
| `mcu_ultrasonic_sorting_aa55_old.c` | Legacy | 旧 mcu_52rc_garbage_sorting_final.c | ❌ |

## 3. 旧协议（不再使用）
MCU→RPi: 0xA1 / 0xCC / 0xDD / 0xEE | RPi→MCU: AA xx 55

## 4. 为什么不推荐
- 无满载检测
- 使用二进制帧协议而非 ASCII 单字符
- 被 ASCII RKHO + T/F/N/D/E 协议取代
- 最终部署请看 `10_Final_Integrated_System/`
