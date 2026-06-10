# 06 — RKHO 串口协议手动测试

## 1. 用途
在不运行 AI 的情况下，手动测试最终 ASCII RKHO 协议的串口通信。

## 2. 包含的脚本

| 文件名 | 类型 | 作用 | 是否最终推荐 |
|--------|------|------|-------------|
| `rpi_manual_rkho_protocol_test.py` | RPi Python | 手动串口测试工具 | 测试用 |
| `mcu_rkho_protocol_test.c` | MCU C | 最小 RKHO 协议测试固件 | 测试用 |

## 3. 串口协议
MCU→RPi: T/F/N/D/E | RPi→MCU: R/K/H/O

## 4. 运行方法
```bash
python3 rpi_manual_rkho_protocol_test.py --serial-port /dev/ttyAMA0
```
输入 r/k/h/o 发送分类，q 退出。MCU 烧录 `mcu_rkho_protocol_test.c`。

## 5. 注意事项
- 这是测试工具，不是最终部署版本
- MCU 测试固件无超声波、无满载、无电机
