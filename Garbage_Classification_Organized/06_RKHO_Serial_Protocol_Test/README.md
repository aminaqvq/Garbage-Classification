# 06 — RKHO 串口协议最小调试工具

## 脚本

| 文件 | 类型 | 作用 |
|------|------|------|
| `rpi_manual_rkho_protocol_test.py` | RPi Python | 手动发送 R/K/H/O，显示 D/F/N/E |
| `mcu_rkho_protocol_test.c` | MCU C | 最小协议测试固件 |

## 协议说明

新版视觉触发系统**不使用 T 触发字符**。本工具的 MCU 固件出于最小验证目的仍会发 T，但 T 标注为 `[legacy]`。
如需测试新版协议（无 T），请直接使用：

```bash
python ../09_Vision_Trigger_5Class_System/rpi/rpi_vision_trigger_sorting.py --test-char R
```

## 运行

```bash
# 预装依赖
pip install pyserial

# 运行（输入 r/k/h/o 发送，q 退出）
python rpi_manual_rkho_protocol_test.py --serial-port /dev/ttyUSB0
```

## 依赖

- Python 3.7+
- pyserial（`pip install pyserial`）

注意：`--help` 不需要 pyserial。只有真正打开串口时才需要。
