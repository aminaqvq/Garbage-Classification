# 09 — 触发 RKHO 协议参考

## 1. 用途
T 触发 + RKHO 分类的旧参考版本。原始 `locked_trigger_system.py` 有触发逻辑但未完整处理 F/N 满载。

## 2. 包含的脚本

| 文件名 | 类型 | 作用 | 是否最终推荐 |
|--------|------|------|-------------|
| `rpi_triggered_ai_sorting_rkho_old.py` | Legacy RPi Python | 旧 T 触发+RKHO 上位机 | ❌ |
| `mcu_triggered_rkho_protocol_test.c` | MCU C | 对应协议测试固件 | ❌ 测试用 |

## 3. 为什么不是最终版
- 原始 Python 脚本未完整处理 F/N 满载
- 最终完整逻辑已整合到 `10_Final_Integrated_System/`
