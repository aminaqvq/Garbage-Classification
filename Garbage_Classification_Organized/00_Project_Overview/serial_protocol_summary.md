# 串口协议总结（五分类视觉触发版）

## 最终协议: ASCII RKHO + D/F/N/E（无 T）

RPi→MCU: R(可回收) K(厨余) H(有害) O(其他)
MCU→RPi: D(完成) F(满载) N(恢复) E(错误)

T 触发字符已在新版中彻底移除（仅作为 [legacy] 在 06 调试工具中显示）。

## 使用

**最终部署**: `09_Vision_Trigger_5Class_System/`
- RPi: `rpi/rpi_vision_trigger_sorting.py`
- MCU: `mcu/mcu_vision_trigger_full_load_rkho.c`

调试工具: `06_RKHO_Serial_Protocol_Test/`
- T 标注为 [legacy]，不是当前协议
