# 项目总览 — 智能垃圾分类分拣系统

## 1. 项目是什么
基于树莓派 + AI (MobileNetV3/TFLite) + STC89C52RC 单片机的智能垃圾分类系统。
四类垃圾：可回收、有害、厨余、其他。

## 2. 最终协议: ASCII RKHO + T/F/N/D/E
MCU→RPi: T(触发) F(满载) N(恢复) D(完成) E(错误)
RPi→MCU: R(可回收) K(厨余) H(有害) O(其他)

旧 AA xx 55 协议已移到 `08_AA55_Ultrasonic_Protocol_Reference/`, 不再使用。

## 3. 目录导航

| 文件夹 | 内容 | 读者 |
|--------|------|------|
| 01-04 | 数据→模型流水线 | 需要训练模型的人 |
| 05 | 树莓派 AI 预览 | 测试模型 |
| 06 | RKHO 手动串口测试 | 调试通信 |
| 07 | 满载 RKHO 直发版 | 过渡参考 |
| 08 | 旧 AA55 协议参考 | 历史参考 |
| 09 | 触发 RKHO 旧版 | 历史参考 |
| **10** | **★ 最终部署** | **正式使用** |

## 4. 新手路径
1. 已有模型 → 直接看 `10_Final_Integrated_System/`
2. 需训练 → 01→02→03→04→10
3. 调试通信 → 06 手动测试

## 5. 最终部署
- 树莓派: `10_Final_Integrated_System/rpi_final_ai_sorting_with_full_load.py`
- 单片机: `10_Final_Integrated_System/mcu_final_ultrasonic_full_load_rkho.c`
