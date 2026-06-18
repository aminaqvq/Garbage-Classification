# 07 — 满载 RKHO 直发测试版

## 1. 用途
树莓派 AI 识别稳定后直接发送 R/K/H/O（无超声波触发 T）。MCU 有满载检测（F/N）。
**不是最终部署版** — 缺少超声波主动触发。

## 2. 包含的脚本

| 文件名 | 类型 | 作用 | 是否最终推荐 |
|--------|------|------|-------------|
| `rpi_ai_rkho_direct_sorting_test.py` | RPi Python | AI+直发RKHO,含F/N满载处理 | ❌ 过渡版 |
| `mcu_full_load_rkho_sorting_test.c` | MCU C | 满载+RKHO,无超声波触发 | ❌ 过渡版 |

## 3. 为什么不是最终版
- 无 HC-SR04 超声波检测
- 无 T 触发流程
- 最终整合版请看 `10_Final_Integrated_System/`
