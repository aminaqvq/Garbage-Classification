# 最终接线说明（五分类视觉触发版）

详细接线说明见 `09_Vision_Trigger_5Class_System/wiring_vision_trigger.md`。

## 核心连接

- 树莓派 UART → STC89C52RC（电平转换，3.3V↔5V）
- 舵机 → P1^5（独立供电）
- 电机 → P0^0/P0^1/P0^2（独立供电，P0 外部上拉）
- 满载传感器 L0-L3 → P0^3/P0^5/P0^6/P0^7（P0 外部上拉）
- 各模块 GND 共地

## 已移除

- HC-SR04 超声波传感器
- TRIG/ECHO 引脚接线
