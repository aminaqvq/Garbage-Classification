# 10 — ★ 最终整合系统（推荐部署版本）

## 1. 这个文件夹是干什么的

**这是整个项目最终推荐运行的版本**。树莓派运行 Python AI 识别脚本，通过串口 AA xx 55 帧协议与 52RC 单片机通信，单片机控制舵机分拣和电机推出/回位，并带有 LED 分类指示。

## 2. 包含的脚本

| 文件名 | 类型 | 作用 | 是否推荐使用 |
|--------|------|------|-------------|
| `final_sorting_system.py` | RPi Python ★ | 最终版 AI + 串口握手上位机 | ✅ 最终推荐 |
| `mcu_52rc_garbage_sorting_final.c` | MCU C ★ | 最终版单片机固件 | ✅ 最终推荐 |

## 3. 系统完整流程

```
┌───────────────────────────────────────────────────────────────────┐
│ 1. [MCU] HC-SR04 超声波持续测距                                    │
│ 2. [MCU] 物体进入检测范围 → 发送 0xA1 给树莓派                     │
│ 3. [RPi] 收到 0xA1 → 触发摄像头 AI 识别                           │
│ 4. [RPi] TFLite 连续推理 → 达到稳定帧数 → 确定分类                  │
│ 5. [RPi] 发送 AA xx 55 给单片机                                    │
│     xx = 01(可回收) / 02(厨余) / 03(有害) / 04(其他)             │
│ 6. [MCU] 收到有效帧 → 回复 0xCC (ACK)                             │
│ 7. [MCU] 点亮对应分类 LED                                         │
│ 8. [MCU] 舵机转到对应角度                                          │
│ 9. [MCU] 电机开门(EN=1,IN1=1,IN2=0) → 停(EN=0) → 关门(IN1=0,IN2=1) → 停 |
│ 10.[MCU] 动作完成 → 回复 0xDD (DONE)                              │
│ 11.[RPi] 记录日志/截图 → 等待下一轮                                │
└───────────────────────────────────────────────────────────────────┘
```

## 4. 串口协议摘要

| 方向 | 数据 | 含义 |
|------|------|------|
| MCU → RPi | `0xA1` | 超声波检测到物体 |
| RPi → MCU | `AA 01 55` | 可回收 |
| RPi → MCU | `AA 02 55` | 厨余 |
| RPi → MCU | `AA 03 55` | 有害 |
| RPi → MCU | `AA 04 55` | 其他 |
| MCU → RPi | `0xCC` | ACK 收到 |
| MCU → RPi | `0xDD` | DONE 完成 |
| MCU → RPi | `0xEE` | ERROR 超时/异常 |

**详见**：`protocol_final.md`

## 5. 单片机外设接线

| 外设 | 52RC 引脚 | 说明 |
|------|-----------|------|
| HC-SR04 TRIG | P1^0 | 超声波触发 |
| HC-SR04 ECHO | P1^1 | 超声波回响 |
| SG90 舵机 | P1^5 | PWM 控制 |
| L298N EN | P0^0 | 电机使能 |
| L298N IN1 | P0^1 | 电机方向1 |
| L298N IN2 | P0^2 | 电机方向2 |
| LED 可回收 | P2^0 | 低电平点亮 |
| LED 厨余 | P2^1 | 低电平点亮 |
| LED 有害 | P2^2 | 低电平点亮 |
| LED 其他 | P2^3 | 低电平点亮 |
| LED 检测状态 | P2^4 | 低电平点亮 |

**详见**：`wiring_final.md`

## 6. 运行方法

```bash
cd 10_Final_Integrated_System/

# 基本运行
python3 final_sorting_system.py

# 指定串口
python3 final_sorting_system.py --serial-port /dev/ttyUSB0

# 手动模式（按空格发送）
python3 final_sorting_system.py --mode manual

# 指定模型
python3 final_sorting_system.py --model-path ../export/latest_tflite_int8.tflite
```

## 7. 运行前检查清单

- [ ] 模型文件已放入 `export/latest_tflite_fp16.tflite`
- [ ] 单片机已烧录 `mcu_52rc_garbage_sorting_final.c`
- [ ] 串口已连接且电平已转换（3.3V ↔ 5V）
- [ ] 串口权限已设置（`sudo usermod -aG dialout pi`）
- [ ] 摄像头连接正常
- [ ] 舵机和电机已独立供电
- [ ] 所有模块 GND 共地
- [ ] 树莓派安装依赖：`pip install tflite-runtime pyserial pillow opencv-python`

## 8. 配套文档

- `run_final_system.md` — 详细运行指南
- `wiring_final.md` — 完整接线图
- `protocol_final.md` — 串口协议详解
