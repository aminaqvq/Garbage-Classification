# 07 — 单字符舵机控制版本（含满载保护）

## 1. 这个文件夹是干什么的

**过渡版本**。树莓派 AI 识别后直接发送 `R/H/K/O` 单字符给单片机，单片机收到字符后改变舵机角度并执行开门/关门动作。包含 **垃圾桶满载保护**：单片机通过 L0-L3 传感器检测满载，发送 `F`/`N` 通知上位机暂停/恢复分类。

## 2. 包含的脚本

| 文件名 | 类型 | 作用 | 是否推荐使用 |
|--------|------|------|-------------|
| `servo_char_control.py` | RPi Python | 单字符舵机控制上位机 | 过渡版本，非最终 |
| `mcu_full_main.c` | MCU C | 单字符协议固件（含 F/N 满载） | 配对使用 |

## 3. 工作流程

```
1. 树莓派摄像头持续识别
2. AI 识别稳定后 → 发送 R/H/K/O 单字符
3. 单片机收到字符 → 改变舵机角度 → 开门 → 停 → 关门 → 停
4. (满载保护) 任意满载传感器触发 → 单片机发 F → 上位机暂停分类
5. (满载恢复) 所有传感器恢复 → 单片机发 N → 上位机恢复分类
```

## 4. 串口协议

| 方向 | 字符 | 含义 | 舵机角度 |
|------|------|------|----------|
| RPi → MCU | R | 可回收 | angle_pwm=8 |
| RPi → MCU | H | 有害 | angle_pwm=19 |
| RPi → MCU | K | 厨余 | angle_pwm=29 |
| RPi → MCU | O | 其他 | angle_pwm=36 |
| MCU → RPi | F | FULL 满载 | 暂停分类 |
| MCU → RPi | N | NORMAL 恢复 | 可继续分类 |

## 5. 单片机外设接线

| 外设 | 52RC 引脚 | 说明 |
|------|-----------|------|
| SG90 舵机 | P1^5 | PWM |
| L298N EN | P0^0 | 电机使能 |
| L298N IN1 | P0^1 | 方向1 |
| L298N IN2 | P0^2 | 方向2 |
| 满载传感器 L0 | P0^3 | 低电平触发 |
| 满载传感器 L1 | P0^5 | 低电平触发 |
| 满载传感器 L2 | P0^6 | 低电平触发 |
| 满载传感器 L3 | P0^7 | 低电平触发 |

## 6. 运行方法

```bash
cd 07_Servo_Char_Control_Version/

# 自动模式（推荐）
python3 servo_char_control.py --mode auto

# 手动模式（按空格发送）
python3 servo_char_control.py --mode manual

# 手动测试舵机
python3 servo_char_control.py --test-char R

# 满载解除后自动恢复
python3 servo_char_control.py --mode auto --auto-resume-after-clear

# 单片机：烧录 mcu_full_main.c
```

## 7. 注意事项

- ⚠ 这不是最终推荐版本（推荐 10_Final_Integrated_System）
- 此版本**没有超声波触发**，树莓派持续识别并自动发送
- **没有 ACK/DONE 握手**，靠冷却时间防止重复发送
- 包含满载保护（F/N）功能
- 相比最终版缺少超声波触发和握手确认机制
- 适合简化部署或教学演示场景
