# 08 — 超声波 + LED + UART 帧协议测试版

## 1. 这个文件夹是干什么的

**协议验证版本**。验证 HC-SR04 超声波检测 → AA xx 55 帧协议通信 → LED 指示的完整流程。固件**只有 LED 指示，没有舵机和电机**——用于在加入机械动作之前先确保串口协议正确。

## 2. 包含的脚本

| 文件名 | 类型 | 作用 | 是否推荐使用 |
|--------|------|------|-------------|
| `ultrasonic_led_test.c` | MCU C | 超声波+LED+帧协议测试固件 | 测试/验证用 |
| `main_duplicate.c` | MCU C | ⚠ 与上面几乎完全相同的副本 | 仅供参考 |

## 3. 工作流程

```
1. HC-SR04 持续测距
2. 物体进入检测范围 → 单片机发送 0xA1
3. 等待树莓派返回 AA xx 55 分类帧
4. 收到有效帧 → 回复 0xCC → 点亮对应 LED
5. 延时后 → 回复 0xDD → 熄灭 LED
6. 回到等待检测状态
```

## 4. 串口协议

与最终版相同的 AA xx 55 帧协议（见 `10_Final_Integrated_System/protocol_final.md`）。

| 方向 | 数据 | 含义 |
|------|------|------|
| MCU → RPi | 0xA1 | 检测到物体 |
| RPi → MCU | AA 01 55 | 可回收 |
| RPi → MCU | AA 02 55 | 厨余 |
| RPi → MCU | AA 03 55 | 有害 |
| RPi → MCU | AA 04 55 | 其他 |
| MCU → RPi | 0xCC | ACK |
| MCU → RPi | 0xDD | DONE |

## 5. 单片机外设

| 外设 | 52RC 引脚 | 说明 |
|------|-----------|------|
| HC-SR04 TRIG | P1^0 | 超声波触发 |
| HC-SR04 ECHO | P1^1 | 超声波回响 |
| LED 可回收 | P2^0 | 低电平点亮 |
| LED 厨余 | P2^1 | 低电平点亮 |
| LED 有害 | P2^2 | 低电平点亮 |
| LED 其他 | P2^3 | 低电平点亮 |
| LED 检测状态 | P2^4 | 低电平点亮 |

⚠ **没有舵机，没有电机** — 这里只是验证超声波检测和串口通信。

## 6. 关于两个 C 文件

`ultrasonic_led_test.c` 和 `main_duplicate.c` 几乎是完全相同的代码，唯一区别是等待超时参数：
- `ultrasonic_led_test.c`：`WAIT_PI_TIMEOUT_MS = 30000`（30秒）
- `main_duplicate.c`：`WAIT_PI_TIMEOUT_MS = 10000`（10秒）

推荐使用 `ultrasonic_led_test.c`（超时更宽容）。

## 7. 注意事项

- 这是**协议测试版**，不是最终运行版本
- 此固件无舵机/电机控制
- 此固件无满载检测
- 可以与 `06_Serial_Handshake_Test/serial_handshake_test.py` 配合进行手动串口测试
