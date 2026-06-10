# 硬件接线汇总

## 最终接线(10_Final_Integrated_System)

### 树莓派↔52RC UART
RPi TXD(GPIO14)→MCU RXD(P3.0), RPi RXD(GPIO15)←MCU TXD(P3.1), GND↔GND
⚠ 3.3V vs 5V 需电平转换

### MCU 外设
| 外设 | 引脚 |
|------|------|
| HC-SR04 TRIG/ECHO | P1^0/P1^1 |
| 舵机 | P1^5 |
| 电机 EN/IN1/IN2 | P0^0/P0^1/P0^2 |
| 满载 L0/L1/L2/L3 | P0^3/P0^5/P0^6/P0^7 |
| LED 可回收/厨余/有害/其他 | P2^0/P2^1/P2^2/P2^3 |
| LED 检测 | P2^4 |

⚠ P0口开漏,需外部上拉。舵机/电机独立供电,所有GND共地。
TODO: 满载传感器具体型号需人工确认。
