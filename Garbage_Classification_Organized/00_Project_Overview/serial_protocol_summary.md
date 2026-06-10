# 串口协议汇总

波特率: 9600bps, 8N1

## 最终协议：ASCII RKHO + T/F/N/D/E ★

| 方向 | 字符 | 含义 |
|------|------|------|
| MCU→RPi | T | 超声波触发 |
| MCU→RPi | F | 满载暂停 |
| MCU→RPi | N | 满载解除 |
| MCU→RPi | D | 分拣完成 |
| MCU→RPi | E | 错误/超时 |
| RPi→MCU | R | 可回收 |
| RPi→MCU | K | 厨余 |
| RPi→MCU | H | 有害 |
| RPi→MCU | O | 其他 |

**使用**: `10_Final_Integrated_System/`

## 旧协议：AA xx 55 字节帧（不再使用）

| 方向 | 数据 | 含义 |
|------|------|------|
| MCU→RPi | 0xA1 | 检测到物体 |
| RPi→MCU | AA 01/02/03/04 55 | 分类 |
| MCU→RPi | 0xCC | ACK |
| MCU→RPi | 0xDD | DONE |

**已移至**: `08_AA55_Ultrasonic_Protocol_Reference/`
