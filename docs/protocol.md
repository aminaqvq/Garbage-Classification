# 串口通信协议

## 最终推荐协议：ASCII RKHO + T/F/N/D/E ★

波特率: 9600, 8N1

### MCU → RPi
| 字符 | 含义 |
|------|------|
| T | 超声波检测到垃圾 |
| F | 满载暂停 |
| N | 满载解除 |
| D | 分拣完成 |
| E | 错误/超时 |

### RPi → MCU
| 字符 | 分类 |
|------|------|
| R | 可回收 |
| K | 厨余 |
| H | 有害 |
| O | 其他 |

使用: `Garbage_Classification_Organized/10_Final_Integrated_System/`

## 历史协议：AA xx 55 字节帧（不再推荐）

MCU→RPi: 0xA1/0xCC/0xDD | RPi→MCU: AA xx 55

旧协议文件在 `Garbage_Classification_Organized/08_AA55_Ultrasonic_Protocol_Reference/`，仅作历史参考。
