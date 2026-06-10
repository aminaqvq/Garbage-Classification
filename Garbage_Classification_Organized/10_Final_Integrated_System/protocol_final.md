# 最终串口协议：ASCII RKHO + T/F/N/D/E

## 基本参数
- 波特率：9600, 8N1
- 电平：树莓派 3.3V TTL, 52RC 5V TTL, 需电平转换

## MCU → RPi

| 字符 | 含义 | 说明 |
|------|------|------|
| T | Trigger | 超声波检测到垃圾 |
| F | Full | 满载, 暂停分类 |
| N | Normal | 满载解除 |
| D | Done | 分拣完成 |
| E | Error | 超时/异常 |

## RPi → MCU

| 字符 | 分类 | 说明 |
|------|------|------|
| R | 可回收 | Recyclable |
| K | 厨余 | Kitchen |
| H | 有害 | Harmful |
| O | 其他 | Other |

## 正常流程
MCU: T → RPi: R/K/H/O → MCU: D

## 满载流程
MCU: F → RPi 暂停 → MCU: N → RPi 恢复

## 旧协议
AA xx 55 已移至 `08_AA55_Ultrasonic_Protocol_Reference/`, 不再使用。
