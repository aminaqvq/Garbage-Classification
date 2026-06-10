# 串口协议汇总

波特率统一为 **9600 bps**，8N1（8 数据位、无校验、1 停止位）。

---

## 协议一：AA xx 55 帧协议 ★ 最终版

**使用脚本**：`final_sorting_system.py` + `mcu_52rc_garbage_sorting_final.c`
**详见**：`10_Final_Integrated_System/protocol_final.md`

| 方向 | 数据 | 含义 |
|------|------|------|
| MCU → RPi | `0xA1` | HC-SR04 检测到物体，请求 AI 识别 |
| RPi → MCU | `AA 01 55` | 分类结果：**可回收** |
| RPi → MCU | `AA 02 55` | 分类结果：**厨余** |
| RPi → MCU | `AA 03 55` | 分类结果：**有害** |
| RPi → MCU | `AA 04 55` | 分类结果：**其他** |
| MCU → RPi | `0xCC` | ACK：已收到分类帧 |
| MCU → RPi | `0xDD` | DONE：分拣动作完成 |
| MCU → RPi | `0xEE` | ERROR：超时或异常 |

**状态**：✅ 最终推荐，Python 和 C 完全匹配。

---

## 协议二：单字符 R/H/K/O（含 F/N 满载保护）

**使用脚本**：`servo_char_control.py` + `mcu_full/main.c`
**详见**：`07_Servo_Char_Control_Version/README.md`

| 方向 | 字符 | 含义 |
|------|------|------|
| RPi → MCU | `R` | 可回收（angle_pwm=8） |
| RPi → MCU | `H` | 有害（angle_pwm=19） |
| RPi → MCU | `K` | 厨余（angle_pwm=29） |
| RPi → MCU | `O` | 其他（angle_pwm=36） |
| MCU → RPi | `F` | FULL：垃圾桶满载，暂停分类 |
| MCU → RPi | `N` | NORMAL：满载解除，可恢复分类 |

**状态**：✅ 配套完整，Python 和 C 匹配。
**说明**：此协议无超声波触发、无 ACK/DONE 握手。适用于简化部署场景。

---

## 协议三：T/A/D/E 锁定触发协议

**使用脚本**：`locked_trigger_system.py`
**详见**：`09_Locked_Trigger_Version/README.md`

| 方向 | 字符 | 含义 |
|------|------|------|
| MCU → RPi | `T` | Trigger：超声波检测到物体，触发识别 |
| RPi → MCU | `R/K/H/O` | 分类结果 |
| MCU → RPi | `A` | ACK：已收到分类 |
| MCU → RPi | `D` | DONE：动作完成 |
| MCU → RPi | `E` | ERROR：异常 |

**状态**：⚠ **无完全匹配的 C 固件**。Python 脚本存在，但 firmware 中没有发送 T/A/D/E 的固件。需人工确认是否使用了未提交的固件版本。

---

## 协议对比

| 特性 | AA xx 55 帧（最终版） | 单字符 R/H/K/O | T/A/D/E 锁定触发 |
|------|----------------------|----------------|-------------------|
| 帧格式 | 3 字节帧：AA xx 55 | 1 字节 ASCII | 1 字节 ASCII |
| 超声波触发 | ✅ HC-SR04 | ❌ 无 | ✅ 触发字符 T |
| ACK/DONE 握手 | ✅ 0xCC / 0xDD | ❌ 无 | ✅ A/D/E |
| 满载保护 | ❌ | ✅ F/N | ✅ F/N |
| 舵机控制 | ✅ | ✅ | ✅ |
| 电机控制 | ✅ | ✅ | ✅ |
| LED 指示 | ✅ | ❌ | ❌ |
| 配套完整性 | ✅ 完全匹配 | ✅ 完全匹配 | ⚠ 缺 MCU 固件 |
| 推荐等级 | ★ 最终推荐 | 过渡/简化版 | 需人工确认 |
