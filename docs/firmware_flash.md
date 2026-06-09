# 固件烧录指南

## 使用 STC-ISP 烧录

### 1. 下载安装

从 STC 官网下载 STC-ISP 烧录软件：
- 官网：https://www.stcmcudata.com/
- 支持 Windows（推荐）/ Linux

### 2. 硬件连接

使用 USB-TTL 下载器连接 STC89C52RC：

| USB-TTL | 52RC | 说明 |
|---------|------|------|
| 5V | VCC (40脚) | 供电 |
| GND | GND (20脚) | 共地 |
| TXD | P3.0/RXD (10脚) | 交叉连接 |
| RXD | P3.1/TXD (11脚) | 交叉连接 |

### 3. 烧录步骤

1. 连接 USB-TTL 到电脑，确认驱动已安装（CH340/CP2102 等）
2. 打开 STC-ISP 软件
3. 选择单片机型号：**STC89C52RC**
4. 选择串口号（在设备管理器中查看 COM 端口号）
5. 点击"打开程序文件"，选择 `firmware/` 下的 C 文件
6. 点击"编译/下载"或直接"下载"
7. **在点击下载后，给 52RC 断电再上电**（冷启动）
8. 等待烧录完成

### 4. 选择固件版本

| 路径 | 功能 | 协议 |
|------|------|------|
| `firmware/mcu_final/mcu_52rc_garbage_sorting_final.c` | 完整版 | AA 帧协议 |
| `firmware/mcu_led_protocol/main.c` | 精简版 | AA 帧协议 |
| `firmware/mcu_led_protocol/52rc_ultrasonic_uart_led_test.c` | 测试版 | AA 帧协议 |
| `firmware/mcu_servo_char/servo_control.c` | 单字符版 | 单字符协议 |

### 5. 验证烧录

烧录完成后：
1. 将 52RC 按照 [hardware_setup.md](hardware_setup.md) 接入电路
2. 树莓派端运行串口测试脚本验证通信：

```bash
cd rpi/
python3 serial_handshake_test.py
```

## 使用 SDCC 编译（Linux/macOS）

如果你使用开源工具链：

```bash
# 安装 SDCC
sudo apt install sdcc  # Debian/Ubuntu

# 编译（以 mcu_final 为例）
cd firmware/mcu_final/
sdcc mcu_52rc_garbage_sorting_final.c

# 生成 .ihx 文件，转换为 .hex 后使用 stcgal 烧录
makebin < mcu_52rc_garbage_sorting_final.ihx > firmware.hex
stcgal -P /dev/ttyUSB0 firmware.hex
```

**注意**：SDCC 编译的 8051 代码可能需要调整头文件路径和寄存器定义。推荐使用 STC-ISP 配合 Keil C51 编译。
