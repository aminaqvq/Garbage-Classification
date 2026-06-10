# 最终系统运行指南

## 1. 硬件准备

- [ ] 树莓派已连接摄像头
- [ ] 52RC 单片机已连接 USB-TTL 下载器
- [ ] 树莓派 TXD(GPIO14) ↔ 52RC RXD(P3.0)
- [ ] 树莓派 RXD(GPIO15) ↔ 52RC TXD(P3.1)
- [ ] GND 共地
- [ ] **电平转换**：树莓派 3.3V，52RC 5V（至少串联 1kΩ 电阻）
- [ ] 舵机独立 5V 供电
- [ ] 电机通过 L298N 供电（5V-12V）

## 2. 树莓派软件准备

### 安装系统依赖

```bash
sudo apt update
sudo apt install -y python3-pip python3-opencv libatlas-base-dev
```

### 安装 Python 依赖

```bash
# 推荐 tflite-runtime（轻量）
pip3 install tflite-runtime

# 或者完整 TensorFlow
# pip3 install tensorflow

# 其他依赖
pip3 install pyserial pillow numpy opencv-python
```

### 检查串口

```bash
# 查看串口设备
ls -l /dev/ttyAMA0 /dev/serial0

# 如果使用 USB-TTL
ls -l /dev/ttyUSB*

# 设置串口权限
sudo usermod -aG dialout $USER
# 需要注销重新登录

# 或临时授权
sudo chmod 666 /dev/ttyAMA0
```

### 禁用蓝牙（释放 /dev/ttyAMA0）

```bash
# 树莓派 3B+/4B 的 /dev/ttyAMA0 默认被蓝牙占用
sudo systemctl disable bluetooth
# 或通过 raspi-config
sudo raspi-config  →  Interface Options  →  Serial Port
    →  "No" to login shell over serial
    →  "Yes" to serial port hardware
```

### 检查摄像头

```bash
# USB 摄像头
ls /dev/video*

# 测试摄像头
python3 -c "import cv2; cap=cv2.VideoCapture(0); print(cap.read()[0]); cap.release()"
```

## 3. 准备模型文件

```bash
# 将 PC 端导出的 TFLite 模型复制到树莓派
# 目标路径：<项目根>/export/latest_tflite_fp16.tflite

# 确认文件存在
ls -la export/latest_tflite_fp16.tflite
```

## 4. 烧录单片机

1. 使用 STC-ISP 软件（Windows）
2. 选择单片机型号：STC89C52RC
3. 打开文件：`10_Final_Integrated_System/mcu_52rc_garbage_sorting_final.c`
4. 点击编译/下载
5. **冷启动**：下载开始后给 52RC 断电再上电
6. 等待烧录完成

## 5. 启动系统

```bash
cd 10_Final_Integrated_System/

# 基本运行
python3 final_sorting_system.py

# 如果串口是 /dev/ttyUSB0
python3 final_sorting_system.py --serial-port /dev/ttyUSB0

# 不显示窗口（SSH 模式）
python3 final_sorting_system.py --no-window
```

## 6. 常见问题排查

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| 串口打不开 | 权限不足 | `sudo chmod 666 /dev/ttyAMA0` |
| 串口被占用 | 蓝牙占用 | `sudo systemctl disable bluetooth` |
| 摄像头打不开 | index 错误 | 改成 `--camera-index 0` 或 `1` |
| 模型加载失败 | 路径不对 | 确认 `export/` 下有 `.tflite` 文件 |
| 单片机不响应 | 波特率/接线 | 确认 9600bps，TX/RX 交叉，GND 共地 |
| 舵机不转 | 供电不足 | 舵机必须独立 5V 供电 |
| 电机不转 | L298N 供电 | 电机电源和逻辑电源都要接 |
| 电平问题 | 3.3V ↔ 5V | 使用电平转换模块 |

## 7. 日志和截图

- 运行日志：`Logs/servo_char_runtime.log`
- 统计日志：`Logs/final_stats.csv`
- 识别截图：`Captures_Final/`
