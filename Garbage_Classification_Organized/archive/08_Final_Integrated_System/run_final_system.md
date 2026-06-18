# 最终系统运行指南

## 1. 硬件准备
见 `wiring_final.md`

## 2. 树莓派依赖
```bash
sudo apt install -y python3-pip python3-opencv
pip3 install tflite-runtime pyserial pillow numpy
```

## 3. 检查模型
```bash
ls -la ../../export/latest_tflite_fp16.tflite
```

## 4. 检查类别映射
```bash
cat ../../config/class_mapping.json
# 确保: 0=其他,1=厨余,2=可回收,3=有害
```

## 5. 烧录单片机
用 STC-ISP 烧录 `mcu_final_ultrasonic_full_load_rkho.c` 到 STC89C52RC

## 6. 检查串口
```bash
ls -l /dev/ttyAMA0  # 或 /dev/ttyUSB0
sudo usermod -aG dialout $USER
```

## 7. 启动
```bash
python3 rpi_final_ai_sorting_with_full_load.py --serial-port /dev/ttyAMA0
# USB-TTL:
python3 rpi_final_ai_sorting_with_full_load.py --serial-port /dev/ttyUSB0
# SSH:
python3 rpi_final_ai_sorting_with_full_load.py --no-window
```

## 8. 常见问题
| 问题 | 解决 |
|------|------|
| 串口打不开 | 检查权限 dialout 组 |
| 摄像头黑屏 | 尝试 --camera-index 1 |
| MCU无响应 | 确认波特率9600, 电平转换 |
| 舵机不动 | 独立供电 |
