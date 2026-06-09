#!/bin/bash
# 树莓派环境安装脚本
# 用法：bash scripts/setup_rpi.sh

set -e

echo "========== 树莓派垃圾分类系统环境安装 =========="

# 系统更新
echo "[1/5] 更新系统..."
sudo apt-get update -y

# 系统依赖
echo "[2/5] 安装系统依赖..."
sudo apt-get install -y \
    python3-pip \
    python3-opencv \
    python3-numpy \
    python3-pil \
    libatlas-base-dev \
    libopenjp2-7 \
    libtiff5 \
    libjpeg-dev \
    libpng-dev \
    libwebp-dev

# Python 依赖
echo "[3/5] 安装 Python 依赖..."
python3 -m pip install --upgrade pip

# 树莓派使用 tflite-runtime（比完整 tensorflow 轻量很多）
python3 -m pip install tflite-runtime

# 其他依赖
python3 -m pip install pyserial pillow numpy

# 可选：如果需要用 USB 摄像头
echo "[4/5] 启用串口..."
# 启用 UART
sudo raspi-config nonint do_serial 2
# 禁用蓝牙串口（释放 /dev/ttyAMA0）
sudo sed -i 's/console=serial0,115200 //' /boot/cmdline.txt 2>/dev/null || true
sudo sed -i 's/console=ttyAMA0,115200 //' /boot/cmdline.txt 2>/dev/null || true

echo "[5/5] 创建必要目录..."
mkdir -p export Logs Captures_Final

echo ""
echo "========== 安装完成 =========="
echo "请将 TFLite 模型文件放到 export/ 目录"
echo "然后运行："
echo "  cd rpi/"
echo "  python3 final_sorting_system.py"
echo ""
echo "如果串口通信失败，请重启树莓派后重试。"
