# 智能垃圾分类分拣系统

基于 AI 视觉的智能垃圾分类分拣系统：摄像头采集 → 树莓派 TFLite 边缘推理 → UART 串口通信 → 52RC 单片机控制舵机/电机分拣。

## 系统架构

```
┌──────────┐    ┌─────────────┐    UART (9600bps)    ┌──────────────┐
│ 摄像头   │───▶│ 树莓派       │──────────────────▶│ STC89C52RC   │
│          │    │ AI 识别      │◀──────────────────│ 单片机       │
└──────────┘    │ (TFLite)    │   AA/CC/DD 协议     │              │
                └─────────────┘                     │ HC-SR04 超声 │
                                                    │ SG90 舵机   │
                                                    │ 直流电机    │
                                                    └──────────────┘
```

## 功能特性

- **4 类垃圾分类**：可回收、有害、厨余、其他
- **TFLite 边缘推理**：树莓派上运行 FP16/INT8 量化模型
- **超声波检测**：HC-SR04 检测物体靠近，触发识别
- **串口通信协议**：AA 帧协议（分类结果）或单字符协议（R/H/K/O）
- **端到端闭环**：识别 → 通信 → 舵机/电机分拣 → 确认
- **完整训练管线**：PC 端数据采集 → 训练 → 量化 → 导出 TFLite

## 硬件清单

| 组件 | 型号 | 数量 |
|------|------|------|
| 上位机 | 树莓派 4B / 3B+ | 1 |
| 下位机 | STC89C52RC | 1 |
| 超声波传感器 | HC-SR04 | 1 |
| 舵机 | SG90 | 1-2 |
| 直流电机 | 小型直流电机 + 驱动模块 | 1 |
| 摄像头 | USB 摄像头 / CSI | 1 |

详细的硬件接线图见 [docs/hardware_setup.md](docs/hardware_setup.md)。

## 项目目录结构

```
garbage-sorting-system/
├── README.md
├── .gitignore
│
├── firmware/                    # 单片机固件（C 源码）
│   ├── mcu_final/               # 最终版：舵机+电机+帧协议
│   ├── mcu_led_protocol/        # 精简版：仅 LED + 帧协议
│   └── mcu_servo_char/          # 单字符版：仅 R/H/K/O
│
├── rpi/                         # 树莓派上位机（Python）
│   ├── final_sorting_system.py  # ★ 推荐入口：最终分拣系统
│   ├── servo_char_control.py    # 单字符舵机控制版
│   ├── locked_trigger_system.py # 锁定触发版
│   ├── ai_preview.py            # AI 识别预览（无串口）
│   └── serial_handshake_test.py # 串口握手测试
│
├── training/                    # PC 端训练工具
│   ├── collect_dataset.py       # 图像采集
│   ├── split_dataset.py         # 样本划分
│   ├── train_model.py           # 模型训练
│   ├── quantize_model.py        # 模型量化导出
│   └── classify_live.py         # 实时分类演示
│
├── config/
│   └── class_mapping.json       # 类别映射配置
│
├── docs/                        # 文档
│   ├── hardware_setup.md        # 硬件接线
│   ├── protocol.md              # 串口通信协议
│   ├── firmware_flash.md        # 烧录固件
│   └── model_training.md        # 模型训练指南
│
├── scripts/                     # 辅助脚本
│   ├── setup_rpi.sh             # 树莓派环境安装
│   └── clean_outputs.sh         # 清理输出
│
├── data/                        # 数据目录（仅占位）
└── tests/                       # 测试
```

## 快速开始

### 1. 使用预训练模型（推荐）

1. 从 [Releases](https://github.com/your/repo/releases) 下载最新 TFLite 模型，放到 `export/` 目录
2. 将模型复制到树莓派的项目目录
3. 烧录单片机固件（见 [docs/firmware_flash.md](docs/firmware_flash.md)）
4. 在树莓派上运行：

```bash
cd rpi/
python3 final_sorting_system.py
```

### 2. 从头训练模型

#### PC 端（Windows/Linux/Mac）

```bash
# 安装依赖
pip install -r requirements.txt

# 1. 采集数据（按类别拍照）
cd training/
python collect_dataset.py

# 2. 划分数据集（7:2:1）
python split_dataset.py

# 3. 训练模型
python train_model.py

# 4. 量化导出 TFLite
python quantize_model.py
```

#### 树莓派端

```bash
# 安装依赖
bash scripts/setup_rpi.sh

# 运行分拣系统
cd rpi/
python3 final_sorting_system.py
```

## 树莓派运行参数

```bash
# 使用默认配置（自动检测项目根目录）
python3 final_sorting_system.py

# 指定项目根目录
python3 final_sorting_system.py --project-root /home/pi/garbage-sorting

# 指定模型路径
python3 final_sorting_system.py --model-path export/latest_tflite_int8.tflite

# 使用单字符协议版本
python3 servo_char_control.py --serial-port /dev/ttyAMA0
```

## 串口通信协议

两种协议可选，在 C 固件和 Python 脚本中需配对使用：

### AA 帧协议（推荐）

| 方向 | 帧 | 说明 |
|------|-----|------|
| 52RC → RPi | `0xA1` | 超声波检测到物体 |
| RPi → 52RC | `AA 01 55` | 分类结果（01=可回收, 02=有害, 03=厨余, 04=其他） |
| 52RC → RPi | `0xCC` | 分拣完成确认 |
| 52RC → RPi | `0xDD` | 流程结束 |

### 单字符协议

| 方向 | 字符 | 说明 |
|------|------|------|
| RPi → 52RC | `R` | 可回收 |
| RPi → 52RC | `H` | 有害 |
| RPi → 52RC | `K` | 厨余 |
| RPi → 52RC | `O` | 其他 |

详见 [docs/protocol.md](docs/protocol.md)。

## 固件烧录

使用 STC-ISP 软件将 `firmware/` 下的 C 文件编译并烧录到 STC89C52RC。

| 固件 | 功能 |
|------|------|
| `firmware/mcu_final/` | 完整版：超声波 + 舵机 + 电机 + AA 帧协议 |
| `firmware/mcu_led_protocol/` | 精简版：超声波 + LED + AA 帧协议 |
| `firmware/mcu_servo_char/` | 单字符版：舵机 + 电机 + 单字符协议 |

详见 [docs/firmware_flash.md](docs/firmware_flash.md)。

## 依赖

### PC 训练环境

```
torch >= 2.0
torchvision >= 0.15
tensorflow >= 2.12
onnx, onnxruntime, onnxsim
scikit-learn, matplotlib, pillow
opencv-python
```

### 树莓派运行环境

```
opencv-python
numpy, pillow
pyserial
tflite-runtime  # 或 tensorflow
```

## 许可证

[GPL-3.0](LICENSE)

## 致谢

本项目为课程设计/毕业设计作品，使用了以下开源技术：
- [PyTorch](https://pytorch.org/) / [TorchVision](https://pytorch.org/vision/) — MobileNetV3 模型
- [TensorFlow Lite](https://www.tensorflow.org/lite) — 边缘推理
- [OpenCV](https://opencv.org/) — 图像处理
