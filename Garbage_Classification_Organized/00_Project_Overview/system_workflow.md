# 系统完整工作流程

## 一、数据准备阶段（PC 端）

```
01_Dataset_Collection       02_Dataset_Splitting
    │                            │
    │ collect_dataset.py         │ split_dataset.py
    │ 摄像头拍照 → dataset/      │ dataset/ → garbage_dataset/
    │                            │   train/ (70%)
    │                            │   val/   (20%)
    │                            │   test/  (10%)
    ▼                            ▼
```

1. **采集数据**：运行 `collect_dataset.py`，按四类（可回收/有害/厨余/其他）拍照保存到 `dataset/`
2. **划分数据**：运行 `split_dataset.py`，将数据按 7:2:1 划分为 train/val/test 放入 `garbage_dataset/`

## 二、模型训练阶段（PC 端，建议有 GPU）

```
03_Model_Training            04_Model_Export_Quantization
    │                            │
    │ train_model.py             │ quantize_model.py
    │ garbage_dataset/ → 训练    │ .pt → ONNX → TFLite
    │ outputs/*.pt              │ export/*.tflite
    │                            │
    │ classify_live.py           │
    │ PyTorch 实时验证           │
    ▼                            ▼
```

3. **训练模型**：运行 `train_model.py`，输出 `outputs/latest_mobilenetv3_best.pt`
4. **量化导出**：运行 `quantize_model.py`，转换 .pt → ONNX → TFLite（FP16/INT8）
5. **实时验证**（可选）：运行 `classify_live.py` 在 PC 上实时测试 PyTorch 模型

## 三、树莓派部署阶段

```
05_RPi_AI_Preview
    │
    │ ai_preview.py
    │ 摄像头 + TFLite 推理
    │ 无串口，仅验证模型
    ▼
    
10_Final_Integrated_System  ← ★ 最终部署
    │
    │ final_sorting_system.py (树莓派)
    │ mcu_52rc_garbage_sorting_final.c (单片机)
    ▼
```

6. **AI 预览测试**：在树莓派上运行 `ai_preview.py`，验证 TFLite 模型识别效果
7. **最终部署**：烧录 MCU 固件 + 运行 RPi 脚本

## 四、最终系统运行时序

```
┌─────────────────────────────────────────────────────────┐
│ 1. MCU 超声波 HC-SR04 持续测距                            │
│ 2. 物体进入检测范围 → MCU 发送 0xA1                      │
│ 3. RPi 收到 0xA1 → 触发 AI 识别                         │
│ 4. RPi TFLite 连续推理 → 达到稳定帧数 → 确定分类           │
│ 5. RPi 发送 AA xx 55 给 MCU                              │
│ 6. MCU 收到有效帧 → 回复 0xCC (ACK)                      │
│ 7. MCU 控制舵机转到对应角度                                │
│ 8. MCU 控制电机开门 → 停 → 关门 → 停                      │
│ 9. MCU 动作完成 → 回复 0xDD (DONE)                       │
│10. RPi 记录日志 → 等待下一轮                              │
└─────────────────────────────────────────────────────────┘
```

## 五、备选方案（非最终版）

### 单字符协议版（07_Servo_Char_Control_Version）
```
RPi 识别完成 → 发送 R/H/K/O 单字符 → MCU 直接执行动作
无超声波触发、无 ACK/DONE 握手
```

### 锁定触发版（09_Locked_Trigger_Version）
```
MCU 发送 T → RPi 识别一次 → RPi 发送 R/H/K/O → 等待 MCU 回复 A/D/E
⚠ 无完全匹配的 C 固件
```
