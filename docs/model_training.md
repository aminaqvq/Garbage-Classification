# 模型训练指南

## 概述

使用 MobileNetV3（Small/Large）在 PC 端训练垃圾分类模型，支持 PyTorch → ONNX → TFLite 完整量化管线。

## 环境准备

### PC 端（GPU 推荐）

```bash
# CUDA 12.1 + PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 其他依赖
pip install tensorflow onnx onnxruntime onnxsim scikit-learn matplotlib pillow opencv-python
```

### 仅 CPU 训练

```bash
pip install torch torchvision tensorflow onnx onnxruntime onnxsim scikit-learn matplotlib pillow opencv-python
```

## 训练流程

### 1. 数据采集

```bash
cd training/
python collect_dataset.py
```

- 选择要采集的类别（可回收/有害/厨余/其他）
- 按空格开始自动采集
- 画面中显示清晰度、亮度等质量指标
- 图片自动保存到 `dataset/<类别>/`

### 2. 数据集划分

```bash
python split_dataset.py
```

- 按 7:2:1 划分 train/val/test
- 输出到 `garbage_dataset/`
- 同时生成 `class_names.json` 映射文件

### 3. 训练模型

```bash
python train_model.py
```

训练配置（可在脚本顶部修改）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_NAME` | `mobilenet_v3_large` | 模型大小 |
| `BATCH_SIZE` | 32 | 批大小 |
| `EPOCHS` | 999 | 最大训练轮数（靠早停终止） |
| `HEAD_LR` | 1e-3 | 分类头学习率 |
| `BACKBONE_LR` | 1e-4 | 微调学习率 |
| `PATIENCE` | 7 | 早停耐心 |
| `USE_CLASS_WEIGHTS` | True | 类别平衡 |

输出：
- `outputs/mobilenetv3_garbage_<时间戳>/` — 训练产物
- `outputs/latest_mobilenetv3_best.pt` — 最佳模型副本

### 4. 量化导出

```bash
python quantize_model.py
```

导出格式：

| 格式 | 精度 | 体积 | 速度 |
|------|------|------|------|
| PyTorch FP32 | 基准 | ~10 MB | 基准 |
| ONNX FP32 | 与 PyTorch 一致 | ~10 MB | 稍快 |
| TFLite FP32 | 无损 | ~10 MB | 快 |
| TFLite FP16 | 近似无损 | ~5 MB | 更快 |
| TFLite INT8 | 轻微损失 | ~2.5 MB | 最快 |

推荐树莓派使用 `latest_tflite_fp16.tflite`（精度和速度平衡）。

### 5. 部署到树莓派

将 `export/latest_tflite_fp16.tflite` 复制到树莓派项目目录的 `export/` 下即可。

## 模型架构

```
MobileNetV3-Large
├── features (backbone)
│   └── 15 个 inverted residual blocks
└── classifier
    ├── Linear(960 → 1280)
    ├── Hardsigmoid
    └── Linear(1280 → 4)
```

## 训练策略

1. **第一阶段**（前 8 个 epoch）：冻结 backbone，只训练分类头
2. **第二阶段**（第 9 个 epoch 起）：解冻 backbone，全模型微调
3. **学习率调度**：ReduceLROnPlateau（val_acc 不提升时自动减半）
4. **早停**：连续 7 轮无提升自动停止
5. **类别权重**：自动计算以处理类别不均衡
