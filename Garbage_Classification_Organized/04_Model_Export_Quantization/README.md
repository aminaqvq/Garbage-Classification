# 04 — 模型量化导出

## 1. 这个文件夹是干什么的

将训练好的 PyTorch `.pt` 模型转换为树莓派可部署的 TFLite 格式，支持 FP32 / FP16 / INT8 三种精度。同时在各后端上评估精度和延迟，输出对比报告。

## 2. 包含的脚本

| 文件名 | 类型 | 作用 | 是否推荐使用 |
|--------|------|------|-------------|
| `quantize_model.py` | PC Training Python | 模型量化导出 | ✅ 推荐 |

## 3. 工作流程

```
outputs/latest_mobilenetv3_best.pt
        │
        ├─ 1) PyTorch FP32 评估
        ├─ 2) 导出 ONNX
        ├─ 3) ONNX Runtime 评估
        ├─ 4) ONNX → SavedModel → TFLite FP32
        ├─ 5) TFLite FP16 量化导出
        ├─ 6) TFLite INT8 量化导出（需校准集）
        └─ 7) 各后端精度/延迟对比
        │
        ▼
export/
  latest_tflite_fp16.tflite    ← ★ 推荐树莓派使用
  latest_tflite_int8.tflite    ← 最小体积
  latest_tflite_float32.tflite ← 无损
  latest_model.onnx
  quant_run_<时间戳>/          ← 详细报告
```

## 4. 运行方法

```bash
cd 04_Model_Export_Quantization/
python quantize_model.py
```

## 5. 导出格式对比

| 格式 | 体积 | 速度 | 精度损失 | 推荐场景 |
|------|------|------|----------|----------|
| TFLite FP32 | ~10 MB | 快 | 无 | 调试对比 |
| TFLite FP16 | ~5 MB | 更快 | 微乎其微 | ★ 树莓派推荐 |
| TFLite INT8 | ~2.5 MB | 最快 | 轻微 | 极致优化 |
| ONNX | ~10 MB | 基准 | 无 | PC 端推理 |

## 6. 部署到树莓派

将 `export/latest_tflite_fp16.tflite` 复制到树莓派的项目 `export/` 目录。

## 7. 注意事项

- 需要 TensorFlow（>2.12）、onnx、onnxruntime、onnxsim
- INT8 量化需要校准集（默认从 train 中抽取）
- `config/class_mapping.json` 的类别顺序必须与模型输出的 index 顺序一致（0=其他,1=厨余,2=可回收,3=有害）
