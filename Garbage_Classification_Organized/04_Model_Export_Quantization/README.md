# 04 — 模型量化导出

## 1. 用途
.pt → ONNX → TFLite (FP32/FP16/INT8) 量化导出。

## 2. 脚本

| 文件名 | 类型 | 作用 |
|--------|------|------|
| `model_export_tflite.py` | PC Python | 量化导出 |

## 3. 运行
```bash
python model_export_tflite.py
```

输出: `export/latest_tflite_fp16.tflite` (推荐树莓派使用)
