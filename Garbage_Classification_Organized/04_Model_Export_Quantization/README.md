# 04 — TFLite 模型导出与量化

## 脚本

`model_export_tflite.py` — PyTorch → ONNX → TFLite 导出 + float16/int8 量化。

## 运行

```bash
# 预检
python model_export_tflite.py --model-path models/vision_trigger_5class_mobilenetv3/latest_mobilenetv3_best.pt --output-dir models/vision_trigger_5class_tflite --class-config 09_Vision_Trigger_5Class_System/config/class_mapping_5class.json --data-dir garbage_dataset --num-classes auto --dry-run

# 导出 + float16 + 验证
python model_export_tflite.py --model-path models/vision_trigger_5class_mobilenetv3/latest_mobilenetv3_best.pt --output-dir models/vision_trigger_5class_tflite --class-config 09_Vision_Trigger_5Class_System/config/class_mapping_5class.json --data-dir garbage_dataset --num-classes auto --quantize-float16 --verify --verify-samples 100
```

## 输出

- `latest_tflite_float32.tflite`
- `export_<timestamp>/` 详细导出结果
- `saved_model/model_float32_simplified_float16.tflite`
