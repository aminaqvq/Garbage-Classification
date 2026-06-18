# 03 — MobileNetV3 五分类模型训练

## 脚本

| 文件 | 作用 |
|------|------|
| `model_train_mobilenetv3.py` | 训练 MobileNetV3 五分类模型 |
| `model_live_test_camera.py` | 实时摄像头测试已训练模型 |

## 运行

```bash
# 训练
python model_train_mobilenetv3.py --dry-run
python model_train_mobilenetv3.py

# 实时测试
python model_live_test_camera.py --model models/vision_trigger_5class_mobilenetv3/latest_mobilenetv3_best.pt --dry-run
python model_live_test_camera.py --model models/vision_trigger_5class_mobilenetv3/latest_mobilenetv3_best.pt
```

## 配置

- 类别映射: `09_Vision_Trigger_5Class_System/config/class_mapping_5class.json`
- 数据目录: `garbage_dataset/`
- 模型输出: `models/vision_trigger_5class_mobilenetv3/`
