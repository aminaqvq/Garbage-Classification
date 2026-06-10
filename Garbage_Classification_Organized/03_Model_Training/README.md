# 03 — 模型训练

## 1. 用途
MobileNetV3 训练 + PyTorch 实时摄像头演示。

## 2. 脚本

| 文件名 | 类型 | 作用 |
|--------|------|------|
| `model_train_mobilenetv3.py` | PC Python | 训练主程序 |
| `model_live_test_camera.py` | PC Python | PyTorch 实时演示 |

## 3. 运行
```bash
python model_train_mobilenetv3.py
python model_live_test_camera.py --ckpt ../outputs/latest_mobilenetv3_best.pt
```

输出: `outputs/latest_mobilenetv3_best.pt`
