# 03 — 模型训练

## 1. 这个文件夹是干什么的

使用 MobileNetV3（Small/Large）在划分好的数据集上训练垃圾分类模型。支持两阶段训练（先冻结 backbone 训练分类头、再解冻微调）、早停、学习率调度、类别权重平衡等特性。同时包含一个 PyTorch 实时摄像头分类演示脚本。

## 2. 包含的脚本

| 文件名 | 类型 | 作用 | 是否推荐使用 |
|--------|------|------|-------------|
| `train_model.py` | PC Training Python | MobileNetV3 训练主程序 | ✅ 推荐 |
| `classify_live.py` | PC Demo Python | PyTorch 实时摄像头分类演示 | 可选 |

## 3. 工作流程

```
garbage_dataset/train|val|test/
        │
        ▼
  train_model.py
        │
        ├─ 第1阶段：冻结 backbone，只训练分类头（前 8 epoch）
        ├─ 第2阶段：解冻 backbone，全模型微调
        ├─ 早停：连续 7 轮 val_acc 不提升自动停止
        ├─ 学习率调度：ReduceLROnPlateau
        └─ 类别权重：自动计算处理样本不均衡
        │
        ▼
  outputs/latest_mobilenetv3_best.pt
```

## 4. 运行方法

```bash
cd 03_Model_Training/
python train_model.py
```

**实时演示**：
```bash
python classify_live.py --ckpt ../outputs/latest_mobilenetv3_best.pt --cam 1
```

## 5. 关键配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_NAME` | `mobilenet_v3_large` | 模型大小 |
| `BATCH_SIZE` | 32 | 批大小 |
| `EPOCHS` | 999 | 最大训练轮（实际靠早停终止） |
| `PATIENCE` | 7 | 早停耐心 |
| `HEAD_LR` | 1e-3 | 分类头学习率 |
| `BACKBONE_LR` | 1e-4 | 微调学习率 |
| `MAX_STEPS` | 0 | 0=不限制，>0 时硬上限停止 |

## 6. 输出

```
outputs/mobilenetv3_garbage_<时间戳>/
  mobilenetv3_best.pt          ← 最佳模型
  mobilenetv3_last.pt          ← 最后一轮
  class_mapping.json           ← 类别映射
  config.json                  ← 训练配置
  val_classification_report.*  ← 验证集报告
  test_classification_report.* ← 测试集报告
  training_log.csv / batch_log.csv
outputs/latest_mobilenetv3_best.pt  ← 最佳模型副本
```

## 7. 注意事项

- 需要 PyTorch + torchvision
- 建议 GPU 训练，CPU 也可但较慢
- `classify_live.py` 是 PC 端 PyTorch 实时演示，不是树莓派版本
- 训练完成后进入 04 进行量化导出
