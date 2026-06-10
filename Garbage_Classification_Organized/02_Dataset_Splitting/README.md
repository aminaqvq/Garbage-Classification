# 02 — 数据集划分

## 1. 这个文件夹是干什么的

将 `dataset/` 中采集好的原始图片按 **7:2:1** 比例划分为 train / val / test 三个子集，输出到 `garbage_dataset/`。

## 2. 包含的脚本

| 文件名 | 类型 | 作用 | 是否推荐使用 |
|--------|------|------|-------------|
| `split_dataset.py` | PC Training Python | 数据集划分 | ✅ 推荐 |

## 3. 工作流程

1. 扫描 `dataset/` 下四类目录中的图片
2. 检查每张图片是否可正常打开（PIL verify）
3. 随机打乱后按 7:2:1 分配
4. 复制（默认 copy，不 move）到 `garbage_dataset/train|val|test/`
5. 生成划分日志和统计报告

## 4. 运行方法

```bash
cd 02_Dataset_Splitting/
python split_dataset.py
```

## 5. 输入输出

| 输入 | 输出 |
|------|------|
| `dataset/` | `garbage_dataset/train/` |
| | `garbage_dataset/val/` |
| | `garbage_dataset/test/` |
| | `garbage_dataset/class_names.json` |
| | `garbage_dataset/logs/` |

## 6. 注意事项

- 默认使用 copy 模式，不会改变原始 dataset/
- 固定随机种子（42），结果可复现
- 对小样本有保护（1-2 张图片也能处理）
- 这是数据集准备阶段，运行在 PC 端
