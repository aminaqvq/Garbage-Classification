# 02 — 数据集划分

## 1. 用途
将 dataset/ 按 7:2:1 划分为 train/val/test 放入 garbage_dataset/。

## 2. 脚本

| 文件名 | 类型 | 作用 |
|--------|------|------|
| `dataset_split_train_val_test.py` | PC Python | 数据划分 |

## 3. 运行
```bash
python dataset_split_train_val_test.py
```

输入: `dataset/` → 输出: `garbage_dataset/train|val|test/`
