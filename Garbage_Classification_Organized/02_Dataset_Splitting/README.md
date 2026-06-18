# 02 — 五分类数据集划分

## 脚本

`dataset_split_train_val_test.py` — 将采集好的五分类图像按比例划分为 train/val/test。

## 输出

- `garbage_dataset/class_names.json`
- `garbage_dataset/class_mapping.json`
- `garbage_dataset/split_summary.json`

## 运行

```bash
python dataset_split_train_val_test.py --dry-run
python dataset_split_train_val_test.py
```

## 配置

- 类别映射: `09_Vision_Trigger_5Class_System/config/class_mapping_5class.json`
- 输入: `dataset/`
- 输出: `garbage_dataset/`
