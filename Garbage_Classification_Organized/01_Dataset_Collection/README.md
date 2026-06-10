# 01 — 数据集采集

## 1. 用途
用摄像头采集垃圾分类图片, 保存到 `dataset/<类别>/`。

## 2. 脚本

| 文件名 | 类型 | 作用 |
|--------|------|------|
| `dataset_collect_images.py` | PC Python | 图像采集 |

## 3. 运行
```bash
python dataset_collect_images.py
```

输出: `dataset/可回收/`, `dataset/有害/`, `dataset/厨余/`, `dataset/其他/`
