# 01 — 五分类图像采集

## 脚本

`dataset_collect_images.py` — 摄像头采集五分类垃圾图像。

## 五分类

待分拣、其他、厨余、可回收、有害

## 运行

```bash
python dataset_collect_images.py
python dataset_collect_images.py --list-classes
```

## 配置

- 类别映射: `09_Vision_Trigger_5Class_System/config/class_mapping_5class.json`
- 数据目录: `dataset/`
- 输出日志: `dataset/capture_log.csv`
