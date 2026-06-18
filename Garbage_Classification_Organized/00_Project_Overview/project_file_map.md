# 项目文件地图（五分类视觉触发版）

## 活动目录

| 目录 | 核心文件 | 作用 |
|------|---------|------|
| 01_Dataset_Collection | dataset_collect_images.py | 五分类图像采集 |
| 02_Dataset_Splitting | dataset_split_train_val_test.py | 划分五分类数据集 |
| 03_Model_Training | model_train_mobilenetv3.py, model_live_test_camera.py | 训练+测试 |
| 04_Model_Export_Quantization | model_export_tflite.py | 导出 TFLite |
| 06_RKHO_Serial_Protocol_Test | rpi_manual_rkho_protocol_test.py, mcu_rkho_protocol_test.c | 串口调试 |
| **09_Vision_Trigger_5Class_System** | rpi/rpi_vision_trigger_sorting.py, mcu/mcu_vision_trigger_full_load_rkho.c | **★ 最终系统** |

## 归档目录

| 目录 | 原因 |
|------|------|
| archive/05_RPi_AI_Preview | 与 07 重复，四分类预览，功能已合并到 09 |
| archive/07_Full_Load_RKHO_Test | 四分类基线，代码已迁移到 09 |
| archive/08_Final_Integrated_System | 损坏截断，超声波版，与视觉触发方向冲突 |
