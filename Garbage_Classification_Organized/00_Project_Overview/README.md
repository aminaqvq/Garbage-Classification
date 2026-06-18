# 项目总览 — 智能垃圾分类分拣系统（五分类视觉触发版）

## 1. 项目是什么
基于树莓派 + AI (MobileNetV3/TFLite) + STC89C52RC 单片机的智能垃圾分类系统。
五分类：待分拣（视觉等待）、其他、厨余、可回收、有害。
摄像头持续推理 → 视觉自主触发分拣 → 串口命令 MCU 执行。

## 2. 最终协议: ASCII RKHO + D/F/N/E（无 T）
MCU→RPi: D(完成) F(满载) N(恢复) E(错误)
RPi→MCU: R(可回收) K(厨余) H(有害) O(其他)
**T 触发字符和 HC-SR04 超声波已在新版中彻底移除。**

## 3. 目录导航

| 文件夹 | 内容 | 读者 |
|--------|------|------|
| 01_Dataset_Collection | 五分类图像采集 | 需要采集数据的人 |
| 02_Dataset_Splitting | 数据集划分 | 划分训练/验证/测试集 |
| 03_Model_Training | MobileNetV3 训练 | 训练模型的人 |
| 04_Model_Export_Quantization | TFLite 导出/量化 | 部署模型的人 |
| 06_RKHO_Serial_Protocol_Test | 最小串口调试工具 | 调试通信 |
| **09_Vision_Trigger_5Class_System** | **★ 最终部署系统** | **正式使用** |
| archive/ | 历史版本存档 | 参考 |

## 4. 新手路径
1. 已有模型 → 直接看 `09_Vision_Trigger_5Class_System/`
2. 需训练 → 01 → 02 → 03 → 04 → 09
3. 调试通信 → 06 手动测试

## 5. 最终部署
- 树莓派: `09_Vision_Trigger_5Class_System/rpi/rpi_vision_trigger_sorting.py`
- 单片机: `09_Vision_Trigger_5Class_System/mcu/mcu_vision_trigger_full_load_rkho.c`
- 配置: `09_Vision_Trigger_5Class_System/config/class_mapping_5class.json`
