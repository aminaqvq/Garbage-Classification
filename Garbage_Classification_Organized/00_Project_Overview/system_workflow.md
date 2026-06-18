# 系统工作流（五分类视觉触发版）

## 最终部署流程

```
01_Dataset_Collection   → 采集五分类图像
02_Dataset_Splitting     → 划分 train/val/test
03_Model_Training        → 训练 MobileNetV3 模型
04_Model_Export_Quantization → 导出 TFLite
09_Vision_Trigger_5Class_System → 树莓派 + MCU 部署运行
```

## 运行时流程

```
摄像头持续推理
→ 五分类模型输出
→ 如果是「待分拣」→ 不发送串口
→ 如果稳定识别到「其他/厨余/可回收/有害」→ 发送 O/K/R/H
→ MCU 执行舵机+电机动作
→ MCU 返回 D
→ 树莓派等待画面稳定回到「待分拣」
→ 允许下一轮分拣
→ 满载时 MCU 返回 F，树莓派暂停
→ 满载解除 MCU 返回 N，树莓派恢复
```
