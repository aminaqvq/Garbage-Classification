# 项目总览 — 智能垃圾分类分拣系统

## 1. 这是什么项目

基于 **树莓派 + 摄像头 + AI 模型（MobileNetV3/TFLite）+ STC89C52RC 单片机** 的智能垃圾分类系统。摄像头拍摄垃圾图像，树莓派运行 TFLite 模型进行 4 类识别，通过串口（UART 9600bps）将分类结果发送给 52RC 单片机，单片机控制舵机和直流电机完成物理分拣。

## 2. 四大类别

| 序号 | 类别   | 模型 Index | LED 指示 |
|------|--------|-----------|----------|
| 1    | 可回收  | 2         | LED_RECOVERABLE (P2^0) |
| 2    | 有害    | 3         | LED_HARMFUL (P2^2) |
| 3    | 厨余    | 1         | LED_KITCHEN (P2^1) |
| 4    | 其他    | 0         | LED_OTHER (P2^3) |

## 3. 目录怎么读

| 文件夹 | 内容 | 目标读者 |
|--------|------|----------|
| `01_Dataset_Collection/` | 摄像头采集垃圾分类图片 | 需要自己采集数据的人 |
| `02_Dataset_Splitting/` | 按 7:2:1 划分 train/val/test | 同上 |
| `03_Model_Training/` | MobileNetV3 训练 + 实时演示 | 需要训练模型的人 |
| `04_Model_Export_Quantization/` | .pt → ONNX → TFLite 量化导出 | 同上 |
| `05_RPi_AI_Preview/` | 树莓派摄像头 AI 预览（无串口） | 测试模型准确率的人 |
| `06_Serial_Handshake_Test/` | 串口握手协议调试工具 | 调试串口通信的人 |
| `07_Servo_Char_Control_Version/` | R/H/K/O 单字符舵机控制版 | 教学/简化版使用者 |
| `08_Ultrasonic_LED_UART_Test_Version/` | 超声波+LED+帧协议测试版 | 协议验证/调试 |
| `09_Locked_Trigger_Version/` | T/A/D/E 锁定触发版 | ⚠ 无匹配固件 |
| `10_Final_Integrated_System/` | **★ 最终推荐部署版本** | **正式使用者** |
| `99_Archive_Legacy/` | 历史备份/旧版文件 | 参考 |

## 4. 新手从哪里开始

1. **如果已经有训练好的模型**：直接看 `10_Final_Integrated_System/`
2. **如果需要从头训练模型**：按 `01→02→03→04→10` 顺序
3. **如果只想了解系统架构**：先看本文和 `system_workflow.md`

## 5. 最终运行版本

**文件夹 `10_Final_Integrated_System/`**：
- 树莓派运行：`final_sorting_system.py`
- 单片机烧录：`mcu_52rc_garbage_sorting_final.c`
- 协议：**AA xx 55 帧协议**（带 0xA1/0xCC/0xDD 握手）

## 6. 硬件清单

| 组件 | 型号 | 数量 |
|------|------|------|
| 上位机 | 树莓派 4B / 3B+ | 1 |
| 下位机 | STC89C52RC | 1 |
| 超声波传感器 | HC-SR04 | 1 |
| 舵机 | SG90 | 1-2 |
| 直流电机 + 驱动 | L298N / L9110S | 1 |
| LED 指示灯 | 5mm LED × 5 | 5 |
| 摄像头 | USB / CSI | 1 |

## 7. 配套文档

- `project_file_map.md` — 原始文件 → 新位置映射表
- `system_workflow.md` — 完整数据流和控制流
- `wiring_summary.md` — 硬件接线汇总
- `serial_protocol_summary.md` — 所有串口协议对比
