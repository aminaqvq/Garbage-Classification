# 09 — 五分类视觉触发最终系统

> 版本：vision_trigger_5class_v1
> 状态：**最终可运行系统**

---

## 1. 系统目标

彻底抛弃 HC-SR04 超声波传感器，改为**摄像头持续五分类推理 → 视觉自主触发分拣**。

核心变化：
- 不再需要 MCU 发送 `T` 触发 AI 识别。
- 树莓派始终保持推理状态，自主判断何时发送分拣命令。
- 新增第五类「待分拣」作为视觉等待状态。

---

## 2. 五分类定义

| 索引 | 类别 | 含义 | 发送串口命令 |
|------|------|------|-------------|
| 0 | 待分拣 | 视觉等待状态：画面中无垃圾 | **否** |
| 1 | 其他 | 其他垃圾 | 是 → `O` |
| 2 | 厨余 | 厨余垃圾 | 是 → `K` |
| 3 | 可回收 | 可回收垃圾 | 是 → `R` |
| 4 | 有害 | 有害垃圾 | 是 → `H` |

---

## 3. 串口协议

### RPi → MCU

| 字符 | 含义 |
|------|------|
| `R` | 可回收 |
| `K` | 厨余 |
| `H` | 有害 |
| `O` | 其他 |

### MCU → RPi

| 字符 | 含义 |
|------|------|
| `D` | 分拣完成 |
| `F` | 满载暂停 |
| `N` | 满载解除 |
| `E` | 错误 |

---

## 4. 快速运行命令

```bash
# 预检（不打开硬件，只检查配置）
python 09_Vision_Trigger_5Class_System/rpi/rpi_vision_trigger_sorting.py --dry-run

# 仅预览识别（不发送串口）
python 09_Vision_Trigger_5Class_System/rpi/rpi_vision_trigger_sorting.py --preview-only

# 完整运行（摄像头 + 串口 + 状态机）
python 09_Vision_Trigger_5Class_System/rpi/rpi_vision_trigger_sorting.py

# 无窗口模式（SSH 运行）
python 09_Vision_Trigger_5Class_System/rpi/rpi_vision_trigger_sorting.py --no-window

# 手动测试单个字符
python 09_Vision_Trigger_5Class_System/rpi/rpi_vision_trigger_sorting.py --test-char R

# 串口调试工具
python 06_RKHO_Serial_Protocol_Test/rpi_manual_rkho_protocol_test.py --serial-port /dev/ttyUSB0
```

---

## 5. 目录结构

```
09_Vision_Trigger_5Class_System/
├── README.md                           ← 本文件
├── protocol_vision_trigger.md          ← 串口协议详细文档
├── state_machine_design.md             ← 状态机设计文档
├── wiring_vision_trigger.md            ← 接线说明（无 HC-SR04）
├── dataset_spec_5class.md              ← 五分类数据集规范
├── config/
│   ├── class_mapping_5class.json       ← 五分类映射（权威来源）
│   └── runtime_config.example.json     ← 运行参数模板
├── rpi/
│   ├── rpi_vision_trigger_sorting.py   ← ★ 最终树莓派运行脚本
│   └── README.md
├── mcu/
│   ├── mcu_vision_trigger_full_load_rkho.c  ← ★ 最终 MCU 固件
│   └── README.md
├── tests/
│   └── README.md
└── docs/archive/                       ← 历史开发报告
```

---

## 6. 状态机说明

```
BOOT → IDLE_WAIT_VISUAL → CANDIDATE_DETECTED → SEND_SORT_COMMAND
→ WAIT_MCU_DONE → WAIT_RETURN_TO_PENDING → IDLE_WAIT_VISUAL

任意状态收到 F → FULL_PAUSED → 收到 N → IDLE_WAIT_VISUAL
收到 E 或 D 超时 → ERROR_RECOVERY → IDLE_WAIT_VISUAL
```

详见 `state_machine_design.md`。

---

## 7. 训练到部署流程

```
01_Dataset_Collection     → 采集五分类图片（含「待分拣」）
02_Dataset_Splitting       → 划分 train/val/test
03_Model_Training          → 训练 MobileNetV3 五分类模型
04_Model_Export_Quantization → 导出 TFLite（float32/float16）
09_Vision_Trigger_5Class_System → 部署运行
```

---

## 8. MCU 固件烧录

使用 STC-ISP 软件将 `mcu/mcu_vision_trigger_full_load_rkho.c` 编译并烧录到 STC89C52RC。

Keil C51 编译注意事项：
- 选择芯片：STC89C52RC
- 晶振频率：11.0592 MHz
- 输出 HEX 文件后使用 STC-ISP 下载

---

## 9. 旧版对比

| 对比维度 | 旧版（超声波触发） | 新版（视觉触发） |
|----------|-------------------|-----------------|
| 触发方式 | MCU 超声波检测 → 发 T | RPi 摄像头持续推理，自主触发 |
| 传感器 | HC-SR04 + 摄像头 | 仅摄像头 |
| 类别数 | 4 | 5（+ 待分拣） |
| 串口协议 | T/F/N/D/E + R/K/H/O | F/N/D/E + R/K/H/O（T 已移除） |
