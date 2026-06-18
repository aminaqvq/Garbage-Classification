# 迁移审计报告 — 从四分类超声波版到五分类视觉触发版

> 版本：vision_trigger_5class_v1  
> 审计日期：2025-06-15  
> 状态：只读审计，未修改任何旧文件

---

## 1. 四分类硬编码点

以下文件/行包含四分类硬编码，需要在新版中改为五分类：

### 1.1 `config/class_mapping.json`（项目根目录）

- **路径**：`config/class_mapping.json`
- **内容**：仅包含四类映射：`其他:0, 厨余:1, 可回收:2, 有害:3`
- **状态**：❌ 需迁移 → 已被 `11_Vision_Trigger_5Class_System/config/class_mapping_5class.json` 替代
- **建议**：只读保留，新版使用新配置文件

### 1.2 `01_Dataset_Collection/dataset_collect_images.py`

- **路径**：`Garbage_Classification_Organized/01_Dataset_Collection/dataset_collect_images.py`
- **行 19**：`CLASS_NAMES = ["可回收", "有害", "厨余", "其他"]`
- **状态**：❌ 需修改 → Step 2 目标
- **建议**：改为 `CLASS_NAMES = ["待分拣", "可回收", "有害", "厨余", "其他"]` 或从配置文件读取

### 1.3 `02_Dataset_Splitting/dataset_split_train_val_test.py`

- **路径**：`Garbage_Classification_Organized/02_Dataset_Splitting/dataset_split_train_val_test.py`
- **行 25**：`CLASS_NAMES = ["可回收", "有害", "厨余", "其他"]`
- **行 400**：注释提到 `class_names.json` 保存类别顺序
- **状态**：❌ 需修改 → Step 2 目标
- **建议**：改为五分类，并从 `class_mapping_5class.json` 读取顺序

### 1.4 `03_Model_Training/model_train_mobilenetv3.py`

- **路径**：`Garbage_Classification_Organized/03_Model_Training/model_train_mobilenetv3.py`
- **行 49**：`NUM_CLASSES = 4`
- **行 198**：校验 `NUM_CLASSES` 与实际数据集类别数一致
- **行 905**：`model = build_model(NUM_CLASSES)`
- **状态**：⚠️ 后续阶段修改（Step 3+），Step 1–2 不涉及
- **建议**：改为 `NUM_CLASSES = 5` 或从数据集自动推断

### 1.5 `03_Model_Training/model_live_test_camera.py`

- **路径**：`Garbage_Classification_Organized/03_Model_Training/model_live_test_camera.py`
- **行 49**：`num_classes = len(idx_to_class)` — 动态获取，但取决于输入的 mapping
- **状态**：⚠️ 间接受影响
- **建议**：如果 mapping 正确传入五分类，无需代码修改

### 1.6 `04_Model_Export_Quantization/model_export_tflite.py`

- **路径**：`Garbage_Classification_Organized/04_Model_Export_Quantization/model_export_tflite.py`
- **行 62**：`"num_classes": "auto"` — 默认自动推断
- **行 336–360**：`get_num_classes()` 自动从 checkpoint 或 state_dict 推断
- **状态**：✅ 已支持自动推断，无需修改
- **建议**：五分类模型导出时自动适配

### 1.7 `05_RPi_AI_Preview/rpi_ai_preview_camera.py`

- **路径**：`Garbage_Classification_Organized/05_RPi_AI_Preview/rpi_ai_preview_camera.py`
- **行 65**：引用 `config/class_mapping.json`
- **行 210–233**：动态读取 `class_mapping.json` 获取 `idx_to_class`
- **状态**：⚠️ 需指向新配置文件或五分类 mapping
- **建议**：修改引用的 mapping 路径或更新 `class_mapping.json` 为五分类

### 1.8 各 RPi 脚本中的 `DEFAULT_IDX_TO_CLASS`

- `07_Full_Load_RKHO_Test/rpi_ai_rkho_direct_sorting_test.py` 行 121–126：`{0:"其他",1:"厨余",2:"可回收",3:"有害"}`
- `10_Final_Integrated_System/rpi_final_ai_sorting_with_full_load.py` 行 19：`DEFAULT_IDX={0:"其他",1:"厨余",2:"可回收",3:"有害"}`
- `09_Triggered_RKHO_Reference/rpi_triggered_ai_sorting_rkho_old.py`：四分类硬编码
- `08_AA55_Ultrasonic_Protocol_Reference/rpi_final_sorting_aa55_old.py`：四分类硬编码
- **状态**：新版脚本需要改为五分类默认映射
- **建议**：新版 RPi 脚本从 `class_mapping_5class.json` 动态读取，不硬编码

---

## 2. 超声波触发点

### 2.1 `08_AA55_Ultrasonic_Protocol_Reference/`（全部）

| 文件 | 说明 |
|------|------|
| `mcu_ultrasonic_led_aa55_test_old.c` | 旧 AA55 协议 LED 测试，含 HC-SR04、TRIG=P1^0、ECHO=P1^1、UltrasonicMeasureCm() |
| `mcu_ultrasonic_sorting_aa55_old.c` | 旧最终固件，含 HC-SR04、ObjectDetectedStable()、MCU_TRIGGER_READY=0xA1 |
| `rpi_aa55_serial_handshake_test_old.py` | 旧 AA55 握手测试，等待 MCU_TRIGGER_READY=0xA1 |
| `rpi_final_sorting_aa55_old.py` | 旧 final 版，含 TRIGGER_DEBOUNCE_SEC、MCU_TRIGGER_READY |

- **状态**：✅ 已标记为旧协议参考（`project_file_map.md` 中标注为 ❌旧协议）
- **建议**：**只读保留**，作为历史参考。不需要任何修改。

### 2.2 `10_Final_Integrated_System/mcu_final_ultrasonic_full_load_rkho.c`

- **路径**：`Garbage_Classification_Organized/10_Final_Integrated_System/mcu_final_ultrasonic_full_load_rkho.c`
- **行 35–37**：`sbit TRIG = P1^0; sbit ECHO = P1^1;`
- **行 65**：`#define MCU_TRIGGER_CHAR 'T'`
- **行 106–107**：`UltrasonicMeasureCm()`、`ObjectDetectedStable()` 声明
- **行 162–194**：`UltrasonicMeasureCm()` 实现
- **行 197–206**：`ObjectDetectedStable()` 实现
- **行 87**：`#define TRIGGER_COOLDOWN_MS 3000`
- **⚠️ 严重问题：文件在第 331 行截断** — `ReceiveClassChar(u8` 函数不完整，缺少 `main()` 函数
- **状态**：❌ 标记为最终版但不完整。新版不应以此为基础。
- **建议**：只读保留。新版 MCU 固件以 `07_Full_Load_RKHO_Test/mcu_full_load_rkho_sorting_test.c` 为起点。

### 2.3 `10_Final_Integrated_System/rpi_final_ai_sorting_with_full_load.py`

- **路径**：`Garbage_Classification_Organized/10_Final_Integrated_System/rpi_final_ai_sorting_with_full_load.py`
- **行 2**：注释声明"超声波触发"
- **行 16**：`MCU_TRIGGER_CHAR="T"`
- **行 42**：`self.state="WAIT_TRIGGER"`
- **行 89**：`elif ch=="T" and not self.full:` — 等待 T 触发
- **状态**：❌ 旧版触发模式，不适用于新版
- **建议**：只读保留。新版 RPi 脚本以 `07_Full_Load_RKHO_Test/rpi_ai_rkho_direct_sorting_test.py` 为起点。

### 2.4 `09_Triggered_RKHO_Reference/`

- `rpi_triggered_ai_sorting_rkho_old.py` 行 81：`MCU_TRIGGER_CHAR = "T"`
- 多处 `WAIT_TRIGGER` 状态和 `"等待超声波 T"` 注释
- **状态**：✅ 已标记为旧参考
- **建议**：只读保留

### 2.5 其他超声波相关文件

| 文件 | 说明 |
|------|------|
| `README.md`（项目根目录） | 多处提到 HC-SR04 超声波传感器 |
| `docs/hardware_setup.md` | 行 11–12：HC-SR04 TRIG/ECHO 接线 |
| `docs/protocol.md` | 行 10：T = 超声波检测到垃圾 |
| `docs/firmware_flash.md` | 引用最终超声波固件 |
| `examples/quick_start.sh` | 引用超声波版 |
| `Garbage_Classification_Organized/00_Project_Overview/` | wiring_summary.md 等提及 HC-SR04 |
| `Garbage_Classification_Organized/10_Final_Integrated_System/wiring_final.md` | HC-SR04 接线图 |

- **状态**：⚠️ 文档需后续更新
- **建议**：新版文档（`11_Vision_Trigger_5Class_System/`）中已包含正确信息。旧文档可在后续统一更新或标注已废弃。

---

## 3. 串口协议相关点

### 3.1 R/K/H/O 发送端

| 文件 | 位置 | 说明 |
|------|------|------|
| `07_Full_Load_RKHO_Test/rpi_ai_rkho_direct_sorting_test.py` | 行 137–142 | `CLASS_TO_SERVO_CHAR`：直接映射四类到 R/K/H/O |
| `10_Final_Integrated_System/rpi_final_ai_sorting_with_full_load.py` | 行 18 | `CLASS_TO_MCU`：同上 |
| `06_RKHO_Serial_Protocol_Test/rpi_manual_rkho_protocol_test.py` | 行 7 | `TX` 字典用于手动测试 |
| `09_Triggered_RKHO_Reference/rpi_triggered_ai_sorting_rkho_old.py` | — | 同上 |

- **状态**：R/K/H/O 映射在新版中保持不变 ✅
- **建议**：新版从 `class_mapping_5class.json` 的 `action_mapping` 读取，不硬编码

### 3.2 MCU 侧响应字符

| 文件 | 说明 |
|------|------|
| `07_Full_Load_RKHO_Test/mcu_full_load_rkho_sorting_test.c` | 仅有 F/N 反馈，无 D/E。处理 R/K/H/O 命令。**无超声波**。 |
| `10_Final_Integrated_System/mcu_final_ultrasonic_full_load_rkho.c` | 含 T/F/N/D/E 全部响应字符。但文件截断。 |
| `06_RKHO_Serial_Protocol_Test/mcu_rkho_protocol_test.c` | 最小协议测试 |

### 3.3 新版协议变化

| 字符 | 旧版 | 新版 |
|------|------|------|
| `T` | ✅ MCU→RPi（超声波触发） | ❌ 已移除 |
| `D` | ✅ MCU→RPi | ✅ 保留 |
| `F` | ✅ MCU→RPi | ✅ 保留 |
| `N` | ✅ MCU→RPi | ✅ 保留 |
| `E` | ✅ MCU→RPi（10 版新增） | ✅ 保留 |

---

## 4. 可复用模块

### 4.1 强烈推荐复用

| 模块 | 路径 | 原因 |
|------|------|------|
| **07 RPi 脚本** | `07_Full_Load_RKHO_Test/rpi_ai_rkho_direct_sorting_test.py` | ✅ 已实现"直接发送 R/K/H/O 不等 T"；✅ 含 F/N 满载处理；✅ 含 TFLite 推理管线；✅ 含 cooldown/stable_frames 逻辑。**最佳迁移起点。** |
| **07 MCU 固件** | `07_Full_Load_RKHO_Test/mcu_full_load_rkho_sorting_test.c` | ✅ 无超声波；✅ 处理 R/K/H/O + 满载检测；✅ 代码完整（无截断）。**最佳 MCU 新基线。** |
| **06 协议测试** | `06_RKHO_Serial_Protocol_Test/` | ✅ 可用于新版协议调试 |
| **04 模型导出** | `04_Model_Export_Quantization/model_export_tflite.py` | ✅ `num_classes` 自动推断，无需修改 |
| **05 AI 预览** | `05_RPi_AI_Preview/rpi_ai_preview_camera.py` | ✅ 动态读取 class_mapping，可用作纯推理测试 |

### 4.2 可参考但需修改

| 模块 | 路径 | 原因 |
|------|------|------|
| **01 数据采集** | `01_Dataset_Collection/dataset_collect_images.py` | ⚠️ 需改 CLASS_NAMES 为五分类（Step 2） |
| **02 数据集划分** | `02_Dataset_Splitting/dataset_split_train_val_test.py` | ⚠️ 需改 CLASS_NAMES 为五分类（Step 2） |
| **03 模型训练** | `03_Model_Training/model_train_mobilenetv3.py` | ⚠️ 需改 NUM_CLASSES=5（Step 3+） |

### 4.3 可复用的基础组件（来自 07 RPi 脚本）

| 组件 | 行号 | 说明 |
|------|------|------|
| TFLite 推理器 | 行 ~240–550 | `GarbageClassifierTFLite`：加载模型、预处理、推理 |
| 稳定预测器 | 行 ~580–650 | `StablePredictor`：连续帧确认 |
| 串口管理器 | 行 ~660–700 | `ServoCharSerial`：发送 R/K/H/O，接收 F/N/D/E |
| 摄像头管理器 | 行 ~700–735 | `CameraManager`：打开、读取、关闭 |
| 日志系统 | 行 189–220 | RotatingFileHandler + CSV 追加 |
| 叠加绘制 | 行 ~480–560 | `draw_overlay()`：状态信息叠加到画面 |

---

## 5. 不建议继续作为基线的模块

| 模块 | 路径 | 原因 |
|------|------|------|
| **08 全部** | `08_AA55_Ultrasonic_Protocol_Reference/` | AA55 二进制协议已废弃，超声波触发已废弃 |
| **09 RPi** | `09_Triggered_RKHO_Reference/rpi_triggered_ai_sorting_rkho_old.py` | 依赖 MCU 发 T 触发 |
| **09 MCU** | `09_Triggered_RKHO_Reference/mcu_triggered_rkho_protocol_test.c` | T 触发协议 |
| **10 RPi** | `10_Final_Integrated_System/rpi_final_ai_sorting_with_full_load.py` | 依赖 WAIT_TRIGGER + T 字符触发 |
| **10 MCU** | `10_Final_Integrated_System/mcu_final_ultrasonic_full_load_rkho.c` | 超声波触发 + **文件截断** |

---

## 6. 推荐下一步修改顺序

| 步骤 | 目标 | 涉及文件 |
|------|------|----------|
| **Step 1** ✅ | 建立重构基线 | 当前步骤（`11_Vision_Trigger_5Class_System/`） |
| **Step 2** | 五分类数据采集与划分 | `01_Dataset_Collection/dataset_collect_images.py`、`02_Dataset_Splitting/dataset_split_train_val_test.py` |
| **Step 3** | 采集「待分拣」类数据 | 新增数据采集任务 |
| **Step 4** | 重新训练五分类模型 | `03_Model_Training/model_train_mobilenetv3.py`（NUM_CLASSES=5） |
| **Step 5** | 导出五分类 TFLite 模型 | `04_Model_Export_Quantization/model_export_tflite.py`（自动适配） |
| **Step 6** | 新版 RPi 推理+状态机 | 以 `07/` 脚本为起点，实现 `state_machine_design.md` 中的状态机 |
| **Step 7** | 新版 MCU 固件 | 以 `07/` MCU 固件为起点，添加 D/E 反馈 |
| **Step 8** | 集成测试 | `tests/` 目录 |
| **Step 9** | 文档更新 | `README.md`、`docs/` 等 |

---

## 7. 下一阶段应修改的文件（Step 2 范围）

1. **`Garbage_Classification_Organized/01_Dataset_Collection/dataset_collect_images.py`**
   - 修改 `CLASS_NAMES` 从 4 类扩展到 5 类
   - 添加「待分拣」采集模式（可选：自动间隔拍摄空平台）

2. **`Garbage_Classification_Organized/02_Dataset_Splitting/dataset_split_train_val_test.py`**
   - 修改 `CLASS_NAMES` 从 4 类扩展到 5 类
   - 确保「待分拣」目录存在时的处理逻辑

3. **（可选）`config/class_mapping.json`**
   - 可选：更新为五分类，或保留旧版供旧脚本使用

---

## 8. 必须只读保留的文件

以下文件在任何阶段都**不应修改或删除**：

| 文件 | 原因 |
|------|------|
| `08_AA55_Ultrasonic_Protocol_Reference/` 全部 | 历史参考，AA55 协议存档 |
| `09_Triggered_RKHO_Reference/` 全部 | T 触发版参考 |
| `10_Final_Integrated_System/` 全部 | 旧超声波最终版存档（即使 MCU 文件截断） |
| `Captures_Final/`、`Captures_Locked_Trigger/`、`Captures_Servo_Char/` | 历史测试截图 |
| `Logs/` 全部 | 历史运行日志 |
| `outputs/` 全部 | 历史训练输出和模型检查点 |
| `dataset/` 全部 | 原始数据集，不可逆操作 |
| `garbage_dataset/` 全部 | 划分后数据集 |
| `config/class_mapping.json` | 旧版四分类配置 |

---

## 9. 潜在风险

| 风险 | 严重程度 | 说明 |
|------|----------|------|
| **`10_Final_Integrated_System/mcu_final_ultrasonic_full_load_rkho.c` 文件截断** | 🔴 高 | 第 331 行不完整，`ReceiveClassChar()` 和 `main()` 缺失。此文件声称是"最终单片机固件"但实际不可用。这解释了为什么 README 标注为最终版但不可靠。 |
| **四分类模型无法直接迁移到五分类** | 🟡 中 | 现有 TFLite 模型是四分类输出，不能简单加一类。需要重新训练。旧模型可用于验证四类垃圾识别能力，但缺少「待分拣」输出。 |
| **「待分拣」类数据采集工作量大** | 🟡 中 | 需要 400–800 张不同光照、背景、状态下的空平台图片，采集耗时。 |
| **「待分拣」recall 不足导致误触发** | 🔴 高 | 如果模型将空画面误判为垃圾类别，会在无垃圾时触发分拣动作，造成电机空转或事故。 |
| **07 MCU 固件缺少 D/E 反馈** | 🟡 中 | 当前 `mcu_full_load_rkho_sorting_test.c` 仅有 F/N 满载反馈，缺少 D（分拣完成）和 E（错误）返回。需要修改 MCU 固件以支持新版协议的 D/E。 |
| **`07_Full_Load_RKHO_Test/` 中只有 cooldown 防重复，无「回到待分拣」逻辑** | 🟡 中 | 07 版使用 `send_cooldown_seconds` 防止重复发送，但缺少 `WAIT_RETURN_TO_PENDING` 状态。新版状态机需要显著增强这部分逻辑。 |
| **旧文档多处引用超声波** | 🟢 低 | `README.md`、`docs/`、`examples/` 等需要后续更新，但不影响代码运行。 |
| **多个目录存在 class_mapping 硬编码不一致** | 🟢 低 | 各脚本的默认映射类顺序可能不一致（如 `["其他","厨余","可回收","有害"]` vs `["可回收","有害","厨余","其他"]`），新版统一从配置文件读取可消除此风险。 |
