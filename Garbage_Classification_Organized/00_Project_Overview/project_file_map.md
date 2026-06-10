# 原始文件 → 新位置映射表

| 原始路径 | 新路径 | 类型 | 作用 | 备注 |
|----------|--------|------|------|------|
| `training/collect_dataset.py` | `01_Dataset_Collection/collect_dataset.py` | Training Python | 摄像头采集垃圾分类图片 | |
| `training/split_dataset.py` | `02_Dataset_Splitting/split_dataset.py` | Training Python | 按 7:2:1 划分数据集 | |
| `training/train_model.py` | `03_Model_Training/train_model.py` | Training Python | MobileNetV3 训练 | |
| `training/classify_live.py` | `03_Model_Training/classify_live.py` | Training Python | PyTorch 实时摄像头分类演示 | PC 端使用 |
| `training/quantize_model.py` | `04_Model_Export_Quantization/quantize_model.py` | Training Python | .pt → ONNX → TFLite 量化 | |
| `rpi/ai_preview.py` | `05_RPi_AI_Preview/ai_preview.py` | RPi Python | 树莓派 TFLite AI 预览 | 无串口通信 |
| `rpi/serial_handshake_test.py` | `06_Serial_Handshake_Test/serial_handshake_test.py` | Test Script | 串口 AA 帧协议手动测试 | 调试工具 |
| `firmware/mcu_led_protocol/52rc_ultrasonic_uart_led_test.c` | `06_Serial_Handshake_Test/mcu_led_test_firmware.c` | MCU C | 超声波+LED+UART 测试固件 | 可配合握手测试 |
| `rpi/servo_char_control.py` | `07_Servo_Char_Control_Version/servo_char_control.py` | RPi Python | 单字符 R/H/K/O 舵机控制版 | 过渡版本 |
| `firmware/mcu_full/main.c` | `07_Servo_Char_Control_Version/mcu_full_main.c` | MCU C | 单字符协议固件（含 F/N 满载） | 配对 R/H/K/O |
| `firmware/mcu_led_protocol/52rc_ultrasonic_uart_led_test.c` | `08_Ultrasonic_LED_UART_Test_Version/ultrasonic_led_test.c` | MCU C | 超声波+LED 帧协议测试版 | |
| `firmware/mcu_led_protocol/main.c` | `08_Ultrasonic_LED_UART_Test_Version/main_duplicate.c` | MCU C | 同上（几乎完全相同的副本） | ⚠ 重复文件 |
| `rpi/locked_trigger_system.py` | `09_Locked_Trigger_Version/locked_trigger_system.py` | RPi Python | T/A/D/E 锁定触发版 | ⚠ 无匹配固件 |
| `rpi/final_sorting_system.py` | `10_Final_Integrated_System/final_sorting_system.py` | RPi Python | ★ 最终版上位机 | AI + AA 帧协议 |
| `firmware/mcu_final/mcu_52rc_garbage_sorting_final.c` | `10_Final_Integrated_System/mcu_52rc_garbage_sorting_final.c` | MCU C | ★ 最终版下位机 | 超声波+舵机+电机+LED |
| `rpi/servo_char_control.py.bak` | `99_Archive_Legacy/servo_char_control.py.bak` | Legacy | servo_char 的旧备份 | 旧版，说配对 servo_control.c |
| `firmware/mcu_servo_char/servo_control.c` | `99_Archive_Legacy/servo_control_basic.c` | Legacy | 基础舵机控制固件 | 旧版，无满载检测 |

**说明**：`firmware/mcu_led_protocol/` 中的 `main.c` 和 `52rc_ultrasonic_uart_led_test.c` 是几乎完全相同的副本（仅 WAIT_TIMEOUT 参数不同），两者均保留在 08 文件夹中并标注了重复关系。
