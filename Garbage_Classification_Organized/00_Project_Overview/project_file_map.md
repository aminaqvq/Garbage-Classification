# 文件映射表

| 原始文件 | 新位置 | 新名称 | 作用 | 最终使用 |
|----------|--------|--------|------|----------|
| training/collect_dataset.py | 01_Dataset_Collection/ | dataset_collect_images.py | 数据采集 | 训练步骤 |
| training/split_dataset.py | 02_Dataset_Splitting/ | dataset_split_train_val_test.py | 数据划分 | 训练步骤 |
| training/train_model.py | 03_Model_Training/ | model_train_mobilenetv3.py | 模型训练 | 训练步骤 |
| training/classify_live.py | 03_Model_Training/ | model_live_test_camera.py | PyTorch实时演示 | PC测试 |
| training/quantize_model.py | 04_Model_Export_Quantization/ | model_export_tflite.py | TFLite导出 | 训练步骤 |
| rpi/ai_preview.py | 05_RPi_AI_Preview/ | rpi_ai_preview_camera.py | AI预览 | 测试用 |
| **NEW** | 06_RKHO_Serial_Protocol_Test/ | rpi_manual_rkho_protocol_test.py | 手动RKHO测试 | 调试用 |
| **NEW** | 06_RKHO_Serial_Protocol_Test/ | mcu_rkho_protocol_test.c | 最小测试固件 | 调试用 |
| rpi/servo_char_control.py | 07_Full_Load_RKHO_Test/ | rpi_ai_rkho_direct_sorting_test.py | RKHO+满载(无T) | ❌过渡版 |
| firmware/mcu_full/main.c | 07_Full_Load_RKHO_Test/ | mcu_full_load_rkho_sorting_test.c | 满载+RKHO(无超声波) | ❌过渡版 |
| rpi/serial_handshake_test.py | 08_AA55_Ultrasonic_Protocol_Reference/ | rpi_aa55_serial_handshake_test_old.py | 旧AA55测试 | ❌旧协议 |
| firmware/mcu_led_protocol/ | 08_AA55_Ultrasonic_Protocol_Reference/ | mcu_ultrasonic_led_aa55_test_old.c | 旧AA55 LED固件 | ❌旧协议 |
| rpi/final_sorting_system.py | 08_AA55_Ultrasonic_Protocol_Reference/ | rpi_final_sorting_aa55_old.py | 旧final AA55 | ❌旧协议 |
| firmware/mcu_final/ | 08_AA55_Ultrasonic_Protocol_Reference/ | mcu_ultrasonic_sorting_aa55_old.c | 旧final固件(无满载) | ❌旧协议 |
| rpi/locked_trigger_system.py | 09_Triggered_RKHO_Reference/ | rpi_triggered_ai_sorting_rkho_old.py | T触发旧版 | ❌历史版 |
| **NEW** | 09_Triggered_RKHO_Reference/ | mcu_triggered_rkho_protocol_test.c | 触发协议测试固件 | 测试用 |
| **NEW** | 10_Final_Integrated_System/ | rpi_final_ai_sorting_with_full_load.py | ★ 最终上位机 | ★ YES |
| **NEW** | 10_Final_Integrated_System/ | mcu_final_ultrasonic_full_load_rkho.c | ★ 最终固件 | ★ YES |
