# RPi — 树莓派最终运行脚本

## 文件

`rpi_vision_trigger_sorting.py` — 五分类视觉触发垃圾分类分拣系统最终上位机。

## 快速运行

```bash
# 预检
python rpi_vision_trigger_sorting.py --dry-run

# 完整运行
python rpi_vision_trigger_sorting.py

# 无窗口运行（GPIO14/15 UART，默认 /dev/ttyAMA0）
python rpi_vision_trigger_sorting.py --no-window

# 仅预览
python rpi_vision_trigger_sorting.py --preview-only

# 发送测试字符（真实动作，确认安全后使用）
python rpi_vision_trigger_sorting.py --test-char R --serial-port /dev/ttyAMA0
```

## 状态机

IDLE_WAIT_VISUAL → CANDIDATE_DETECTED → SEND_SORT_COMMAND → WAIT_MCU_DONE → WAIT_RETURN_TO_PENDING
任意状态收到 F → FULL_PAUSED → 收到 N → IDLE_WAIT_VISUAL
收到 E 或超时 → ERROR_RECOVERY → 锁定退出

## 配置

- 类别映射: `../config/class_mapping_5class.json`
- 运行参数: `../config/runtime_config.example.json`
- 默认串口: `/dev/ttyAMA0`（GPIO14 TXD0、GPIO15 RXD0，9600 bps）
- 默认 D 超时: `15` 秒；MCU 固件动作约 11 秒
- 日志: `../../Logs/`
- 截图: `../../Captures_Vision_Trigger/`
