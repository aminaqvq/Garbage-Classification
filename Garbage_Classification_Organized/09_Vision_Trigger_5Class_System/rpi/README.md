# RPi — 树莓派最终运行脚本

## 文件

`rpi_vision_trigger_sorting.py` — 五分类视觉触发垃圾分类分拣系统最终上位机。

## 快速运行

```bash
# 预检
python rpi_vision_trigger_sorting.py --dry-run

# 完整运行
python rpi_vision_trigger_sorting.py

# 仅预览
python rpi_vision_trigger_sorting.py --preview-only

# 发送测试字符
python rpi_vision_trigger_sorting.py --test-char R
```

## 状态机

IDLE_WAIT_VISUAL → CANDIDATE_DETECTED → SEND_SORT_COMMAND → WAIT_MCU_DONE → WAIT_RETURN_TO_PENDING
任意状态收到 F → FULL_PAUSED → 收到 N → IDLE_WAIT_VISUAL
收到 E 或超时 → ERROR_RECOVERY

## 配置

- 类别映射: `../config/class_mapping_5class.json`
- 运行参数: `../config/runtime_config.example.json`
- 日志: `../../Logs/`
- 截图: `../../Captures_Vision_Trigger/`
