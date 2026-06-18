# Tests — 集成测试

## 目标

此目录用于存放五分类视觉触发系统的集成测试，包括：

- 串口协议单元测试
- 状态机转移测试
- 满载/恢复场景模拟
- 模型推理一致性验证

## 当前状态

系统核心逻辑已在 `rpi/rpi_vision_trigger_sorting.py` 中通过 `--dry-run` 验证通过。
硬件相关测试（串口、摄像头、舵机）需要在真实树莓派上运行。

## 运行

```bash
# Dry-run 预检（无硬件）
python ../rpi/rpi_vision_trigger_sorting.py --dry-run

# 仅预览
python ../rpi/rpi_vision_trigger_sorting.py --preview-only
```
