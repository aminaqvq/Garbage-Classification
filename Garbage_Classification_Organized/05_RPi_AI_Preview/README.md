# 05 — 树莓派 AI 识别预览

## 1. 这个文件夹是干什么的

在树莓派上运行 TFLite 模型进行实时垃圾分类识别预览。**不涉及任何串口通信**，纯粹测试模型在树莓派上的推理效果和帧率。

## 2. 包含的脚本

| 文件名 | 类型 | 作用 | 是否推荐使用 |
|--------|------|------|-------------|
| `ai_preview.py` | RPi Python | TFLite 实时 AI 预览 | 测试用 |

## 3. 工作流程

1. 加载 TFLite 模型（默认 `export/latest_tflite_fp16.tflite`）
2. 打开摄像头
3. 每帧截取中心 ROI 区域
4. TFLite 推理 → 显示识别类别和置信度
5. 不发送任何串口数据
6. 按 s 截图，q 退出

## 4. 运行方法

```bash
cd 05_RPi_AI_Preview/
python3 ai_preview.py
```

## 5. 注意事项

- 这不是最终运行版本，仅用于测试模型在树莓派上的识别效果
- 无串口通信，无法控制硬件
- 需要 `tflite-runtime`（推荐）或 `tensorflow`
- 模型默认路径：`<项目根>/export/latest_tflite_fp16.tflite`
