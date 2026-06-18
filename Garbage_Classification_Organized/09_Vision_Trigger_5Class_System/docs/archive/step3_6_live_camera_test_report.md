# Step 3.6 报告 — 实时摄像头模型测试脚本适配

> 日期：2025-06-15  
> 文件：`03_Model_Training/model_live_test_camera.py`

---

## 1. 改造内容

| 改造项 | 旧版 | 新版 |
|--------|------|------|
| 类别读取 | 仅从 checkpoint 读取 `class_to_idx` | checkpoint → 权威 `class_mapping_5class.json`，验证必须 5 类 |
| 预处理 | 缺少 `Normalize` | Resize 256→CenterCrop 224→ToTensor→ImageNet Normalize（与训练一致） |
| 稳定预测 | 无 | `StablePredictor`，连续 N 帧一致 + 平均置信度 ≥ 阈值 |
| ROI | 不支持 | 支持 `roi_config.json` 加载/保存 + 鼠标拖拽选区 |
| 显示 | `cv2.putText` 英文 | PIL 中文面板：top1-3 类别+置信度/稳定状态/理论动作/FPS |
| 理论动作 | 无 | 四类垃圾稳定时显示 R/K/H/O，待分拣显示"不触发分拣" |
| 串口 | — | **不发送串口，不控制 MCU** |
| argparse | 4 参数 | 12 参数：`--model/--class-config/--roi-config/--no-roi/--confidence-threshold/--stable-frames/--camera-index/--device/--save-snapshots/--dry-run` 等 |
| dry-run | 无 | 验证模型/类别/ROI，不打开摄像头 |

---

## 2. 与原脚本的兼容性

- 旧脚本 130 行 → 新脚本完全重写，功能大幅增强
- 旧脚本未被删除（git 可回溯）
- 新脚本保留原 `--ckpt` 功能（通过 `--model` 参数，语义一致）
- 类别从 checkpoint 自动读取，兼容旧四分类 checkpoint（但会因类别数 ≠ 5 报错）

---

## 3. 关键功能

### 3.1 显示内容

- top1 类别 + 置信度
- top2/top3 类别 + 置信度
- 稳定状态（绿=待分拣不触发 / 橙黄=垃圾→理论动作 R/K/H/O）
- FPS + 设备
- ROI 红框

### 3.2 按键操作

| 键 | 功能 |
|----|------|
| Q | 退出 |
| S | 保存原图 + ROI crop 快照 |
| O | 保存当前 ROI 配置 |
| C | 清除 ROI |
| R | 鼠标拖拽重新选择 ROI |
| V | 显示/隐藏详情面板 |
| +/- | 调整置信度阈值 |
| [/] | 调整稳定帧数 |

### 3.3 干运行

```cmd
"D:\SoftWare\miniconda3\envs\yunet\python.exe" "03_Model_Training\model_live_test_camera.py" --dry-run
```

验证：模型存在、类别 5、输出层 5、ROI 配置状态，不打开摄像头。

---

## 4. 实际运行命令 (Windows cmd)

```cmd
cd /d "D:\Garbage Classification\Garbage_Classification_Organized"

REM dry-run
"D:\SoftWare\miniconda3\envs\yunet\python.exe" "03_Model_Training\model_live_test_camera.py" --dry-run

REM 实时测试 (camera 1)
"D:\SoftWare\miniconda3\envs\yunet\python.exe" "03_Model_Training\model_live_test_camera.py" --camera-index 1 --confidence-threshold 0.82 --stable-frames 5

REM 实时测试 (camera 0, 无 ROI)
"D:\SoftWare\miniconda3\envs\yunet\python.exe" "03_Model_Training\model_live_test_camera.py" --camera-index 0 --no-roi
```

---

## 5. 状态声明

- ✅ 脚本改造完成
- ✅ 无四分类硬编码
- ✅ 不发送串口
- ✅ 不控制 MCU
- ⚠️ 未实际运行验证（需用户在有摄像头的机器上测试）

---

## 6. 下一步

**Step 4**：模型导出与量化（`04_Model_Export_Quantization/model_export_tflite.py`），将五分类 PyTorch 模型导出为 TFLite，验证推理一致性。
