# Step 4 报告：五分类模型 TFLite 导出与量化验证

**生成时间**: 2025-06-18  
**状态**: ⚠️ **部分完成** — Dry-run 通过，实际导出因缺少依赖被阻塞

---

## 1. 当前使用的 PyTorch 模型

| 项目 | 值 |
|------|-----|
| 模型路径 | `models/vision_trigger_5class_mobilenetv3/latest_mobilenetv3_best.pt` |
| 模型类型 | MobileNet V3 Small |
| 输入尺寸 | 224 × 224 |
| 输出层 | `classifier[3]` → 5 类 |
| state_dict | 244 keys |

## 2. 当前训练指标摘要

| 指标 | 值 |
|------|-----|
| best epoch | 49 |
| best val_acc | 0.952381 |
| final test_acc | 0.922118 |
| pending_recall | 0.9873 |
| garbage_macro_precision | 0.8974 |

### Per-Class Test Report

| 类别 | Precision | Recall | F1 | Support |
|------|-----------|--------|-----|---------|
| 待分拣 | 0.99 | 0.99 | 0.99 | 79 |
| 其他 | 0.95 | 0.80 | 0.87 | 71 |
| 厨余 | 0.95 | 1.00 | 0.98 | 42 |
| 可回收 | 0.75 | 0.95 | 0.84 | 43 |
| 有害 | 0.94 | 0.91 | 0.92 | 86 |

### Final Test Confusion Matrix

```
真实\预测     待分拣    其他    厨余   可回收    有害  |  总计   recall
待分拣          78      1      0      0      0  |   79    0.987
其他            1      57      0      9      4  |   71    0.803
厨余            0       0     42      0      0  |   42    1.000
可回收          0       0      1     41      1  |   43    0.953
有害            0       2      1      5     78  |   86    0.907
```

## 3. 导出脚本修改点

修改文件: `04_Model_Export_Quantization/model_export_tflite.py`

### 新增功能

| 功能 | 描述 |
|------|------|
| **argparse CLI** | 支持 `--model-path`, `--output-dir`, `--class-config`, `--data-dir`, `--model-name`, `--num-classes`, `--image-size`, `--batch-size`, `--quantize-float16`, `--verify`, `--verify-samples`, `--device`, `--dry-run` |
| **`--class-config`** | 从外部 JSON 文件加载类别映射（如 `class_mapping_5class.json`），优先于 checkpoint 内置映射 |
| **`--dry-run`模式** | 检查 checkpoint 存在、可加载、num_classes=5、输出层=5、class mapping 正确、output-dir 可创建、导出依赖齐全。不实际导出文件 |
| **依赖懒加载** | `onnx`, `onnxruntime`, `onnxsim`, `tensorflow` 改为 try/except 导入，dry-run 模式无需这些依赖 |
| **输出命名** | ONNX: `model_float32.onnx`; TFLite: `garbage_mobilenetv3_5class_float32.tflite` / `garbage_mobilenetv3_5class_float16.tflite`; 目录: `export_YYYYMMDD_HHMMSS` |
| **配置产物** | 自动生成 `class_names.json`, `class_mapping.json`, `export_config.json` |
| **类别顺序验证** | 自动检查类别顺序是否为 `["待分拣", "其他", "厨余", "可回收", "有害"]` |

### 未修改部分

- `build_model()` — 使用 `torchvision.models.mobilenet_v3_small`，与训练脚本一致
- ONNX → SavedModel → TFLite 导出核心流程保持不变
- 预处理管道（resize + center crop + normalize）保持不变

## 4. 类别顺序确认

```
0 = 待分拣
1 = 其他
2 = 厨余
3 = 可回收
4 = 有害
```

来源: `11_Vision_Trigger_5Class_System/config/class_mapping_5class.json`

Dry-run 验证通过 ✅

## 5. 导出命令

### Dry-run（已验证通过）
```cmd
cd /d "D:\Garbage Classification\Garbage_Classification_Organized"
"D:\SoftWare\miniconda3\envs\yunet\python.exe" "04_Model_Export_Quantization\model_export_tflite.py" --model-path "models\vision_trigger_5class_mobilenetv3\latest_mobilenetv3_best.pt" --output-dir "models\vision_trigger_5class_tflite" --class-config "11_Vision_Trigger_5Class_System\config\class_mapping_5class.json" --data-dir "garbage_dataset" --num-classes auto --image-size 224 --dry-run
```

### 实际导出（依赖修复后执行）
```cmd
cd /d "D:\Garbage Classification\Garbage_Classification_Organized"
"D:\SoftWare\miniconda3\envs\yunet\python.exe" "04_Model_Export_Quantization\model_export_tflite.py" --model-path "models\vision_trigger_5class_mobilenetv3\latest_mobilenetv3_best.pt" --output-dir "models\vision_trigger_5class_tflite" --class-config "11_Vision_Trigger_5Class_System\config\class_mapping_5class.json" --data-dir "garbage_dataset" --num-classes auto --image-size 224 --quantize-float16 --verify --verify-samples 100
```

## 6. 导出产物列表（预期）

| 文件 | 路径（预期） |
|------|-------------|
| ONNX FP32 | `models/vision_trigger_5class_tflite/export_YYYYMMDD_HHMMSS/model_float32.onnx` |
| SavedModel | `models/vision_trigger_5class_tflite/export_YYYYMMDD_HHMMSS/saved_model/` |
| TFLite Float32 | `models/vision_trigger_5class_tflite/export_YYYYMMDD_HHMMSS/garbage_mobilenetv3_5class_float32.tflite` |
| TFLite Float16 | `models/vision_trigger_5class_tflite/export_YYYYMMDD_HHMMSS/garbage_mobilenetv3_5class_float16.tflite` |
| class_names.json | `[...]/class_names.json` |
| class_mapping.json | `[...]/class_mapping.json` |
| export_config.json | `[...]/export_config.json` |
| export_summary.json | `[...]/export_summary.json` |

**当前状态**: 因环境依赖缺失，以上文件尚未生成。

## 7. TFLite 输出 shape

期望输出: `(1, 5)` — 5 类 softmax 概率

## 8. 验证样本数

`--verify-samples 100`（从 `garbage_dataset/test/` 随机抽取）

## 9. PyTorch vs TFLite top1 一致率

**未执行** — 阻塞于依赖缺失。

## 10. Mismatch 明细

**未执行** — 阻塞于依赖缺失。

## 11. 当前是否通过 Step 4

⚠️ **未完全通过** — Dry-run (PyTorch 模型验证) 全部通过，但实际导出和一致性验证被环境依赖阻塞。

### Dry-run 结果: 8/9 通过

| 检查项 | 结果 |
|--------|------|
| PyTorch checkpoint 存在 | ✅ |
| checkpoint 可加载 | ✅ |
| state_dict 可提取 (244 keys) | ✅ |
| num_classes = 5 | ✅ |
| 输出层 = 5 (classifier[3]) | ✅ |
| 类别顺序正确 | ✅ |
| output-dir 可创建 | ✅ |
| image_size = 224 | ✅ |
| 导出依赖齐全 | ❌ |

## 12. 阻塞问题：缺失依赖

### 环境依赖状态

| 依赖 | 状态 | 说明 |
|------|------|------|
| torch 2.11.0 | ✅ | 正常 |
| numpy 2.2.6 | ✅ | 正常（但太新，与 TF 冲突） |
| onnx | ❌ | 未安装 |
| onnxruntime | ❌ | 未安装 |
| onnxsim | ❌ | 未安装 |
| tensorflow | ❌ | 已安装但不可用 — 与 numpy 2.2.6 不兼容 |

### 修复命令

```cmd
cd /d "D:\Garbage Classification\Garbage_Classification_Organized"
"D:\SoftWare\miniconda3\envs\yunet\python.exe" -m pip install "numpy<2"
"D:\SoftWare\miniconda3\envs\yunet\python.exe" -m pip install onnx onnxruntime onnxsim
"D:\SoftWare\miniconda3\envs\yunet\python.exe" -m pip install tensorflow
```

> ⚠️ 注意：必须先降级 numpy 到 1.x，再安装 tensorflow。否则 TF 无法在 numpy 2.x 下运行。

或者创建新 conda 环境（更安全）:
```cmd
conda create -n yunet-tf python=3.10
conda activate yunet-tf
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install numpy==1.26.4 onnx onnxruntime onnxsim tensorflow scikit-learn matplotlib pillow
```

## 13. 下一步建议

1. **安装缺失依赖**（见上节修复命令）
2. **重新运行实际导出 + verify**：
   ```cmd
   cd /d "D:\Garbage Classification\Garbage_Classification_Organized"
   "D:\SoftWare\miniconda3\envs\yunet\python.exe" "04_Model_Export_Quantization\model_export_tflite.py" --model-path "models\vision_trigger_5class_mobilenetv3\latest_mobilenetv3_best.pt" --output-dir "models\vision_trigger_5class_tflite" --class-config "11_Vision_Trigger_5Class_System\config\class_mapping_5class.json" --data-dir "garbage_dataset" --num-classes auto --image-size 224 --quantize-float16 --verify --verify-samples 100
   ```
3. **如果 TFLite 一致性验证通过（top1 ≥ 98%）**，进入 Step 5：五分类 RPi 实时预览脚本适配，加载 TFLite 模型，使用 ROI crop，显示稳定预测和理论动作，但不发送串口。

4. **关于硬件闭环**：当前不建议直接进入最终硬件自动闭环。主要风险是可回收 precision = 0.745，仍需观察和后续优化。
