# Step 3 报告 — 五分类模型训练脚本改造 + Baseline 训练

> 日期：2025-06-15  
> 状态：**脚本改造完成，真实训练已完成（Baseline v1）**

---

## 1. 修改文件

| 文件 | 操作 | 说明 |
|------|------|------|
| `03_Model_Training/model_train_mobilenetv3.py` | **已覆盖** | 完整五分类版本 |
| `11_Vision_Trigger_5Class_System/step3_model_training_report.md` | **新增** | 本报告 |

### 1.1 关键代码变更

| 变更项 | 旧代码 | 新代码 |
|--------|--------|--------|
| 类别数 | `NUM_CLASSES = 4` | `load_class_order()` 自动读取 → `num_classes = 5` |
| 数据集 | `datasets.ImageFolder` 隐式排序 | `SafeImageFolder` 强制按权威顺序映射 |
| 预训练 | `Weights.DEFAULT`（可能联网） | 默认 `weights=None`，仅 `--pretrained` 时尝试加载 |
| 命令行 | 无 argparse | 完整 argparse（15+ 参数） |
| 评估指标 | 标准 classification_report | 增加 pending_recall / garbage_macro_precision / false_trigger_count |

---

## 2. 数据集状态

### 2.1 `dataset/` 五类目录

| 目录 | 存在 | 约数 |
|------|------|------|
| `待分拣/` | ✅ | ~280 |
| `其他/` | ✅ | ~280 |
| `厨余/` | ✅ | ~280 |
| `可回收/` | ✅ | ~280 |
| `有害/` | ✅ | ~280 |

### 2.2 训练数据划分

| split | 数量 |
|-------|------|
| train | 980 |
| val | 209 |
| test | 212 |
| **总计** | **1401** |

类别顺序：`['待分拣', '其他', '厨余', '可回收', '有害']` → `{待分拣: 0, 其他: 1, 厨余: 2, 可回收: 3, 有害: 4}`

---

## 3. 训练脚本新增能力

### 3.1 `load_class_order()`

优先级：`garbage_dataset/class_names.json` → `class_mapping.json` → `class_mapping_5class.json`

### 3.2 `SafeImageFolder`

- 继承 `torch.utils.data.Dataset`
- `class_to_idx` 由外部 `class_names` 列表决定，不依赖文件夹名排序
- 按 `class_names` 顺序遍历目录，确保 index 0=待分拣 ... 4=有害

### 3.3 argparse 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data-dir` | `garbage_dataset` | 数据集根目录 |
| `--output-dir` | `models/vision_trigger_5class_mobilenetv3` | 输出根目录 |
| `--class-config` | `class_mapping_5class.json` | 五分类配置文件 |
| `--model-name` | `mobilenet_v3_small` | 模型架构 |
| `--epochs` | 30 | 训练轮数 |
| `--batch-size` | 16 | 批次大小 |
| `--lr` | 0.001 | 初始学习率 |
| `--device` | auto | auto/cuda/cpu |
| `--pretrained` | 否 | 使用本地缓存预训练权重 |
| `--patience` | 8 | 早停耐心 |
| `--dry-run` | — | 只检查不训练 |
| `--eval-only` | — | 加载模型评估 |
| `--resume` | — | 从检查点恢复 |

### 3.4 自定义评估指标

| 指标 | 说明 |
|------|------|
| `pending_recall` | 待分拣 recall |
| `garbage_macro_precision` | 四类垃圾 macro precision |
| `pending_false_trigger_count` | 待分拣→垃圾误判次数 |
| `garbage_to_pending_count` | 垃圾→待分拣漏判次数 |

---

# ═══════════════════════════════════════════
# Step 3.5：真实训练结果（Baseline v1）
# ═══════════════════════════════════════════

## A. 训练基本信息

| 项目 | 值 |
|------|-----|
| 训练目录 | `models/vision_trigger_5class_mobilenetv3/mobilenetv3_garbage_20260615_201122` |
| 模型路径 | `models/vision_trigger_5class_mobilenetv3/latest_mobilenetv3_best.pt` |
| 模型 | MobileNetV3 Small |
| 设备 | cuda |
| 类别数 | 5 |
| train / val / test | 980 / 209 / 212 |
| 最佳 epoch | 18 / 30 |
| 最佳 val_acc | 0.8660 |

---

## B. 混淆矩阵 (test)

```
真实 \ 预测    待分拣  其他  厨余  可回收  有害   |  总计  recall
待分拣           42     0     1      0     0   |   43   0.977 ✅
其他              2    26     0      5     9   |   42   0.619 🔴
厨余              2     0    39      0     1   |   42   0.929 ✅
可回收            0     0     0     38     5   |   43   0.884 ⚠️
有害              0     5     2      1    34   |   42   0.810 ⚠️
─────────────────────────────────────────────────
预测总计         46    31    42     44    49
precision    0.913 0.839 0.929  0.864  0.694
              ✅    ⚠️    ✅     ⚠️     🔴
```

---

## C. 验收判定

| 指标 | 目标 | 实际值 | 状态 |
|------|------|--------|------|
| test accuracy | ≥ 90% | **84.43%** | ❌ 未达标 |
| 待分拣 recall | ≥ 95% | **97.67%** | ✅ 达标 |
| 四类垃圾 macro precision | ≥ 90% | **83.12%** | ❌ 未达标 |

**结论：当前模型只能作为 Baseline v1 / 工程验证模型，不能作为最终闭环部署模型。**

---

## D. 逐类深度分析

### 待分拣 ✅ (precision 0.913, recall 0.977)
- 仅 1 张被误判为厨余
- `pending_false_trigger_count = 1`
- 待分拣 recall 已达标，但 precision 仍可提升（46 个预测中 4 个实际是垃圾）

### 其他 🔴 (precision 0.839, recall 0.619)
- **严重问题**：42 张仅 26 张正确
- 9 张 → 有害，5 张 → 可回收，2 张 → 待分拣
- 「其他」类特征边界模糊，与有害和可回收高度重叠

### 厨余 ✅ (precision 0.929, recall 0.929)
- 表现最优，仅 3 张误判

### 可回收 ⚠️ (precision 0.864, recall 0.884)
- 5 张可回收 → 有害

### 有害 🔴 (precision 0.694, recall 0.810)
- 预测为有害的 49 张中仅 34 张真实为有害
- 误判来源：9 张来自「其他」+ 5 张来自「可回收」+ 1 张来自「厨余」
- 「其他」和「可回收」大量样本被误判为有害

---

## E. 关键自定义指标

| 指标 | 值 | 说明 |
|------|-----|------|
| `pending_recall` | 0.9767 | 待分拣 43 张中 42 张正确 |
| `garbage_macro_precision` | 0.8312 | |
| `pending_false_trigger_count` | **1** | 待分拣→厨余 1 次 |
| `garbage_to_pending_count` | **4** | 其他 2 次 + 厨余 2 次→待分拣 |

---

## F. 核心问题总结

| 问题 | 严重度 | 描述 |
|------|--------|------|
| 「其他」recall 极低 | 🔴 | 仅 0.619，38% 的其他样本分错 |
| 「有害」precision 极低 | 🔴 | 仅 0.694，预测有害中 30% 是假的 |
| 「其他 ↔ 有害」双向混淆 | 🔴 | 9 张其他→有害 + 5 张有害→其他 |
| 「可回收 → 有害」单向混淆 | 🟡 | 5 张可回收→有害 |
| 数据量不足 | 🟡 | 每类仅 ~200 train 张 |

---

## G. 下一轮优化计划

### G.1 数据补充（最高优先级）

| 类别 | 当前 train 约 | 建议补充 | 目标 |
|------|-------------|---------|------|
| 其他 | 196 | **150–250 张** | 350–450 |
| 有害 | 196 | **150–250 张** | 350–450 |
| 待分拣 | 196 | **100–200 张** | 300–400 |
| 可回收 | 196 | 50–100 张 | 250–300 |
| 厨余 | 196 | 50–100 张 | 250–300 |

### G.2 采集策略

1. **「其他」类**：与有害外观相似但实际是其他垃圾的物品（废纸、污染塑料、一次性餐具）
2. **「有害」类**：多样化有害垃圾（电池、药品、灯泡、油漆罐）
3. **边界样本**：其他 vs 有害 vs 可回收的混淆样本
4. **「待分拣」类**：空平台不同光照、阴影、分拣后残留

### G.3 采集规范

- 使用同一 ROI，与部署环境一致
- 避免背景泄漏
- train/val/test 来自不同拍摄时段和光照
- 不要混入文字界面、黑色面板、ROI 红框

### G.4 下一轮训练命令 (Windows cmd)

```cmd
cd /d "D:\Garbage Classification\Garbage_Classification_Organized"

"D:\SoftWare\miniconda3\envs\yunet\python.exe" "02_Dataset_Splitting\dataset_split_train_val_test.py" --source-dir dataset --output-dir garbage_dataset --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15 --seed 42 --clean-output

"D:\SoftWare\miniconda3\envs\yunet\python.exe" "03_Model_Training\model_train_mobilenetv3.py" --epochs 50 --batch-size 16 --patience 10 --num-workers 0 --no-pretrained
```

> **注意**：50 epochs 不是强制跑满。early stopping 设定 patience=10，若验证集长期无提升则提前停止。

---

## H. 状态声明

- ✅ 训练流程已跑通
- ✅ 五分类脚本验证通过
- ❌ 不满足最终闭环验收
- ❌ 暂不建议进入正式硬件自动分拣
- 📌 当前模型 = **Baseline v1 / 工程验证模型**

---

## I. 下一步

1. **优先**：按 G 节建议补充数据
2. **次优先**：重新训练 Baseline v2
3. **暂缓**：Step 4 TFLite 导出（应在模型指标达标后执行）
4. **暂缓**：Step 5+ RPi 集成（同）
