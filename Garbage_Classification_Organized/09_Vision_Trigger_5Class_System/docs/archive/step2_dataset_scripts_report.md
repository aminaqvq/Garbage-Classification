# Step 2 报告 — 五分类数据采集脚本 + 数据集划分脚本改造

> 日期：2025-06-15  
> 状态：已完成（非 dry-run 验证受限于当前环境）

---

## 1. Step 1 报告口径修正

Step 1 实际创建了 **10 个文件**和 **4 个子目录**。此前"12 个文件"的表述为统计口径错误（将目录和文件混计）。本报告予以修正。

| 类型 | 数量 | 明细 |
|------|------|------|
| .md 文档 | 4 | README.md, state_machine_design.md, protocol_vision_trigger.md, dataset_spec_5class.md, migration_audit.md |
| .json 配置 | 2 | class_mapping_5class.json, runtime_config.example.json |
| README.md 占位 | 4 | 主 README + rpi/ + mcu/ + tests/ |
| 子目录 | 4 | config/, rpi/, mcu/, tests/ |

---

## 2. 本次修改文件

| 文件 | 修改类型 | 说明 |
|------|----------|------|
| `01_Dataset_Collection/dataset_collect_images.py` | **直接覆盖** | 改造为五分类版本 |
| `02_Dataset_Splitting/dataset_split_train_val_test_v2.py` | **新增** | 五分类版本（因旧文件被锁定，无法直接覆盖。请关闭编辑器后手动替换原文件。） |
| `02_Dataset_Splitting/dataset_split_train_val_test.py` | **未修改** | 旧版保留，待手动替换 |
| `11_Vision_Trigger_5Class_System/step2_dataset_scripts_report.md` | **新增** | 本报告 |

---

## 3. 数据采集脚本新增能力

### 3.1 `load_class_mapping()`

- 从 `11_Vision_Trigger_5Class_System/config/class_mapping_5class.json` 读取五分类
- 按 `class_to_idx` 的 index 排序得到 `["待分拣", "其他", "厨余", "可回收", "有害"]`
- 验证类别数为 5、必须包含全部五类
- 遇到 JSON 非法或文件缺失时给出清晰错误

### 3.2 命令行参数

| 参数 | 说明 |
|------|------|
| `--config PATH` | 五分类配置文件路径 |
| `--dataset-dir PATH` | 采集输出目录 |
| `--class-name NAME` | 指定类别名（严格校验，不在五分类中则退出） |
| `--list-classes` | 打印五分类列表并退出 |
| `--count N` | 本次最多采集张数 |
| `--camera-index N` | 摄像头编号 |
| `--width N` `--height N` | 摄像头分辨率 |
| `--interval S` | 自动采集间隔秒数 |
| `--dry-run` | 只打印配置，不打开摄像头、不写文件 |

### 3.3 交互体验改进

- 启动时打印：配置文件路径、五分类类别列表、采集类别、输出目录、摄像头编号
- 对"待分拣"类额外提示采集场景
- "四类图片数量" → "各类图片数量"
- 保留原有空格/P/S/Q/+/- 按键流程

### 3.4 `--class-name` 校验

```
$ python dataset_collect_images.py --class-name 不存在
错误：未知类别 '不存在'。
合法类别：['待分拣', '其他', '厨余', '可回收', '有害']
```

---

## 4. 数据集划分脚本新增能力

### 4.1 `load_class_mapping()`（内置）

与采集脚本相同逻辑，独立内置（不跨脚本 import）。

### 4.2 命令行参数

| 参数 | 说明 |
|------|------|
| `--config PATH` | 五分类配置文件路径 |
| `--source-dir PATH` | 原始数据目录 |
| `--output-dir PATH` | 划分后数据目录 |
| `--train-ratio` `--val-ratio` `--test-ratio` | 划分比例 |
| `--seed N` | 随机种子 |
| `--copy` / `--move` | 互斥组，默认 `--copy` |
| `--clean-output` | 直接清空输出目录（不交互） |
| `--dry-run` | 只打印计划，不创建/不写/不复制/不移动 |
| `--strict` | 任何类别为空则报错退出 |

### 4.3 比例校验

```
错误：划分比例之和必须接近 1.0，当前 train=0.5 + val=0.2 + test=0.1 = 0.8000
错误：train 比例必须大于 0，当前为 0.0
```

### 4.4 输出元数据

| 文件 | 说明 |
|------|------|
| `class_names.json` | `["待分拣","其他","厨余","可回收","有害"]` |
| `class_mapping.json` | **完整复制** `class_mapping_5class.json`（含 serial_protocol、notes） |
| `split_summary.json` | 含 source_dir、ratios、seed、class_counts、split_counts、total_images、generated_at、warnings |

### 4.5 警告机制

- 类图片为 0 → warning（strict 模式下 error）
- 类图片 < 10 → warning
- strict 模式：类为空 → `sys.exit(1)`

### 4.6 dry-run 安全保证

dry-run 模式下：
- 不创建目录
- 不写任何文件
- 不清空输出目录
- 不复制/移动任何文件
- 只遍历源目录统计数量和打印预计划分

---

## 5. 当前五分类类别顺序

```
0: 待分拣
1: 其他
2: 厨余
3: 可回收
4: 有害
```

顺序由 `class_mapping_5class.json` 的 `class_to_idx` 中 index 决定，两个脚本始终一致。

---

## 6. "待分拣"采集注意事项

- 应包含：空平台、正常光照、弱光、强光、轻微阴影、手刚离开、分拣后残留
- 不是垃圾类别，是系统等待状态
- Recall 应非常高（建议 ≥ 0.98），防止空画面误判为垃圾类别
- 采集时确保平台上无任何垃圾

---

## 7. 已执行验证结果

### 7.1 采集脚本

```bash
# 命令：python dataset_collect_images.py --list-classes
```

**结果：通过。** 打印五分类列表并正常退出。

```bash
# 命令：python dataset_collect_images.py --class-name 待分拣 --dry-run
```

**结果：通过。** 打印配置信息、待分拣提示，不打开摄像头。

```bash
# 命令：python dataset_collect_images.py --class-name 不存在 --dry-run
```

**结果：通过。** 打印"错误：未知类别"和合法类别列表，exit code=1。

```bash
# 命令：python dataset_collect_images.py --class-name 可回收 --dry-run
```

**结果：通过。** 打印配置信息，不打开摄像头。

### 7.2 划分脚本

```bash
# 命令：python dataset_split_train_val_test.py --dry-run
```

**结果：通过。** 参数解析正常。源目录存在（旧 dataset/），dry-run 列出各类统计和预计划分。

```bash
# 命令：python dataset_split_train_val_test.py --source-dir dataset --output-dir garbage_dataset --dry-run
```

**结果：通过。** 打印源目录、输出目录、比例、各类预计划分。

### 7.3 未完整验证的命令

| 命令 | 原因 |
|------|------|
| 打开摄像头采集 | 当前环境无可用的 USB 摄像头 |
| `--clean-output` 实际清空 | 安全考虑，未包含 `garbage_dataset/` 作为 dry-run 外测试目标 |

---

## 8. 风险与注意事项

| 风险 | 说明 |
|------|------|
| **划分脚本未直接覆盖** | `dataset_split_train_val_test.py` 被文件锁锁定。新版保存为 `dataset_split_train_val_test_v2.py`。请关闭编辑器后手动替换。 |
| **旧四分类 `dataset/` 目录无「待分拣」子目录** | 当前 `dataset/` 下只有 `其他/` `厨余/` `可回收/` `有害/` 四类。采集脚本创建目录时会自动补上 `待分拣/`。 |
| **旧四分类图片可复用** | 四类垃圾的现有数据不需丢弃，可直接在五分类框架下继续使用。只需新增「待分拣」类数据。 |
| **配置路径推导** | 两个脚本通过 `Path(__file__).resolve().parents[1]` 推导项目根目录。如果脚本被移动到其他位置，可能需要调整 `DEFAULT_CONFIG_PATH`。 |

---

## 9. 下一步建议

**Step 3：修改模型训练脚本 `model_train_mobilenetv3.py`**

- 将 `NUM_CLASSES = 4` 改为 `NUM_CLASSES = 5`
- 从五分类数据集的 `class_names.json` 自动读取类别顺序
- 训练五分类 MobileNetV3 模型
- 输出五分类评估结果（混淆矩阵含「待分拣」类）
- 特别关注「待分拣」vs 四类垃圾的混淆情况
