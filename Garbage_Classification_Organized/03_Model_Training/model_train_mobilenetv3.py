import csv
import json
import time
import random
import shutil
import logging
import argparse
import sys
from pathlib import Path
from datetime import datetime
from collections import Counter

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models

from sklearn.metrics import classification_report, confusion_matrix

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================================================
# 五分类配置读取
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "garbage_dataset"
DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT / "09_Vision_Trigger_5Class_System" / "config" / "class_mapping_5class.json"
)


def load_class_order(data_dir=None, config_path=None):
    """
    从垃圾数据集目录或权威配置文件读取五分类类别顺序。
    优先级：class_names.json → class_mapping.json → class_mapping_5class.json
    返回 (class_names, num_classes)
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    data_dir = Path(data_dir)

    # 1) 尝试 class_names.json
    names_path = data_dir / "class_names.json"
    if names_path.exists():
        try:
            with open(names_path, "r", encoding="utf-8") as f:
                class_names = json.load(f)
            if isinstance(class_names, list) and len(class_names) == 5:
                return class_names, len(class_names)
        except Exception:
            pass

    # 2) 尝试 class_mapping.json
    mapping_path = data_dir / "class_mapping.json"
    if mapping_path.exists():
        try:
            with open(mapping_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            c2i = data.get("class_to_idx", {})
            if c2i and len(c2i) == 5:
                sorted_pairs = sorted(c2i.items(), key=lambda kv: kv[1])
                class_names = [name for name, _ in sorted_pairs]
                return class_names, len(class_names)
        except Exception:
            pass

    # 3) 尝试权威配置
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    config_path = Path(config_path)
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        c2i = data.get("class_to_idx", {})
        if c2i:
            sorted_pairs = sorted(c2i.items(), key=lambda kv: kv[1])
            class_names = [name for name, _ in sorted_pairs]
            if len(class_names) == 5:
                return class_names, len(class_names)

    raise RuntimeError(
        "无法确定五分类类别顺序。请确保以下文件之一存在：\n"
        f"  1) {data_dir / 'class_names.json'}\n"
        f"  2) {data_dir / 'class_mapping.json'}\n"
        f"  3) {config_path}"
    )


# =========================================================
# SafeImageFolder — 强制按权威顺序映射 index
# =========================================================

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class SafeImageFolder(Dataset):
    """
    类似 ImageFolder，但 class_to_idx 由外部 class_names 决定，
    不依赖文件夹名排序。
    """

    def __init__(self, root, class_names, transform=None):
        self.root = Path(root)
        self.class_names = list(class_names)
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.transform = transform
        self.samples = []
        self.targets = []

        for class_name in self.class_names:
            class_dir = self.root / class_name
            if not class_dir.exists():
                print(f"警告：类别目录不存在：{class_dir}")
                continue
            for file_path in sorted(class_dir.iterdir()):
                if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append(str(file_path))
                    self.targets.append(self.class_to_idx[class_name])

        if not self.samples:
            raise RuntimeError(f"在 {self.root} 中未找到任何图片文件。")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        label = self.targets[idx]
        try:
            from PIL import Image
            image = Image.open(path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"无法读取图片：{path}") from e
        if self.transform:
            image = self.transform(image)
        return image, label


# =========================================================
# 配置区：以下大部分可通过 argparse 覆盖
# =========================================================

DATA_DIR = DEFAULT_DATA_DIR
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
MODEL_NAME = "mobilenet_v3_small"
NUM_CLASSES = None  # 将由 load_class_order() 动态设置

BATCH_SIZE = 100
MAX_STEPS = 0
EPOCHS = 100
HEAD_LR = 1e-3
USE_FINE_TUNE = True
UNFREEZE_AFTER_EPOCH = 5
BACKBONE_LR = 1e-4
HEAD_FINE_TUNE_LR = 5e-4
USE_EARLY_STOP = True
PATIENCE = 10
MIN_DELTA = 1e-4
USE_SCHEDULER = True
SCHEDULER_PATIENCE = 3
SCHEDULER_FACTOR = 0.5
USE_CLASS_WEIGHTS = True
NUM_WORKERS = 0
LOG_INTERVAL = 0  # 默认不打印 step 中间日志；每个 epoch 只显示一次汇总
SEED = 42
USE_AMP = True
USE_PRETRAINED = False  # 默认不使用预训练权重
SAVE_PER_IMAGE_PREDICTIONS = False  # 默认关闭，使用 --save-per-image 开启
IMAGE_SIZE = 224


# =========================================================
# 工具函数
# =========================================================

def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def make_run_dir(output_base=None):
    if output_base is None:
        output_base = OUTPUT_ROOT
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_base) / f"mobilenetv3_garbage_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def setup_logger(run_dir: Path):
    log_path = run_dir / "train_console.log"
    logger = logging.getLogger("garbage_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(preferred=None):
    if preferred and preferred != "auto":
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def check_dataset_dir(data_dir: Path, class_names):
    required = ["train", "val", "test"]
    if not data_dir.exists():
        raise FileNotFoundError(f"数据集目录不存在：{data_dir.resolve()}")
    for split in required:
        split_dir = data_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(f"缺少 {split} 目录：{split_dir.resolve()}")
        for class_name in class_names:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                raise FileNotFoundError(f"缺少分类目录：{class_dir.resolve()}")


def get_transforms(image_size=224):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=8),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.03),
        transforms.ToTensor(),
        normalize,
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(image_size + 32),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])
    return train_tf, eval_tf


def build_datasets(data_dir: Path, class_names):
    train_tf, eval_tf = get_transforms(IMAGE_SIZE)
    train_ds = SafeImageFolder(data_dir / "train", class_names, transform=train_tf)
    val_ds = SafeImageFolder(data_dir / "val", class_names, transform=eval_tf)
    test_ds = SafeImageFolder(data_dir / "test", class_names, transform=eval_tf)
    return train_ds, val_ds, test_ds


def build_loader(dataset, batch_size, shuffle, device, num_workers):
    pin_memory = device.type == "cuda"
    kwargs = {
        "batch_size": batch_size, "shuffle": shuffle,
        "num_workers": num_workers, "pin_memory": pin_memory, "drop_last": False,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
    return DataLoader(dataset, **kwargs)


def dataset_class_counts(dataset, class_names):
    counter = Counter(dataset.targets)
    result = {}
    for idx, name in enumerate(class_names):
        result[name] = counter.get(idx, 0)
    return result


def save_dataset_summary(run_dir: Path, train_ds, val_ds, test_ds, class_names):
    path = run_dir / "dataset_summary.csv"
    rows = []
    split_map = {"train": train_ds, "val": val_ds, "test": test_ds}
    for split_name, ds in split_map.items():
        counts = dataset_class_counts(ds, class_names)
        for class_name, count in counts.items():
            rows.append({"split": split_name, "class_name": class_name, "count": count})
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "class_name", "count"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def save_class_mapping(run_dir: Path, class_to_idx: dict, full_config=None):
    path = run_dir / "class_mapping.json"
    if full_config:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(full_config, f, ensure_ascii=False, indent=2)
    else:
        idx_to_class = {str(v): k for k, v in class_to_idx.items()}
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"class_to_idx": class_to_idx, "idx_to_class": idx_to_class}, f, ensure_ascii=False, indent=4)
    return path


def save_config(run_dir: Path, device, train_ds, val_ds, test_ds, num_classes, class_names, args):
    config = {
        "created_time": now_str(),
        "model_name": MODEL_NAME,
        "num_classes": num_classes,
        "class_names": class_names,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "max_steps": MAX_STEPS,
        "image_size": IMAGE_SIZE,
        "head_lr": HEAD_LR,
        "use_fine_tune": USE_FINE_TUNE,
        "unfreeze_after_epoch": UNFREEZE_AFTER_EPOCH,
        "backbone_lr": BACKBONE_LR,
        "head_fine_tune_lr": HEAD_FINE_TUNE_LR,
        "use_early_stop": USE_EARLY_STOP,
        "patience": PATIENCE,
        "min_delta": MIN_DELTA,
        "use_scheduler": USE_SCHEDULER,
        "scheduler_patience": SCHEDULER_PATIENCE,
        "scheduler_factor": SCHEDULER_FACTOR,
        "use_class_weights": USE_CLASS_WEIGHTS,
        "use_pretrained": USE_PRETRAINED,
        "num_workers": NUM_WORKERS,
        "seed": SEED,
        "use_amp": USE_AMP,
        "device": str(device),
        "data_dir": str(DATA_DIR.resolve()),
        "output_dir": str(run_dir.resolve()),
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds),
        "class_to_idx": {name: idx for idx, name in enumerate(class_names)},
    }
    path = run_dir / "config.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
    return path


def build_model(num_classes: int, use_pretrained=False):
    if MODEL_NAME == "mobilenet_v3_small":
        weights = None
        if use_pretrained:
            try:
                weights = models.MobileNet_V3_Small_Weights.DEFAULT
            except Exception:
                print("警告：无法加载 MobileNetV3 Small 预训练权重（本地无缓存），回退到随机初始化。")
                weights = None
        model = models.mobilenet_v3_small(weights=weights)
    elif MODEL_NAME == "mobilenet_v3_large":
        weights = None
        if use_pretrained:
            try:
                weights = models.MobileNet_V3_Large_Weights.DEFAULT
            except Exception:
                print("警告：无法加载 MobileNetV3 Large 预训练权重（本地无缓存），回退到随机初始化。")
                weights = None
        model = models.mobilenet_v3_large(weights=weights)
    else:
        raise ValueError("MODEL_NAME 只能是 mobilenet_v3_small 或 mobilenet_v3_large")

    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


def freeze_backbone(model):
    for p in model.features.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True


def unfreeze_backbone(model):
    for p in model.parameters():
        p.requires_grad = True


def build_optimizer_head_only(model):
    return torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=HEAD_LR, weight_decay=1e-4
    )


def build_optimizer_fine_tune(model):
    return torch.optim.AdamW([
        {"params": model.features.parameters(), "lr": BACKBONE_LR},
        {"params": model.classifier.parameters(), "lr": HEAD_FINE_TUNE_LR}
    ], weight_decay=1e-4)


def build_scheduler(optimizer):
    if not USE_SCHEDULER:
        return None
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE
    )


def get_current_lr(optimizer):
    return [group["lr"] for group in optimizer.param_groups]


def make_class_weights(train_ds, num_classes, device):
    counts = Counter(train_ds.targets)
    total = len(train_ds)
    weights = []
    for idx in range(num_classes):
        count = counts.get(idx, 0)
        if count <= 0:
            raise ValueError(f"类别 index={idx} 在 train 中没有图片，无法训练。")
        weight = total / (num_classes * count)
        weights.append(weight)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def append_csv(path: Path, fieldnames: list, row: dict):
    file_exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, logger, batch_log_path, scaler=None, use_amp=False):
    model.train()
    loss_sum = 0.0
    correct = 0
    total = 0
    start_time = time.time()
    for step, (x, y) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        batch_size = x.size(0)
        loss_sum += loss.item() * batch_size
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += batch_size
        if LOG_INTERVAL and step % LOG_INTERVAL == 0:
            avg_loss = loss_sum / total
            avg_acc = correct / total
            lr_list = get_current_lr(optimizer)
            logger.info(f"[Epoch {epoch:03d}] step {step:04d}/{len(loader)} | loss={avg_loss:.4f} | acc={avg_acc:.4f} | lr={lr_list}")
            append_csv(batch_log_path,
                ["time","epoch","step","total_steps","train_loss_so_far","train_acc_so_far","lr"],
                {"time":now_str(),"epoch":epoch,"step":step,"total_steps":len(loader),
                 "train_loss_so_far":avg_loss,"train_acc_so_far":avg_acc,"lr":json.dumps(lr_list)})
    epoch_loss = loss_sum / max(total, 1)
    epoch_acc = correct / max(total, 1)
    epoch_time = time.time() - start_time
    return epoch_loss, epoch_acc, epoch_time


@torch.no_grad()
def evaluate_model(model, loader, criterion, device, collect_predictions=False):
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    all_true = []
    all_pred = []
    all_conf = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        batch_size = x.size(0)
        loss_sum += loss.item() * batch_size
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        correct += (pred == y).sum().item()
        total += batch_size
        if collect_predictions:
            all_true.extend(y.cpu().numpy().tolist())
            all_pred.extend(pred.cpu().numpy().tolist())
            all_conf.extend(conf.cpu().numpy().tolist())
    result = {"loss": loss_sum / max(total, 1), "acc": correct / max(total, 1), "total": total}
    if collect_predictions:
        result["y_true"] = all_true
        result["y_pred"] = all_pred
        result["confidence"] = all_conf
    return result


def save_checkpoint(path, model, optimizer, epoch, best_val_acc, train_ds, num_classes, extra):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "best_val_acc": best_val_acc,
        "class_to_idx": train_ds.class_to_idx,
        "idx_to_class": {v: k for k, v in train_ds.class_to_idx.items()},
        "model_name": MODEL_NAME,
        "num_classes": num_classes,
        "extra": extra
    }, path)


def safe_torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def setup_matplotlib_chinese():
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun",
        "WenQuanYi Zen Hei", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def save_confusion_matrix_csv(path, cm, class_names):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["真实\\预测"] + class_names)
        for i, row in enumerate(cm):
            writer.writerow([class_names[i]] + row.tolist())


def format_confusion_matrix_text(cm, class_names):
    """
    生成适合控制台和 txt 文件查看的混淆矩阵文本。

    行 = 真实类别，列 = 预测类别。
    """
    cm = np.asarray(cm)
    row_sum = cm.sum(axis=1)
    col_sum = cm.sum(axis=0)
    total = int(cm.sum())

    recalls = np.divide(
        np.diag(cm),
        row_sum,
        out=np.zeros_like(row_sum, dtype=float),
        where=row_sum > 0,
    )
    precisions = np.divide(
        np.diag(cm),
        col_sum,
        out=np.zeros_like(col_sum, dtype=float),
        where=col_sum > 0,
    )

    name_width = max(8, max(len(name) for name in class_names) + 2)
    cell_width = 8

    lines = []
    lines.append("混淆矩阵（行=真实类别，列=预测类别）")
    lines.append("")
    header = "真实\\预测".ljust(name_width) + "".join(
        name.center(cell_width) for name in class_names
    ) + " | " + "总计".center(cell_width) + "recall".rjust(10)
    lines.append(header)
    lines.append("-" * len(header))

    for i, class_name in enumerate(class_names):
        row = class_name.ljust(name_width) + "".join(
            str(int(v)).center(cell_width) for v in cm[i]
        )
        row += " | " + str(int(row_sum[i])).center(cell_width)
        row += f"{recalls[i]:>10.3f}"
        lines.append(row)

    lines.append("-" * len(header))
    pred_total = "预测总计".ljust(name_width) + "".join(
        str(int(v)).center(cell_width) for v in col_sum
    ) + " | " + str(total).center(cell_width)
    lines.append(pred_total)

    precision_line = "precision".ljust(name_width) + "".join(
        f"{p:.3f}".center(cell_width) for p in precisions
    )
    lines.append(precision_line)

    return "\n".join(lines)


def save_confusion_matrix_txt(path, cm, class_names):
    text = format_confusion_matrix_text(cm, class_names)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return text


def save_confusion_matrix_png(path, cm, class_names, title):
    setup_matplotlib_chinese()
    fig, ax = plt.subplots(figsize=(max(8, len(class_names)*1.5), max(6, len(class_names)*1.2)))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("预测")
    ax.set_ylabel("真实")
    ax.set_title(title)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def compute_custom_metrics(y_true, y_pred, class_names):
    """计算待分拣 recall、垃圾类别 macro precision、false trigger count 等"""
    # class_names[0] 应该永远是"待分拣"
    pending_idx = 0
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    # 每类 precision/recall
    tp = np.diag(cm)
    col_sum = cm.sum(axis=0)
    row_sum = cm.sum(axis=1)
    precision_per_class = np.divide(tp, col_sum, out=np.zeros_like(tp, dtype=float), where=col_sum>0)
    recall_per_class = np.divide(tp, row_sum, out=np.zeros_like(tp, dtype=float), where=row_sum>0)
    # 待分拣 recall
    pending_recall = float(recall_per_class[pending_idx])
    # 四类垃圾 macro precision (idx 1-4)
    garbage_precisions = precision_per_class[1:5]
    garbage_macro_precision = float(np.mean(garbage_precisions)) if len(garbage_precisions) > 0 else 0.0
    # 待分拣被误判为垃圾 (false trigger)
    pending_false_trigger_count = int(cm[pending_idx, 1:].sum())
    # 垃圾被误判为待分拣
    garbage_to_pending_count = int(cm[1:, pending_idx].sum())
    return {
        "precision_per_class": {class_names[i]: round(float(precision_per_class[i]), 4) for i in range(len(class_names))},
        "recall_per_class": {class_names[i]: round(float(recall_per_class[i]), 4) for i in range(len(class_names))},
        "pending_recall": round(pending_recall, 4),
        "garbage_macro_precision": round(garbage_macro_precision, 4),
        "pending_false_trigger_count": pending_false_trigger_count,
        "garbage_to_pending_count": garbage_to_pending_count,
    }


def save_final_evaluation(run_dir, split_name, dataset, eval_result, idx_to_class, class_names):
    """保存评估结果：report txt/json、混淆矩阵 csv/png、自定义指标"""
    y_true = eval_result["y_true"]
    y_pred = eval_result["y_pred"]
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    report_text = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)

    # 标准评估文件
    report_txt_path = run_dir / f"{split_name}_classification_report.txt"
    report_json_path = run_dir / f"{split_name}_classification_report.json"
    cm_csv_path = run_dir / f"{split_name}_confusion_matrix.csv"
    cm_png_path = run_dir / f"{split_name}_confusion_matrix.png"
    cm_txt_path = run_dir / f"{split_name}_confusion_matrix.txt"
    metrics_path = run_dir / f"{split_name}_metrics.json"

    save_confusion_matrix_csv(cm_csv_path, cm, class_names)
    cm_text = save_confusion_matrix_txt(cm_txt_path, cm, class_names)
    save_confusion_matrix_png(cm_png_path, cm, class_names, f"{split_name} Confusion Matrix")

    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, ensure_ascii=False, indent=4)

    # 自定义指标
    custom = compute_custom_metrics(y_true, y_pred, class_names)
    metrics = {
        "accuracy": float(eval_result["acc"]),
        "loss": float(eval_result["loss"]),
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
        **custom,
        "classification_report_txt": str(report_txt_path.resolve()),
        "classification_report_json": str(report_json_path.resolve()),
        "confusion_matrix_csv": str(cm_csv_path.resolve()),
        "confusion_matrix_png": str(cm_png_path.resolve()),
        "confusion_matrix_txt": str(cm_txt_path.resolve()),
        "confusion_matrix_text": cm_text,
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)

    return metrics, report_text


def write_run_summary(run_dir, summary):
    path = run_dir / "run_summary.txt"
    lines = [
        "========== MobileNetV3 五分类垃圾分类训练报告 ==========",
        f"生成时间：{now_str()}", "",
        f"数据集目录：{summary['data_dir']}",
        f"输出目录：{summary['output_dir']}",
        f"设备：{summary['device']}",
        f"模型：{summary['model_name']}", "",
        "类别映射：",
    ]
    for idx, name in summary["idx_to_class"].items():
        lines.append(f"  {idx}: {name}")
    lines += [
        "", "训练结果：",
        f"最佳 epoch：{summary['best_epoch']}",
        f"最佳 val_acc：{summary['best_val_acc']:.6f}",
        f"最终 val_acc：{summary['final_val_acc']:.6f}",
        f"最终 test_acc：{summary['final_test_acc']:.6f}",
    ]
    for metric_name in ["pending_recall", "garbage_macro_precision", "pending_false_trigger_count", "garbage_to_pending_count"]:
        if metric_name in summary:
            lines.append(f"{metric_name}：{summary[metric_name]}")
    lines += ["", "关键文件："]
    for key, value in summary["files"].items():
        lines.append(f"  {key}: {value}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


# =========================================================
# 命令行参数
# =========================================================

def parse_args(class_names):
    parser = argparse.ArgumentParser(description="五分类 MobileNetV3 垃圾分类训练")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="划分后的数据集根目录")
    parser.add_argument("--output-dir", default=None, help="输出根目录 (默认 models/vision_trigger_5class_mobilenetv3)")
    parser.add_argument("--class-config", default=str(DEFAULT_CONFIG_PATH), help="五分类配置文件路径")
    parser.add_argument("--model-name", default="mobilenet_v3_small", choices=["mobilenet_v3_small", "mobilenet_v3_large"])
    parser.add_argument("--epochs", type=int, default=100, help="训练 epoch 数")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001, help="初始学习率")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (Windows 建议 0)")
    parser.add_argument("--log-interval", type=int, default=0, help="step 中间日志间隔；0 表示不打印，每个 epoch 只显示一次汇总")
    parser.add_argument("--device", default="auto", help="auto / cuda / cpu")
    parser.add_argument("--pretrained", action="store_true", help="尝试使用本地缓存的预训练权重（失败则回退随机初始化）")
    parser.add_argument("--no-pretrained", action="store_true", default=True, help="不使用预训练权重（默认）")
    parser.add_argument("--freeze-backbone", action="store_true", default=True, help="先冻结 backbone 训练分类头")
    parser.add_argument("--patience", type=int, default=10, help="早停 patience")
    parser.add_argument("--dry-run", action="store_true", help="只检查数据和模型结构，不训练")
    parser.add_argument("--eval-only", action="store_true", help="加载已有模型进行评估")
    parser.add_argument("--resume", default=None, help="从检查点恢复训练")
    args = parser.parse_args()

    # --pretrained 覆盖 --no-pretrained
    if args.pretrained:
        args.no_pretrained = False

    return args


# =========================================================
# 主程序
# =========================================================

def main():
    global DATA_DIR, OUTPUT_ROOT, MODEL_NAME, BATCH_SIZE, EPOCHS, HEAD_LR
    global UNFREEZE_AFTER_EPOCH, BACKBONE_LR, HEAD_FINE_TUNE_LR
    global USE_EARLY_STOP, PATIENCE, MIN_DELTA, USE_SCHEDULER
    global SCHEDULER_PATIENCE, SCHEDULER_FACTOR, USE_CLASS_WEIGHTS
    global NUM_WORKERS, LOG_INTERVAL, SEED, USE_AMP, USE_PRETRAINED, IMAGE_SIZE

    # 加载类别顺序
    data_dir_temp = Path(DEFAULT_DATA_DIR)
    config_temp = Path(DEFAULT_CONFIG_PATH)
    # 先尝试加载 class order（即使 data_dir 尚未划分也能从权威 config 获取）
    try:
        class_names, num_classes = load_class_order(
            str(data_dir_temp) if data_dir_temp.exists() else None,
            str(config_temp)
        )
    except Exception as e:
        print(f"错误：无法加载五分类配置：{e}")
        sys.exit(1)

    if num_classes != 5:
        print(f"错误：类别数必须为 5，当前为 {num_classes}。")
        sys.exit(1)

    print(f"类别顺序：{class_names}")
    print(f"类别数：{num_classes}")

    args = parse_args(class_names)

    # 应用命令行覆盖
    DATA_DIR = Path(args.data_dir)
    MODEL_NAME = args.model_name
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    HEAD_LR = args.lr
    BACKBONE_LR = args.lr * 0.1
    HEAD_FINE_TUNE_LR = args.lr * 0.5
    PATIENCE = args.patience
    SEED = args.seed
    NUM_WORKERS = args.num_workers
    LOG_INTERVAL = args.log_interval
    USE_PRETRAINED = args.pretrained
    IMAGE_SIZE = args.image_size

    if args.output_dir:
        output_base = Path(args.output_dir)
    else:
        output_base = PROJECT_ROOT / "models" / "vision_trigger_5class_mobilenetv3"

    set_seed(SEED)
    device = get_device(args.device)
    use_amp = USE_AMP and device.type == "cuda"

    print(f"设备：{device}")
    print(f"预训练：{USE_PRETRAINED}")

    # ========== dry-run ==========
    if args.dry_run:
        print("\n========== DRY-RUN ==========")
        print(f"数据目录：{DATA_DIR.resolve()}")
        print(f"输出目录：{output_base.resolve()}")
        print(f"模型：{MODEL_NAME}")
        print(f"num_classes：{num_classes}")
        print(f"类别映射：")
        for i, name in enumerate(class_names):
            print(f"  {i} = {name}")

        if not DATA_DIR.exists():
            print(f"\n[Dry-Run] 警告：数据目录不存在：{DATA_DIR.resolve()}")
            print(f"[Dry-Run] 请先运行数据集划分脚本生成 garbage_dataset/。")
            print(f"[Dry-Run] 验证通过（类别配置正确，数据目录待生成）。")
            sys.exit(0)

        try:
            check_dataset_dir(DATA_DIR, class_names)
            train_ds, val_ds, test_ds = build_datasets(DATA_DIR, class_names)
            print(f"\ntrain: {len(train_ds)} 张, val: {len(val_ds)} 张, test: {len(test_ds)} 张")
            print(f"class_to_idx: {train_ds.class_to_idx}")
            # 抽样验证标签范围
            sample_labels = {train_ds[i][1] for i in range(min(10, len(train_ds)))}
            print(f"抽样标签：{sorted(sample_labels)} (范围应为 0~{num_classes-1})")
            assert min(sample_labels) >= 0 and max(sample_labels) < num_classes, "标签范围异常！"
        except Exception as e:
            print(f"\n[Dry-Run] 数据检查失败：{e}")
            sys.exit(1)

        # 模型结构检查
        model = build_model(num_classes, USE_PRETRAINED)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n模型参数：total={total_params:,}, trainable={trainable_params:,}")
        out_features = model.classifier[3].out_features
        print(f"输出层：{out_features} (应为 {num_classes})")
        assert out_features == num_classes, f"输出层 {out_features} != {num_classes}！"
        print("\n[Dry-Run] 全部检查通过！")
        sys.exit(0)

    # ========== eval-only mode ==========
    if args.eval_only:
        if not args.resume:
            print("错误：--eval-only 需要 --resume 指定模型路径。")
            sys.exit(1)
        print(f"加载模型：{args.resume}")
        model = build_model(num_classes, USE_PRETRAINED)
        ckpt = safe_torch_load(Path(args.resume), device)
        model.load_state_dict(ckpt["model"])
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        check_dataset_dir(DATA_DIR, class_names)
        _, _, test_ds = build_datasets(DATA_DIR, class_names)
        test_loader = build_loader(test_ds, BATCH_SIZE, False, device, NUM_WORKERS)
        result = evaluate_model(model, test_loader, criterion, device, collect_predictions=True)
        custom = compute_custom_metrics(result["y_true"], result["y_pred"], class_names)
        print(f"Test Accuracy: {result['acc']:.4f}")
        print(f"待分拣 Recall: {custom['pending_recall']:.4f}")
        print(f"垃圾类别 Macro Precision: {custom['garbage_macro_precision']:.4f}")
        print(f"误触发次数 (待分拣→垃圾): {custom['pending_false_trigger_count']}")
        sys.exit(0)

    # ========== 确保数据目录存在 ==========
    if not DATA_DIR.exists():
        print(f"错误：数据目录不存在：{DATA_DIR.resolve()}")
        print("请先运行数据集划分脚本：")
        print(f"  python 02_Dataset_Splitting/dataset_split_train_val_test.py --source-dir dataset --output-dir garbage_dataset --clean-output")
        sys.exit(1)

    # ========== 正常训练 ==========
    check_dataset_dir(DATA_DIR, class_names)
    train_ds, val_ds, test_ds = build_datasets(DATA_DIR, class_names)

    # 尝试读取完整权威配置用于 class_mapping.json
    full_config = None
    if config_temp.exists():
        try:
            with open(config_temp, "r", encoding="utf-8") as f:
                full_config = json.load(f)
        except Exception:
            pass

    run_dir = make_run_dir(output_base)
    logger = setup_logger(run_dir)

    logger.info("========== MobileNetV3 五分类垃圾分类训练开始 ==========")
    logger.info(f"数据集目录：{DATA_DIR.resolve()}")
    logger.info(f"输出目录：{run_dir.resolve()}")
    logger.info(f"设备：{device}")
    logger.info(f"AMP：{use_amp}")
    logger.info(f"预训练：{USE_PRETRAINED}")
    logger.info(f"类别数：{num_classes}")
    logger.info(f"类别：{class_names}")

    logger.info("========== Dataset ==========")
    logger.info(f"train size: {len(train_ds)}")
    logger.info(f"val size: {len(val_ds)}")
    logger.info(f"test size: {len(test_ds)}")
    logger.info(f"class_to_idx: {train_ds.class_to_idx}")

    dataset_summary_path = save_dataset_summary(run_dir, train_ds, val_ds, test_ds, class_names)
    class_mapping_path = save_class_mapping(run_dir, train_ds.class_to_idx, full_config)
    config_path = save_config(run_dir, device, train_ds, val_ds, test_ds, num_classes, class_names, args)

    logger.info(f"数据集统计已保存：{dataset_summary_path}")
    logger.info(f"类别映射已保存：{class_mapping_path}")
    logger.info(f"配置文件已保存：{config_path}")

    train_loader = build_loader(train_ds, BATCH_SIZE, True, device, NUM_WORKERS)
    val_loader = build_loader(val_ds, BATCH_SIZE, False, device, NUM_WORKERS)
    test_loader = build_loader(test_ds, BATCH_SIZE, False, device, NUM_WORKERS)

    model = build_model(num_classes, USE_PRETRAINED)
    freeze_backbone(model)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型参数：total={total_params:,}, trainable={trainable_params:,}")

    if USE_CLASS_WEIGHTS:
        class_weights = make_class_weights(train_ds, num_classes, device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        logger.info(f"启用类别权重：{class_weights.detach().cpu().numpy().tolist()}")
    else:
        criterion = nn.CrossEntropyLoss()
        logger.info("未启用类别权重。")

    optimizer = build_optimizer_head_only(model)
    scheduler = build_scheduler(optimizer)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_model_path = run_dir / "mobilenetv3_best.pt"
    last_model_path = run_dir / "mobilenetv3_last.pt"
    train_log_path = run_dir / "training_log.csv"
    batch_log_path = run_dir / "batch_log.csv"

    train_log_fields = [
        "time","epoch","stage","lr","train_loss","train_acc",
        "val_loss","val_acc","best_val_acc","epoch_time_seconds","is_best","bad_epochs"
    ]

    best_val_acc = 0.0
    best_epoch = 0
    bad_epochs = 0
    global_step = 0
    current_stage = "head_only"
    idx_to_class = {v: k for k, v in train_ds.class_to_idx.items()}

    logger.info("========== 开始训练 ==========")

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        if (USE_FINE_TUNE and current_stage == "head_only" and epoch == UNFREEZE_AFTER_EPOCH + 1):
            logger.info("========== 进入第二阶段：解冻 backbone ==========")
            unfreeze_backbone(model)
            optimizer = build_optimizer_fine_tune(model)
            scheduler = build_scheduler(optimizer)
            current_stage = "fine_tune"
            bad_epochs = 0

        train_loss, train_acc, train_time = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, logger,
            batch_log_path, scaler, use_amp
        )
        val_result = evaluate_model(model, val_loader, criterion, device, collect_predictions=False)
        val_loss = val_result["loss"]
        val_acc = val_result["acc"]

        if scheduler is not None:
            scheduler.step(val_acc)

        lr_now = get_current_lr(optimizer)
        epoch_time = time.time() - epoch_start
        is_best = val_acc > best_val_acc + MIN_DELTA

        if is_best:
            best_val_acc = val_acc
            best_epoch = epoch
            bad_epochs = 0
            save_checkpoint(best_model_path, model, optimizer, epoch, best_val_acc, train_ds, num_classes,
                {"stage": current_stage, "val_loss": val_loss, "val_acc": val_acc,
                 "train_loss": train_loss, "train_acc": train_acc, "lr": lr_now})
        else:
            bad_epochs += 1

        save_checkpoint(last_model_path, model, optimizer, epoch, best_val_acc, train_ds, num_classes,
            {"stage": current_stage, "val_loss": val_loss, "val_acc": val_acc,
             "train_loss": train_loss, "train_acc": train_acc, "lr": lr_now})

        best_tag = " | BEST saved=mobilenetv3_best.pt" if is_best else ""
        logger.info(
            f"Epoch {epoch:03d}/{EPOCHS} | stage={current_stage} | lr={lr_now} | "
            f"train_loss={train_loss:.6f} | train_acc={train_acc:.6f} | "
            f"val_loss={val_loss:.6f} | val_acc={val_acc:.6f} | "
            f"best_val_acc={best_val_acc:.6f} | time={epoch_time:.1f}s | "
            f"bad_epochs={bad_epochs}/{PATIENCE}{best_tag}"
        )

        append_csv(train_log_path, train_log_fields, {
            "time": now_str(), "epoch": epoch, "stage": current_stage,
            "lr": json.dumps(lr_now), "train_loss": train_loss,
            "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc,
            "best_val_acc": best_val_acc, "epoch_time_seconds": epoch_time,
            "is_best": int(is_best), "bad_epochs": bad_epochs
        })

        global_step += len(train_loader)
        if MAX_STEPS > 0 and global_step >= MAX_STEPS:
            logger.info(f"[MAX_STEPS] 达到 {MAX_STEPS} 步，停止于 epoch={epoch}。")
            break
        if USE_EARLY_STOP and bad_epochs >= PATIENCE:
            logger.info(f"[EARLY STOP] 连续 {PATIENCE} 轮无提升，停止于 epoch={epoch}。")
            break

    logger.info("========== 训练结束 ==========")
    logger.info(f"最佳 epoch：{best_epoch}, 最佳 val_acc：{best_val_acc:.6f}")

    # 加载 best 模型做最终评估
    logger.info("========== 加载 best 模型 ==========")
    ckpt = safe_torch_load(best_model_path, device)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)

    final_val_result = evaluate_model(model, val_loader, criterion, device, collect_predictions=True)
    final_test_result = evaluate_model(model, test_loader, criterion, device, collect_predictions=True)

    val_metrics, val_report = save_final_evaluation(run_dir, "val", val_ds, final_val_result, idx_to_class, class_names)
    test_metrics, test_report = save_final_evaluation(run_dir, "test", test_ds, final_test_result, idx_to_class, class_names)

    logger.info("========== Final Val Report ==========\n" + val_report)
    logger.info("========== Final Val Confusion Matrix ==========\n" + val_metrics["confusion_matrix_text"])
    logger.info("========== Final Test Report ==========\n" + test_report)
    logger.info("========== Final Test Confusion Matrix ==========\n" + test_metrics["confusion_matrix_text"])
    logger.info(f"最终 val_acc：{final_val_result['acc']:.6f}")
    logger.info(f"最终 test_acc：{final_test_result['acc']:.6f}")
    logger.info(f"待分拣 recall：{test_metrics.get('pending_recall', 'N/A')}")
    logger.info(f"垃圾 macro precision：{test_metrics.get('garbage_macro_precision', 'N/A')}")

    # 复制 latest
    latest_best_path = output_base / "latest_mobilenetv3_best.pt"
    shutil.copy2(best_model_path, latest_best_path)

    summary = {
        "data_dir": str(DATA_DIR.resolve()),
        "output_dir": str(run_dir.resolve()),
        "device": str(device),
        "model_name": MODEL_NAME,
        "num_classes": num_classes,
        "idx_to_class": {str(k): v for k, v in idx_to_class.items()},
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "final_val_acc": final_val_result["acc"],
        "final_test_acc": final_test_result["acc"],
        "pending_recall": test_metrics.get("pending_recall"),
        "garbage_macro_precision": test_metrics.get("garbage_macro_precision"),
        "pending_false_trigger_count": test_metrics.get("pending_false_trigger_count"),
        "garbage_to_pending_count": test_metrics.get("garbage_to_pending_count"),
        "files": {
            "best_model": str(best_model_path.resolve()),
            "last_model": str(last_model_path.resolve()),
            "latest_best_model_copy": str(latest_best_path.resolve()),
            "training_log": str(train_log_path.resolve()),
            "batch_log": str(batch_log_path.resolve()),
            "dataset_summary": str(dataset_summary_path.resolve()),
            "class_mapping": str(class_mapping_path.resolve()),
            "config": str(config_path.resolve()),
            "val_report": str(val_metrics["classification_report_txt"]),
            "val_confusion_matrix_csv": str(val_metrics["confusion_matrix_csv"]),
            "val_confusion_matrix_png": str(val_metrics["confusion_matrix_png"]),
            "val_confusion_matrix_txt": str(val_metrics["confusion_matrix_txt"]),
            "test_report": str(test_metrics["classification_report_txt"]),
            "test_confusion_matrix_csv": str(test_metrics["confusion_matrix_csv"]),
            "test_confusion_matrix_png": str(test_metrics["confusion_matrix_png"]),
            "test_confusion_matrix_txt": str(test_metrics["confusion_matrix_txt"]),
            "console_log": str((run_dir / "train_console.log").resolve())
        }
    }

    summary_json_path = run_dir / "run_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)
    write_run_summary(run_dir, summary)

    logger.info("========== 所有流程完成 ==========")
    logger.info(f"运行总结 JSON：{summary_json_path}")
    logger.info(f"后续推理建议使用：{latest_best_path.resolve()}")


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()