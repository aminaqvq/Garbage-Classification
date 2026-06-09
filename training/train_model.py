import csv
import json
import time
import random
import shutil
import logging
from pathlib import Path
from datetime import datetime
from collections import Counter

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision import datasets, transforms, models

from sklearn.metrics import classification_report, confusion_matrix

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================================================
# 配置区：主要改这里
# =========================================================

# 划分后的数据集根目录
# 目录下面应该有 train / val / test
# 例如：
# dataset/
#   train/
#   val/
#   test/
DATA_DIR = Path("garbage_dataset")



# 输出目录
OUTPUT_ROOT = Path("outputs")

# 模型选择：small 或 large
MODEL_NAME = "mobilenet_v3_large"

# 固定四分类
NUM_CLASSES = 4

# 训练参数
BATCH_SIZE = 32
MAX_STEPS = 0  # 0 = 不限制；>0 时按 batch step 硬上限，超出后终止训练
EPOCHS = 999

# 第一阶段：只训练分类头
HEAD_LR = 1e-3

# 第二阶段：解冻 backbone 后微调
USE_FINE_TUNE = True
UNFREEZE_AFTER_EPOCH = 8
BACKBONE_LR = 1e-4
HEAD_FINE_TUNE_LR = 5e-4

# 早停
USE_EARLY_STOP = True
PATIENCE = 7
MIN_DELTA = 1e-4

# 学习率调度
USE_SCHEDULER = True
SCHEDULER_PATIENCE = 3
SCHEDULER_FACTOR = 0.5

# 类别不均衡时建议开启
USE_CLASS_WEIGHTS = True

# DataLoader
NUM_WORKERS = 2

# 日志打印频率
LOG_INTERVAL = 20

# 随机种子
SEED = 42

# 是否使用 AMP 混合精度
# 只有 CUDA 可用时才会启用
USE_AMP = True

# 是否保存每张图片的预测结果
SAVE_PER_IMAGE_PREDICTIONS = True


# =========================================================
# 工具函数
# =========================================================

def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def make_run_dir() -> Path:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_ROOT / f"mobilenetv3_garbage_{run_id}"
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
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def check_dataset_dir(data_dir: Path):
    required = ["train", "val", "test"]

    if not data_dir.exists():
        raise FileNotFoundError(f"数据集目录不存在：{data_dir.resolve()}")

    for split in required:
        split_dir = data_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(f"缺少 {split} 目录：{split_dir.resolve()}")


def get_transforms():
    """
    使用 ImageNet 预训练模型常用 Normalize。
    MobileNetV3 预训练权重默认输入尺寸为 224。
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.70, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05
        ),
        transforms.ToTensor(),
        normalize,
    ])

    eval_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    return train_tf, eval_tf


def build_datasets(data_dir: Path):
    train_tf, eval_tf = get_transforms()

    train_ds = datasets.ImageFolder(data_dir / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(data_dir / "val", transform=eval_tf)
    test_ds = datasets.ImageFolder(data_dir / "test", transform=eval_tf)

    if len(train_ds.classes) != NUM_CLASSES:
        raise ValueError(
            f"检测到类别数为 {len(train_ds.classes)}，但 NUM_CLASSES={NUM_CLASSES}。"
            f"请检查 train 目录下类别文件夹。"
        )

    if train_ds.class_to_idx != val_ds.class_to_idx:
        raise ValueError(
            "train 和 val 的 class_to_idx 不一致，请检查类别文件夹是否一致。"
        )

    if train_ds.class_to_idx != test_ds.class_to_idx:
        raise ValueError(
            "train 和 test 的 class_to_idx 不一致，请检查类别文件夹是否一致。"
        )

    return train_ds, val_ds, test_ds


def build_loader(dataset, batch_size, shuffle, device, num_workers):
    pin_memory = device.type == "cuda"

    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": False,
    }

    if num_workers > 0:
        kwargs["persistent_workers"] = True

    return DataLoader(dataset, **kwargs)


def dataset_class_counts(dataset):
    counter = Counter(dataset.targets)
    result = {}

    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}

    for idx in range(len(idx_to_class)):
        class_name = idx_to_class[idx]
        result[class_name] = counter.get(idx, 0)

    return result


def save_dataset_summary(run_dir: Path, train_ds, val_ds, test_ds):
    path = run_dir / "dataset_summary.csv"

    rows = []

    split_map = {
        "train": train_ds,
        "val": val_ds,
        "test": test_ds,
    }

    for split_name, ds in split_map.items():
        counts = dataset_class_counts(ds)

        for class_name, count in counts.items():
            rows.append({
                "split": split_name,
                "class_name": class_name,
                "count": count
            })

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "class_name", "count"])
        writer.writeheader()
        writer.writerows(rows)

    return path


def save_class_mapping(run_dir: Path, class_to_idx: dict):
    idx_to_class = {str(v): k for k, v in class_to_idx.items()}

    path = run_dir / "class_mapping.json"

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "class_to_idx": class_to_idx,
                "idx_to_class": idx_to_class
            },
            f,
            ensure_ascii=False,
            indent=4
        )

    return path


def save_config(run_dir: Path, device, train_ds, val_ds, test_ds):
    config = {
        "created_time": now_str(),
        "data_dir": str(DATA_DIR.resolve()),
        "output_dir": str(run_dir.resolve()),
        "device": str(device),
        "model_name": MODEL_NAME,
        "num_classes": NUM_CLASSES,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "max_steps": MAX_STEPS,
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
        "num_workers": NUM_WORKERS,
        "seed": SEED,
        "use_amp": USE_AMP,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds),
        "class_to_idx": train_ds.class_to_idx,
        "transforms": {
            "train": [
                "RandomResizedCrop(224, scale=(0.70, 1.0))",
                "RandomHorizontalFlip(p=0.5)",
                "RandomRotation(10)",
                "ColorJitter(0.2, 0.2, 0.2, 0.05)",
                "ToTensor",
                "Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])"
            ],
            "val_test": [
                "Resize(256)",
                "CenterCrop(224)",
                "ToTensor",
                "Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])"
            ]
        }
    }

    path = run_dir / "config.json"

    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

    return path


def build_model(num_classes: int):
    if MODEL_NAME == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        model = models.mobilenet_v3_small(weights=weights)

    elif MODEL_NAME == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT
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
        lr=HEAD_LR,
        weight_decay=1e-4
    )


def build_optimizer_fine_tune(model):
    return torch.optim.AdamW(
        [
            {
                "params": model.features.parameters(),
                "lr": BACKBONE_LR
            },
            {
                "params": model.classifier.parameters(),
                "lr": HEAD_FINE_TUNE_LR
            }
        ],
        weight_decay=1e-4
    )


def build_scheduler(optimizer):
    if not USE_SCHEDULER:
        return None

    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE
    )


def get_current_lr(optimizer):
    return [group["lr"] for group in optimizer.param_groups]


def make_class_weights(train_ds, device):
    counts = Counter(train_ds.targets)
    total = len(train_ds)
    num_classes = len(train_ds.classes)

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


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    epoch,
    logger,
    batch_log_path,
    scaler=None,
    use_amp=False
):
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

            logger.info(
                f"[Epoch {epoch:03d}] "
                f"step {step:04d}/{len(loader)} | "
                f"loss={avg_loss:.4f} | "
                f"acc={avg_acc:.4f} | "
                f"lr={lr_list}"
            )

            append_csv(
                batch_log_path,
                [
                    "time",
                    "epoch",
                    "step",
                    "total_steps",
                    "train_loss_so_far",
                    "train_acc_so_far",
                    "lr"
                ],
                {
                    "time": now_str(),
                    "epoch": epoch,
                    "step": step,
                    "total_steps": len(loader),
                    "train_loss_so_far": avg_loss,
                    "train_acc_so_far": avg_acc,
                    "lr": json.dumps(lr_list)
                }
            )

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

    result = {
        "loss": loss_sum / max(total, 1),
        "acc": correct / max(total, 1),
        "total": total,
    }

    if collect_predictions:
        result["y_true"] = all_true
        result["y_pred"] = all_pred
        result["confidence"] = all_conf

    return result


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    epoch,
    best_val_acc,
    train_ds,
    extra: dict
):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "epoch": epoch,
            "best_val_acc": best_val_acc,
            "class_to_idx": train_ds.class_to_idx,
            "idx_to_class": {v: k for k, v in train_ds.class_to_idx.items()},
            "model_name": MODEL_NAME,
            "num_classes": NUM_CLASSES,
            "extra": extra
        },
        path
    )


def safe_torch_load(path: Path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def setup_matplotlib_chinese():
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "PingFang SC",
        "Noto Sans CJK SC",
        "Arial Unicode MS"
    ]
    plt.rcParams["axes.unicode_minus"] = False


def save_confusion_matrix_csv(path: Path, cm: np.ndarray, class_names: list):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)

        writer.writerow(["真实\\预测"] + class_names)

        for i, row in enumerate(cm):
            writer.writerow([class_names[i]] + row.tolist())


def save_confusion_matrix_png(path: Path, cm: np.ndarray, class_names: list, title: str):
    setup_matplotlib_chinese()

    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(cm)
    fig.colorbar(im, ax=ax)

    ax.set_title(title)
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))

    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center"
            )

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_per_image_predictions(
    path: Path,
    dataset,
    y_true: list,
    y_pred: list,
    confidence: list,
    idx_to_class: dict
):
    samples = dataset.samples

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "image_path",
            "true_idx",
            "true_class",
            "pred_idx",
            "pred_class",
            "confidence",
            "correct"
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, sample in enumerate(samples):
            image_path, _ = sample

            true_idx = y_true[i]
            pred_idx = y_pred[i]

            writer.writerow({
                "image_path": image_path,
                "true_idx": true_idx,
                "true_class": idx_to_class[true_idx],
                "pred_idx": pred_idx,
                "pred_class": idx_to_class[pred_idx],
                "confidence": confidence[i],
                "correct": int(true_idx == pred_idx)
            })


def save_final_evaluation(
    run_dir: Path,
    split_name: str,
    dataset,
    eval_result: dict,
    idx_to_class: dict
):
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    y_true = eval_result["y_true"]
    y_pred = eval_result["y_pred"]
    confidence = eval_result["confidence"]

    labels = list(range(len(class_names)))

    report_text = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        digits=4,
        zero_division=0
    )

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        digits=4,
        zero_division=0,
        output_dict=True
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    report_txt_path = run_dir / f"{split_name}_classification_report.txt"
    report_json_path = run_dir / f"{split_name}_classification_report.json"
    cm_csv_path = run_dir / f"{split_name}_confusion_matrix.csv"
    cm_png_path = run_dir / f"{split_name}_confusion_matrix.png"
    prediction_csv_path = run_dir / f"{split_name}_per_image_predictions.csv"

    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, ensure_ascii=False, indent=4)

    save_confusion_matrix_csv(cm_csv_path, cm, class_names)
    save_confusion_matrix_png(
        cm_png_path,
        cm,
        class_names,
        title=f"{split_name} Confusion Matrix"
    )

    if SAVE_PER_IMAGE_PREDICTIONS:
        save_per_image_predictions(
            prediction_csv_path,
            dataset,
            y_true,
            y_pred,
            confidence,
            idx_to_class
        )

    metrics_path = run_dir / f"{split_name}_metrics.json"

    metrics = {
        "split": split_name,
        "loss": eval_result["loss"],
        "accuracy": eval_result["acc"],
        "total": eval_result["total"],
        "classification_report_txt": str(report_txt_path.resolve()),
        "classification_report_json": str(report_json_path.resolve()),
        "confusion_matrix_csv": str(cm_csv_path.resolve()),
        "confusion_matrix_png": str(cm_png_path.resolve()),
        "per_image_predictions_csv": str(prediction_csv_path.resolve())
        if SAVE_PER_IMAGE_PREDICTIONS else None
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)

    return metrics, report_text


def write_run_summary(run_dir: Path, summary: dict):
    path = run_dir / "run_summary.txt"

    lines = []
    lines.append("========== MobileNetV3 垃圾分类训练报告 ==========")
    lines.append(f"生成时间：{now_str()}")
    lines.append("")
    lines.append(f"数据集目录：{summary['data_dir']}")
    lines.append(f"输出目录：{summary['output_dir']}")
    lines.append(f"设备：{summary['device']}")
    lines.append(f"模型：{summary['model_name']}")
    lines.append("")
    lines.append("类别映射：")

    for idx, class_name in summary["idx_to_class"].items():
        lines.append(f"{idx}: {class_name}")

    lines.append("")
    lines.append("训练结果：")
    lines.append(f"最佳 epoch：{summary['best_epoch']}")
    lines.append(f"最佳 val_acc：{summary['best_val_acc']:.6f}")
    lines.append(f"最终 val_acc：{summary['final_val_acc']:.6f}")
    lines.append(f"最终 test_acc：{summary['final_test_acc']:.6f}")
    lines.append("")
    lines.append("关键文件：")

    for key, value in summary["files"].items():
        lines.append(f"{key}: {value}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return path


# =========================================================
# 主程序
# =========================================================

def main():
    set_seed(SEED)

    run_dir = make_run_dir()
    logger = setup_logger(run_dir)

    device = get_device()
    use_amp = USE_AMP and device.type == "cuda"

    logger.info("========== MobileNetV3 垃圾分类训练开始 ==========")
    logger.info(f"数据集目录：{DATA_DIR.resolve()}")
    logger.info(f"输出目录：{run_dir.resolve()}")
    logger.info(f"设备：{device}")
    logger.info(f"AMP：{use_amp}")

    check_dataset_dir(DATA_DIR)

    train_ds, val_ds, test_ds = build_datasets(DATA_DIR)

    logger.info("========== Dataset ==========")
    logger.info(f"train size: {len(train_ds)}")
    logger.info(f"val size: {len(val_ds)}")
    logger.info(f"test size: {len(test_ds)}")
    logger.info(f"class_to_idx: {train_ds.class_to_idx}")

    dataset_summary_path = save_dataset_summary(run_dir, train_ds, val_ds, test_ds)
    class_mapping_path = save_class_mapping(run_dir, train_ds.class_to_idx)
    config_path = save_config(run_dir, device, train_ds, val_ds, test_ds)

    logger.info(f"数据集统计已保存：{dataset_summary_path}")
    logger.info(f"类别映射已保存：{class_mapping_path}")
    logger.info(f"配置文件已保存：{config_path}")

    train_loader = build_loader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        device=device,
        num_workers=NUM_WORKERS
    )

    val_loader = build_loader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        device=device,
        num_workers=NUM_WORKERS
    )

    test_loader = build_loader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        device=device,
        num_workers=NUM_WORKERS
    )

    model = build_model(NUM_CLASSES)

    freeze_backbone(model)
    model = model.to(device)

    if USE_CLASS_WEIGHTS:
        class_weights = make_class_weights(train_ds, device)
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
        "time",
        "epoch",
        "stage",
        "lr",
        "train_loss",
        "train_acc",
        "val_loss",
        "val_acc",
        "best_val_acc",
        "epoch_time_seconds",
        "is_best",
        "bad_epochs"
    ]

    best_val_acc = 0.0
    best_epoch = 0
    bad_epochs = 0
    global_step = 0  # batch-level step counter for MAX_STEPS safety net
    current_stage = "head_only"

    idx_to_class = {v: k for k, v in train_ds.class_to_idx.items()}

    logger.info("========== 开始训练 ==========")

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        if (
            USE_FINE_TUNE
            and current_stage == "head_only"
            and epoch == UNFREEZE_AFTER_EPOCH + 1
        ):
            logger.info("========== 进入第二阶段：解冻 backbone，开始 fine-tune ==========")

            unfreeze_backbone(model)
            optimizer = build_optimizer_fine_tune(model)
            scheduler = build_scheduler(optimizer)

            current_stage = "fine_tune"
            bad_epochs = 0

        train_loss, train_acc, train_time = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            logger=logger,
            batch_log_path=batch_log_path,
            scaler=scaler,
            use_amp=use_amp
        )

        val_result = evaluate_model(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            collect_predictions=False
        )

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

            save_checkpoint(
                path=best_model_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_acc=best_val_acc,
                train_ds=train_ds,
                extra={
                    "stage": current_stage,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "lr": lr_now
                }
            )

            logger.info(
                f"[BEST] epoch={epoch} | "
                f"best_val_acc={best_val_acc:.6f} | "
                f"saved={best_model_path.name}"
            )

        else:
            bad_epochs += 1

        save_checkpoint(
            path=last_model_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_val_acc=best_val_acc,
            train_ds=train_ds,
            extra={
                "stage": current_stage,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "lr": lr_now
            }
        )

        logger.info(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"stage={current_stage} | "
            f"lr={lr_now} | "
            f"train_loss={train_loss:.6f} | "
            f"train_acc={train_acc:.6f} | "
            f"val_loss={val_loss:.6f} | "
            f"val_acc={val_acc:.6f} | "
            f"best_val_acc={best_val_acc:.6f} | "
            f"time={epoch_time:.1f}s | "
            f"bad_epochs={bad_epochs}/{PATIENCE}"
        )

        append_csv(
            train_log_path,
            train_log_fields,
            {
                "time": now_str(),
                "epoch": epoch,
                "stage": current_stage,
                "lr": json.dumps(lr_now),
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "best_val_acc": best_val_acc,
                "epoch_time_seconds": epoch_time,
                "is_best": int(is_best),
                "bad_epochs": bad_epochs
            }
        )

        # MAX_STEPS 硬上限安全检查（0 = 不限制）
        global_step += len(train_loader)
        if MAX_STEPS > 0 and global_step >= MAX_STEPS:
            logger.info(
                f"[MAX_STEPS] 已达到最大步数 {MAX_STEPS}，"
                f"停止于 epoch={epoch}，step={global_step}。"
            )
            break

        if USE_EARLY_STOP and bad_epochs >= PATIENCE:
            logger.info(
                f"[EARLY STOP] 连续 {PATIENCE} 轮 val_acc 没有提升，"
                f"提前停止于 epoch={epoch}。"
            )
            break

    logger.info("========== 训练结束 ==========")
    logger.info(f"最佳 epoch：{best_epoch}")
    logger.info(f"最佳 val_acc：{best_val_acc:.6f}")
    logger.info(f"best 模型路径：{best_model_path}")

    # =====================================================
    # 训练结束后：自动加载 best 模型，重新验证 val 和 test
    # =====================================================

    logger.info("========== 加载 best 模型，开始最终验证和测试 ==========")

    checkpoint = safe_torch_load(best_model_path, device)
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)

    final_val_result = evaluate_model(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        collect_predictions=True
    )

    final_test_result = evaluate_model(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        collect_predictions=True
    )

    val_metrics, val_report_text = save_final_evaluation(
        run_dir=run_dir,
        split_name="val",
        dataset=val_ds,
        eval_result=final_val_result,
        idx_to_class=idx_to_class
    )

    test_metrics, test_report_text = save_final_evaluation(
        run_dir=run_dir,
        split_name="test",
        dataset=test_ds,
        eval_result=final_test_result,
        idx_to_class=idx_to_class
    )

    logger.info("========== Final Val Report ==========")
    logger.info("\n" + val_report_text)

    logger.info("========== Final Test Report ==========")
    logger.info("\n" + test_report_text)

    logger.info(f"最终 val_acc：{final_val_result['acc']:.6f}")
    logger.info(f"最终 test_acc：{final_test_result['acc']:.6f}")

    # 复制一份 best 模型到 OUTPUT_ROOT 下，方便后续 predict.py 直接找
    latest_best_path = OUTPUT_ROOT / "latest_mobilenetv3_best.pt"
    shutil.copy2(best_model_path, latest_best_path)

    summary = {
        "data_dir": str(DATA_DIR.resolve()),
        "output_dir": str(run_dir.resolve()),
        "device": str(device),
        "model_name": MODEL_NAME,
        "idx_to_class": {str(k): v for k, v in idx_to_class.items()},
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "final_val_acc": final_val_result["acc"],
        "final_test_acc": final_test_result["acc"],
        "files": {
            "best_model": str(best_model_path.resolve()),
            "last_model": str(last_model_path.resolve()),
            "latest_best_model_copy": str(latest_best_path.resolve()),
            "training_log": str(train_log_path.resolve()),
            "batch_log": str(batch_log_path.resolve()),
            "dataset_summary": str(dataset_summary_path.resolve()),
            "class_mapping": str(class_mapping_path.resolve()),
            "config": str(config_path.resolve()),
            "val_report": val_metrics["classification_report_txt"],
            "val_confusion_matrix_csv": val_metrics["confusion_matrix_csv"],
            "val_confusion_matrix_png": val_metrics["confusion_matrix_png"],
            "val_per_image_predictions": val_metrics["per_image_predictions_csv"],
            "test_report": test_metrics["classification_report_txt"],
            "test_confusion_matrix_csv": test_metrics["confusion_matrix_csv"],
            "test_confusion_matrix_png": test_metrics["confusion_matrix_png"],
            "test_per_image_predictions": test_metrics["per_image_predictions_csv"],
            "console_log": str((run_dir / "train_console.log").resolve())
        }
    }

    summary_json_path = run_dir / "run_summary.json"

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)

    summary_txt_path = write_run_summary(run_dir, summary)

    logger.info("========== 所有流程完成 ==========")
    logger.info(f"运行总结 JSON：{summary_json_path}")
    logger.info(f"运行总结 TXT：{summary_txt_path}")
    logger.info(f"后续推理建议使用：{latest_best_path.resolve()}")


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
