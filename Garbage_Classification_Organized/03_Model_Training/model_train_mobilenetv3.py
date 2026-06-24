# -*- coding: utf-8 -*-
"""
MobileNetV3 五分类垃圾分类训练脚本（优化版）

适用于目录结构：

garbage_dataset/
  class_names.json
  class_mapping.json
  train/
    待分拣/
    其他/
    厨余/
    可回收/
    有害/
  val/
    待分拣/
    其他/
    厨余/
    可回收/
    有害/
  test/
    待分拣/
    其他/
    厨余/
    可回收/
    有害/

修复与优化：
1. PyCharm 直接运行友好：内置默认 data_dir / output_dir。
2. 修复 Windows 路径转义 warning：所有默认路径使用 raw string。
3. 修复 torch.cuda.amp FutureWarning：优先使用 torch.amp 新接口。
4. 修复中文混淆矩阵字体 warning：自动尝试 Microsoft YaHei / SimHei 等中文字体。
5. 修复 --pretrained 默认 True 但不能关闭的问题：新增 --no-pretrained。
6. 修复无预训练却冻结 backbone 的风险：关闭预训练时自动取消冻结。
7. 保留 class_names.json / class_mapping.json 的权威类别顺序，不使用 ImageFolder 字典序。
8. 默认保存逐图预测 CSV，便于定位错图。
"""

import argparse
import csv
import json
import random
import shutil
import time
from collections import Counter
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageOps

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.metrics import classification_report, confusion_matrix

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


DEFAULT_DATA_DIR = r"D:\Garbage Classification\Garbage_Classification_Organized\garbage_dataset"
DEFAULT_OUTPUT_DIR = r"D:\Garbage Classification\Garbage_Classification_Organized\models\vision_trigger_5class_mobilenetv3"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EXPECTED_CLASSES = ["待分拣", "其他", "厨余", "可回收", "有害"]


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:")


def timestamp_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(name: str) -> torch.device:
    name = str(name).lower()

    if name == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if name == "cpu":
        return torch.device("cpu")

    if name == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def safe_torch_load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def setup_matplotlib_chinese() -> None:
    candidate_paths = [
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\msyh.ttf"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\simsun.ttc"),
        Path("/System/Library/Fonts/PingFang.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
    ]

    chosen_font_name = None

    for font_path in candidate_paths:
        if font_path.exists():
            try:
                font_manager.fontManager.addfont(str(font_path))
                chosen_font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
                break
            except Exception:
                pass

    if chosen_font_name:
        plt.rcParams["font.sans-serif"] = [
            chosen_font_name,
            "Microsoft YaHei",
            "SimHei",
            "SimSun",
            "Noto Sans CJK SC",
            "Arial Unicode MS",
            "DejaVu Sans",
        ]
    else:
        plt.rcParams["font.sans-serif"] = [
            "Microsoft YaHei",
            "SimHei",
            "SimSun",
            "Noto Sans CJK SC",
            "Arial Unicode MS",
            "DejaVu Sans",
        ]

    plt.rcParams["axes.unicode_minus"] = False


def make_grad_scaler(device: torch.device, use_amp: bool):
    if not use_amp:
        try:
            return torch.amp.GradScaler("cuda", enabled=False)
        except Exception:
            return torch.cuda.amp.GradScaler(enabled=False)

    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=True)


def autocast_context(device: torch.device, use_amp: bool):
    if not use_amp:
        return nullcontext()

    try:
        return torch.amp.autocast(device_type=device.type, enabled=True)
    except Exception:
        return torch.cuda.amp.autocast(enabled=True)


def load_class_names(data_dir: Path) -> List[str]:
    names_path = data_dir / "class_names.json"

    if names_path.exists():
        with open(names_path, "r", encoding="utf-8") as f:
            class_names = json.load(f)

        if isinstance(class_names, list) and len(class_names) == 5:
            return [str(x) for x in class_names]

    mapping_path = data_dir / "class_mapping.json"

    if mapping_path.exists():
        with open(mapping_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        c2i = data.get("class_to_idx", {})

        if isinstance(c2i, dict) and len(c2i) == 5:
            return [
                str(name)
                for name, _idx in sorted(c2i.items(), key=lambda kv: int(kv[1]))
            ]

    return EXPECTED_CLASSES[:]


def validate_class_names(class_names: Sequence[str]) -> None:
    if len(class_names) != 5:
        raise ValueError(f"类别数必须为 5，实际为 {len(class_names)}：{class_names}")

    if set(class_names) != set(EXPECTED_CLASSES):
        raise ValueError(
            "类别集合不匹配。\n"
            f"期望：{EXPECTED_CLASSES}\n"
            f"实际：{list(class_names)}"
        )

    if list(class_names) != EXPECTED_CLASSES:
        print("警告：类别顺序不是默认顺序，请确认训练、导出、推理都使用同一顺序：")
        print(f"默认顺序：{EXPECTED_CLASSES}")
        print(f"当前顺序：{list(class_names)}")


def validate_dataset_structure(data_dir: Path, class_names: Sequence[str]) -> None:
    if not data_dir.exists():
        raise FileNotFoundError(f"数据集目录不存在：{data_dir}")

    for split in ["train", "val", "test"]:
        split_dir = data_dir / split

        if not split_dir.exists():
            raise FileNotFoundError(f"缺少 split 目录：{split_dir}")

        for class_name in class_names:
            class_dir = split_dir / class_name

            if not class_dir.exists():
                raise FileNotFoundError(f"缺少类别目录：{class_dir}")


class SafeImageFolder(Dataset):
    def __init__(self, root: Path, class_names: Sequence[str], transform=None):
        self.root = Path(root)
        self.class_names = list(class_names)
        self.class_to_idx = {
            name: idx
            for idx, name in enumerate(self.class_names)
        }
        self.transform = transform
        self.samples: List[str] = []
        self.targets: List[int] = []

        for class_name in self.class_names:
            class_dir = self.root / class_name

            if not class_dir.exists():
                raise FileNotFoundError(f"缺少类别目录：{class_dir}")

            files = [
                p for p in class_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            ]
            files.sort()

            for file_path in files:
                self.samples.append(str(file_path))
                self.targets.append(self.class_to_idx[class_name])

        if not self.samples:
            raise RuntimeError(f"没有找到图片：{self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path = self.samples[idx]
        label = self.targets[idx]

        try:
            img = Image.open(path)
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"无法读取图片：{path}") from exc

        if self.transform:
            img = self.transform(img)

        return img, label, path


def get_transforms(image_size: int):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(
            brightness=0.25,
            contrast=0.25,
            saturation=0.18,
            hue=0.03,
        ),
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


def build_model(model_name: str, num_classes: int, pretrained: bool):
    model_name = str(model_name).lower()

    if model_name == "mobilenet_v3_small":
        weights = None

        if pretrained:
            try:
                weights = models.MobileNet_V3_Small_Weights.DEFAULT
            except Exception as exc:
                print(f"警告：MobileNetV3 Small 预训练权重加载失败，将使用随机初始化。原因：{exc}")

        model = models.mobilenet_v3_small(weights=weights)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        return model

    if model_name == "mobilenet_v3_large":
        weights = None

        if pretrained:
            try:
                weights = models.MobileNet_V3_Large_Weights.DEFAULT
            except Exception as exc:
                print(f"警告：MobileNetV3 Large 预训练权重加载失败，将使用随机初始化。原因：{exc}")

        model = models.mobilenet_v3_large(weights=weights)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        return model

    raise ValueError("model-name 只能是 mobilenet_v3_small 或 mobilenet_v3_large")


def freeze_backbone(model) -> None:
    for param in model.features.parameters():
        param.requires_grad = False

    for param in model.classifier.parameters():
        param.requires_grad = True


def unfreeze_all(model) -> None:
    for param in model.parameters():
        param.requires_grad = True


def make_class_weights(dataset: SafeImageFolder, num_classes: int, device: torch.device, enabled: bool = True):
    if not enabled:
        return None

    counts = Counter(dataset.targets)
    total = len(dataset.targets)

    weights = []

    for idx in range(num_classes):
        count = max(1, counts.get(idx, 0))
        weights.append(total / (num_classes * count))

    return torch.tensor(weights, dtype=torch.float32, device=device)


def count_trainable_params(model) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def train_one_epoch(model, loader, criterion, optimizer, device: torch.device, scaler, use_amp: bool):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_num = 0

    for images, labels, _paths in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast_context(device, use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        preds = logits.argmax(dim=1)

        total_loss += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_num += labels.size(0)

    return {
        "loss": total_loss / max(total_num, 1),
        "acc": total_correct / max(total_num, 1),
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device: torch.device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_num = 0

    y_true = []
    y_pred = []
    y_conf = []
    paths = []

    for images, labels, batch_paths in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        total_loss += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_num += labels.size(0)

        y_true.extend(labels.cpu().numpy().tolist())
        y_pred.extend(preds.cpu().numpy().tolist())
        y_conf.extend(probs.max(dim=1).values.cpu().numpy().tolist())
        paths.extend(list(batch_paths))

    return {
        "loss": total_loss / max(total_num, 1),
        "acc": total_correct / max(total_num, 1),
        "y_true": y_true,
        "y_pred": y_pred,
        "confidence": y_conf,
        "paths": paths,
    }


def save_csv(path: Path, rows: List[Dict], fieldnames: Sequence[str]) -> None:
    ensure_dir(path.parent)

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_confusion_matrix_png(path: Path, cm: np.ndarray, class_names: Sequence[str], title: str) -> None:
    setup_matplotlib_chinese()

    fig, ax = plt.subplots(figsize=(8, 7))
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
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def compute_custom_metrics(y_true: Sequence[int], y_pred: Sequence[int], class_names: Sequence[str]) -> Dict[str, float]:
    result: Dict[str, float] = {}

    if "待分拣" in class_names:
        pending_idx = class_names.index("待分拣")

        pending_total = sum(1 for y in y_true if y == pending_idx)
        pending_correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == pending_idx and yp == pending_idx)
        pending_false_trigger = sum(1 for yt, yp in zip(y_true, y_pred) if yt == pending_idx and yp != pending_idx)
        garbage_to_pending = sum(1 for yt, yp in zip(y_true, y_pred) if yt != pending_idx and yp == pending_idx)

        garbage_total = sum(1 for y in y_true if y != pending_idx)
        garbage_correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt != pending_idx and yt == yp)

        result["pending_recall"] = pending_correct / max(pending_total, 1)
        result["pending_false_trigger_count"] = int(pending_false_trigger)
        result["garbage_to_pending_count"] = int(garbage_to_pending)
        result["garbage_accuracy_excluding_pending"] = garbage_correct / max(garbage_total, 1)

    return result


def save_evaluation(run_dir: Path, split_name: str, eval_result, class_names: Sequence[str], save_per_image: bool):
    y_true = eval_result["y_true"]
    y_pred = eval_result["y_pred"]
    labels = list(range(len(class_names)))

    report_text = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=list(class_names),
        digits=4,
        zero_division=0,
    )

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=list(class_names),
        digits=4,
        zero_division=0,
        output_dict=True,
    )

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    report_txt_path = run_dir / f"{split_name}_classification_report.txt"
    report_json_path = run_dir / f"{split_name}_classification_report.json"
    cm_csv_path = run_dir / f"{split_name}_confusion_matrix.csv"
    cm_png_path = run_dir / f"{split_name}_confusion_matrix.png"
    metrics_path = run_dir / f"{split_name}_metrics.json"

    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, ensure_ascii=False, indent=4)

    cm_rows = []

    for i, true_name in enumerate(class_names):
        row = {"true_class": true_name}
        for j, pred_name in enumerate(class_names):
            row[pred_name] = int(cm[i, j])
        cm_rows.append(row)

    save_csv(cm_csv_path, cm_rows, ["true_class"] + list(class_names))
    save_confusion_matrix_png(cm_png_path, cm, class_names, f"{split_name} 混淆矩阵")

    custom = compute_custom_metrics(y_true, y_pred, class_names)

    metrics = {
        "split": split_name,
        "accuracy": float(eval_result["acc"]),
        "loss": float(eval_result["loss"]),
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
        **custom,
        "files": {
            "report_txt": str(report_txt_path.resolve()),
            "report_json": str(report_json_path.resolve()),
            "confusion_matrix_csv": str(cm_csv_path.resolve()),
            "confusion_matrix_png": str(cm_png_path.resolve()),
        },
    }

    if save_per_image:
        per_image_path = run_dir / f"{split_name}_per_image_predictions.csv"

        rows = []

        for path, yt, yp, conf in zip(eval_result["paths"], y_true, y_pred, eval_result["confidence"]):
            rows.append({
                "image_path": path,
                "true_idx": int(yt),
                "true_class": class_names[int(yt)],
                "pred_idx": int(yp),
                "pred_class": class_names[int(yp)],
                "confidence": float(conf),
                "correct": int(yt == yp),
            })

        save_csv(
            per_image_path,
            rows,
            ["image_path", "true_idx", "true_class", "pred_idx", "pred_class", "confidence", "correct"]
        )

        metrics["files"]["per_image_predictions_csv"] = str(per_image_path.resolve())

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)

    return metrics, report_text


def save_checkpoint(path: Path, model, optimizer, epoch: int, best_val_acc: float, class_names: Sequence[str], args) -> None:
    ensure_dir(path.parent)

    ckpt = {
        "epoch": int(epoch),
        "best_val_acc": float(best_val_acc),
        "model_name": args.model_name,
        "num_classes": len(class_names),
        "class_names": list(class_names),
        "class_to_idx": {name: idx for idx, name in enumerate(class_names)},
        "idx_to_class": {str(idx): name for idx, name in enumerate(class_names)},
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
        "created_time": now_str(),
    }

    torch.save(ckpt, path)


def write_dataset_summary(path: Path, train_ds: SafeImageFolder, val_ds: SafeImageFolder, test_ds: SafeImageFolder, class_names: Sequence[str]) -> None:
    rows = []

    for split_name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        counts = Counter(ds.targets)

        for idx, class_name in enumerate(class_names):
            rows.append({
                "split": split_name,
                "class_idx": idx,
                "class_name": class_name,
                "count": counts.get(idx, 0),
            })

    save_csv(path, rows, ["split", "class_idx", "class_name", "count"])


def save_class_mapping_files(run_dir: Path, class_names: Sequence[str]) -> None:
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    idx_to_class = {str(idx): name for idx, name in enumerate(class_names)}

    with open(run_dir / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(list(class_names), f, ensure_ascii=False, indent=4)

    with open(run_dir / "class_mapping.json", "w", encoding="utf-8") as f:
        json.dump({"class_to_idx": class_to_idx, "idx_to_class": idx_to_class}, f, ensure_ascii=False, indent=4)


def save_training_curve(run_dir: Path, log_rows: List[Dict]) -> None:
    if not log_rows:
        return

    setup_matplotlib_chinese()

    epochs = [int(r["epoch"]) for r in log_rows]
    train_acc = [float(r["train_acc"]) for r in log_rows]
    val_acc = [float(r["val_acc"]) for r in log_rows]
    train_loss = [float(r["train_loss"]) for r in log_rows]
    val_loss = [float(r["val_loss"]) for r in log_rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_acc, label="train_acc")
    ax.plot(epochs, val_acc, label="val_acc")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_title("训练/验证准确率")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "training_accuracy_curve.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_loss, label="train_loss")
    ax.plot(epochs, val_loss, label="val_loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("训练/验证损失")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "training_loss_curve.png", dpi=180)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser("MobileNetV3 五分类垃圾分类训练")

    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="划分后的 garbage_dataset 目录")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="模型输出目录")

    parser.add_argument("--model-name", default="mobilenet_v3_small", choices=["mobilenet_v3_small", "mobilenet_v3_large"])

    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true", help="启用确定性训练，可能变慢")

    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu", "mps"])

    parser.add_argument("--pretrained", dest="pretrained", action="store_true", default=True, help="使用 torchvision ImageNet 预训练权重，默认开启")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false", help="关闭预训练权重")

    parser.add_argument("--freeze-epochs", type=int, default=3, help="前几轮冻结 backbone，只训练分类头")
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--min-delta", type=float, default=1e-4)

    parser.add_argument("--use-class-weights", dest="use_class_weights", action="store_true", default=True, help="使用类别权重，默认开启")
    parser.add_argument("--no-class-weights", dest="use_class_weights", action="store_false", help="关闭类别权重")

    parser.add_argument("--save-per-image", dest="save_per_image", action="store_true", default=True, help="保存逐图预测 CSV，默认开启")
    parser.add_argument("--no-save-per-image", dest="save_per_image", action="store_false", help="关闭逐图预测 CSV")

    parser.add_argument("--dry-run", action="store_true", help="只检查数据，不开始训练")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.pretrained and args.freeze_epochs > 0:
        print("警告：当前关闭了预训练权重，不能冻结随机初始化的 backbone；已自动将 freeze_epochs 改为 0。")
        args.freeze_epochs = 0

    set_seed(args.seed, deterministic=args.deterministic)
    setup_matplotlib_chinese()

    data_dir = Path(args.data_dir).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()

    ensure_dir(output_root)

    run_dir = output_root / f"mobilenetv3_garbage_{timestamp_str()}"
    ensure_dir(run_dir)

    class_names = load_class_names(data_dir)
    validate_class_names(class_names)
    validate_dataset_structure(data_dir, class_names)

    num_classes = len(class_names)
    save_class_mapping_files(run_dir, class_names)

    print("========== MobileNetV3 五分类训练 ==========")
    print(f"数据集：{data_dir}")
    print(f"输出：{run_dir}")
    print(f"类别：{class_names}")

    train_tf, eval_tf = get_transforms(args.image_size)

    train_ds = SafeImageFolder(data_dir / "train", class_names, train_tf)
    val_ds = SafeImageFolder(data_dir / "val", class_names, eval_tf)
    test_ds = SafeImageFolder(data_dir / "test", class_names, eval_tf)

    write_dataset_summary(run_dir / "dataset_summary.csv", train_ds, val_ds, test_ds, class_names)

    print(f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
    print(f"class_to_idx={train_ds.class_to_idx}")

    if args.dry_run:
        print("dry-run 通过，不开始训练。")
        print(f"dataset_summary：{run_dir / 'dataset_summary.csv'}")
        return

    device = get_device(args.device)
    use_amp = device.type == "cuda"

    print(f"device={device}")
    print(f"use_amp={use_amp}")
    print(f"pretrained={args.pretrained}")
    print(f"use_class_weights={args.use_class_weights}")

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "drop_last": False,
    }

    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    model = build_model(args.model_name, num_classes, args.pretrained).to(device)

    if args.freeze_epochs > 0:
        freeze_backbone(model)
        print(f"前 {args.freeze_epochs} 个 epoch 冻结 backbone，只训练分类头。")
    else:
        unfreeze_all(model)
        print("不冻结 backbone，直接全模型训练。")

    total_params, trainable_params = count_trainable_params(model)
    print(f"参数量：total={total_params:,} trainable={trainable_params:,}")

    class_weights = make_class_weights(train_ds, num_classes, device, enabled=args.use_class_weights)

    if class_weights is not None:
        print(f"class_weights={class_weights.detach().cpu().numpy().round(4).tolist()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
    )

    scaler = make_grad_scaler(device, use_amp)

    best_val_acc = -1.0
    best_epoch = 0
    bad_epochs = 0

    best_path = run_dir / "mobilenetv3_best.pt"
    last_path = run_dir / "mobilenetv3_last.pt"
    latest_best_path = output_root / "latest_mobilenetv3_best.pt"

    train_log_path = run_dir / "training_log.csv"
    log_rows: List[Dict] = []

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        if epoch == args.freeze_epochs + 1 and args.freeze_epochs > 0:
            print("解冻 backbone，开始全模型微调。")
            unfreeze_all(model)

            optimizer = torch.optim.AdamW(
                [
                    {"params": model.features.parameters(), "lr": args.backbone_lr},
                    {"params": model.classifier.parameters(), "lr": args.lr},
                ],
                weight_decay=args.weight_decay,
            )

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=0.5,
                patience=3,
            )

            total_params, trainable_params = count_trainable_params(model)
            print(f"解冻后参数量：total={total_params:,} trainable={trainable_params:,}")

        train_result = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, use_amp)
        val_result = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_result["acc"])

        improved = val_result["acc"] > best_val_acc + args.min_delta

        if improved:
            best_val_acc = val_result["acc"]
            best_epoch = epoch
            bad_epochs = 0
            save_checkpoint(best_path, model, optimizer, epoch, best_val_acc, class_names, args)
            shutil.copy2(best_path, latest_best_path)
        else:
            bad_epochs += 1

        lr_values = [group["lr"] for group in optimizer.param_groups]

        row = {
            "epoch": epoch,
            "train_loss": train_result["loss"],
            "train_acc": train_result["acc"],
            "val_loss": val_result["loss"],
            "val_acc": val_result["acc"],
            "best_val_acc": best_val_acc,
            "best_epoch": best_epoch,
            "lr": "|".join(str(x) for x in lr_values),
            "improved": int(improved),
            "bad_epochs": bad_epochs,
            "time": now_str(),
        }

        log_rows.append(row)

        save_csv(
            train_log_path,
            log_rows,
            [
                "epoch",
                "train_loss",
                "train_acc",
                "val_loss",
                "val_acc",
                "best_val_acc",
                "best_epoch",
                "lr",
                "improved",
                "bad_epochs",
                "time",
            ]
        )

        save_training_curve(run_dir, log_rows)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_result['loss']:.4f} "
            f"train_acc={train_result['acc']:.4f} | "
            f"val_loss={val_result['loss']:.4f} "
            f"val_acc={val_result['acc']:.4f} | "
            f"best={best_val_acc:.4f}@{best_epoch}"
        )

        save_checkpoint(last_path, model, optimizer, epoch, best_val_acc, class_names, args)

        if bad_epochs >= args.patience:
            print(f"早停：连续 {bad_epochs} 个 epoch 验证集没有提升。")
            break

    if not best_path.exists():
        raise RuntimeError("训练结束后没有找到最佳模型，请检查训练过程是否异常。")

    print("加载最佳模型进行最终验证/测试...")

    loaded = safe_torch_load(best_path, device)
    state_dict = loaded.get("model_state_dict", loaded)

    model.load_state_dict(state_dict)
    model.eval()

    val_result = evaluate(model, val_loader, criterion, device)
    test_result = evaluate(model, test_loader, criterion, device)

    val_metrics, _val_report = save_evaluation(run_dir, "val", val_result, class_names, args.save_per_image)
    test_metrics, test_report = save_evaluation(run_dir, "test", test_result, class_names, args.save_per_image)

    elapsed_seconds = round(time.time() - start_time, 2)

    run_summary = {
        "created_time": now_str(),
        "elapsed_seconds": elapsed_seconds,
        "data_dir": str(data_dir),
        "output_dir": str(run_dir),
        "model_name": args.model_name,
        "pretrained": args.pretrained,
        "device": str(device),
        "use_amp": use_amp,
        "use_class_weights": args.use_class_weights,
        "class_names": class_names,
        "class_to_idx": {name: idx for idx, name in enumerate(class_names)},
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds),
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "final_val_acc": val_metrics["accuracy"],
        "final_test_acc": test_metrics["accuracy"],
        "test_custom_metrics": {
            k: v
            for k, v in test_metrics.items()
            if k in [
                "pending_recall",
                "pending_false_trigger_count",
                "garbage_to_pending_count",
                "garbage_accuracy_excluding_pending",
            ]
        },
        "files": {
            "best_model": str(best_path.resolve()),
            "last_model": str(last_path.resolve()),
            "latest_best_model": str(latest_best_path.resolve()),
            "training_log": str(train_log_path.resolve()),
            "dataset_summary": str((run_dir / "dataset_summary.csv").resolve()),
            "test_metrics": str((run_dir / "test_metrics.json").resolve()),
            "test_report": str((run_dir / "test_classification_report.txt").resolve()),
            "test_confusion_matrix": str((run_dir / "test_confusion_matrix.png").resolve()),
            "training_accuracy_curve": str((run_dir / "training_accuracy_curve.png").resolve()),
            "training_loss_curve": str((run_dir / "training_loss_curve.png").resolve()),
        },
        "args": vars(args),
    }

    with open(run_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(run_summary, f, ensure_ascii=False, indent=4)

    with open(run_dir / "run_summary.txt", "w", encoding="utf-8") as f:
        f.write("========== MobileNetV3 五分类训练总结 ==========\n")
        f.write(f"时间：{run_summary['created_time']}\n")
        f.write(f"耗时秒数：{elapsed_seconds}\n")
        f.write(f"数据集：{run_summary['data_dir']}\n")
        f.write(f"输出目录：{run_summary['output_dir']}\n")
        f.write(f"模型：{run_summary['model_name']}\n")
        f.write(f"预训练：{run_summary['pretrained']}\n")
        f.write(f"设备：{run_summary['device']}\n")
        f.write(f"AMP：{run_summary['use_amp']}\n")
        f.write(f"类别权重：{run_summary['use_class_weights']}\n")
        f.write(f"类别：{run_summary['class_names']}\n\n")
        f.write(f"best_epoch：{best_epoch}\n")
        f.write(f"best_val_acc：{best_val_acc:.6f}\n")
        f.write(f"final_val_acc：{val_metrics['accuracy']:.6f}\n")
        f.write(f"final_test_acc：{test_metrics['accuracy']:.6f}\n\n")
        f.write("========== Test Custom Metrics ==========\n")
        for key, value in run_summary["test_custom_metrics"].items():
            f.write(f"{key}: {value}\n")
        f.write("\n========== Test Report ==========\n")
        f.write(test_report)

    print("\n========== 训练完成 ==========")
    print(f"best_epoch: {best_epoch}")
    print(f"best_val_acc: {best_val_acc:.4f}")
    print(f"final_val_acc: {val_metrics['accuracy']:.4f}")
    print(f"final_test_acc: {test_metrics['accuracy']:.4f}")
    print(f"最佳模型：{best_path}")
    print(f"latest 模型：{latest_best_path}")
    print(f"测试报告：{run_dir / 'test_classification_report.txt'}")
    print(f"混淆矩阵：{run_dir / 'test_confusion_matrix.png'}")
    print(f"训练曲线：{run_dir / 'training_accuracy_curve.png'}")
    print(f"训练总结：{run_dir / 'run_summary.txt'}")


if __name__ == "__main__":
    main()
