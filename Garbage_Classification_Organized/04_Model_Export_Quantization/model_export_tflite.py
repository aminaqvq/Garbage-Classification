import os
import sys
import csv
import json
import time
import shutil
import random
import argparse
import logging
import warnings
from pathlib import Path
from datetime import datetime
from collections import OrderedDict, Counter
from typing import Any, Dict, List, Tuple, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent

import numpy as np
from PIL import Image

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import torch
import torch.nn as nn
from torchvision import models

# Truly lazy imports — never executed at module level, only when accessed
_ONNX_AVAILABLE = False
_ONNXRT_AVAILABLE = False
_ONNXSIM_AVAILABLE = False
_TF_AVAILABLE = False

class _DummyModule:
    def __getattr__(self, name):
        return _DummyModule()
    def __call__(self, *args, **kwargs):
        return _DummyModule()

_dummy = _DummyModule()
onnx = _dummy
ort = _dummy
simplify = _dummy
tf = _dummy

def _lazy_import_onnx():
    global onnx, _ONNX_AVAILABLE
    if _ONNX_AVAILABLE: return
    try:
        import onnx as _m
        onnx = _m; _ONNX_AVAILABLE = True
    except Exception: pass

def _lazy_import_ort():
    global ort, _ONNXRT_AVAILABLE
    if _ONNXRT_AVAILABLE: return
    try:
        import onnxruntime as _m
        ort = _m; _ONNXRT_AVAILABLE = True
    except Exception: pass

def _lazy_import_onnxsim():
    global simplify, _ONNXSIM_AVAILABLE
    if _ONNXSIM_AVAILABLE: return
    try:
        from onnxsim import simplify as _f
        simplify = _f; _ONNXSIM_AVAILABLE = True
    except Exception: pass

def _lazy_import_tf():
    global tf, _TF_AVAILABLE
    if _TF_AVAILABLE: return
    try:
        import tensorflow as _m
        tf = _m; _TF_AVAILABLE = True
    except Exception: pass


# =========================================================
# 默认配置（可由命令行参数覆盖）
# =========================================================

CONFIG: Dict[str, Any] = {
    "data_dir": "garbage_dataset",
    "ckpt": "models/vision_trigger_5class_mobilenetv3/latest_mobilenetv3_best.pt",
    "outdir": "models/vision_trigger_5class_tflite",

    "model_name": "auto",
    "num_classes": "auto",
    "device": "cpu",

    "img_size": 224,
    "resize_size": 256,
    "eval_preprocess": "resize_center_crop",
    "normalize": True,
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "rgb": True,

    "export_tflite_float32": True,
    "export_tflite_fp16": False,
    "export_tflite_int8": False,

    "test_split": "test",
    "verify_limit": 0,

    "save_per_image_predictions": True,
    "save_confusion_matrix_png": True,

    "class_config_path": r"D:\Garbage Classification\Garbage_Classification_Organized\garbage_dataset\class_mapping.json",

    "quantize_float16": False,
    "quantize_int8": False,
}



def merge_cli_to_config(args: argparse.Namespace) -> Dict[str, Any]:
    """将命令行参数合并到 CONFIG 字典中。"""
    cfg = dict(CONFIG)
    if getattr(args, "model_path", None):
        cfg["ckpt"] = str(args.model_path)
    if getattr(args, "output_dir", None):
        cfg["outdir"] = str(args.output_dir)
    if getattr(args, "data_dir", None):
        cfg["data_dir"] = str(args.data_dir)
    if getattr(args, "class_config", None):
        cfg["class_config_path"] = str(args.class_config)
    if getattr(args, "model_name", None) and str(args.model_name).lower() != "auto":
        cfg["model_name"] = str(args.model_name).lower()
    if getattr(args, "num_classes", None):
        raw = str(args.num_classes)
        if raw.lower() == "auto":
            cfg["num_classes"] = "auto"
        else:
            cfg["num_classes"] = int(args.num_classes)
    if getattr(args, "image_size", None):
        cfg["img_size"] = int(args.image_size)
    if getattr(args, "batch_size", None):
        cfg["batch_size"] = int(args.batch_size)
    if getattr(args, "device", None):
        d = str(args.device).lower()
        if d == "auto":
            cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            cfg["device"] = d
    if getattr(args, "dry_run", None):
        cfg["run_dry_run"] = True
    if getattr(args, "verify", None):
        cfg["run_verify"] = True
    if getattr(args, "verify_samples", None):
        cfg["verify_limit"] = int(args.verify_samples)
    if getattr(args, "quantize_float16", None):
        cfg["quantize_float16"] = True
        cfg["export_tflite_fp16"] = True
    else:
        cfg["quantize_float16"] = False
    if getattr(args, "quantize_int8", None):
        cfg["quantize_int8"] = True
        cfg["export_tflite_int8"] = True
    return cfg


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# =========================================================
# 日志与基础工具
# =========================================================

def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def timestamp_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def remove_path(path: Path):
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def prepare_run_dir(cfg: Dict[str, Any]) -> Path:
    out_root = Path(cfg["outdir"])

    if out_root.exists() and cfg.get("overwrite_outdir", False):
        shutil.rmtree(out_root)

    run_dir = out_root / f"export_{timestamp_str()}"
    run_dir.mkdir(parents=True, exist_ok=True)

    return run_dir


def setup_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("quant_export")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)

    file_handler = logging.FileHandler(run_dir / "console.log", encoding="utf-8")
    file_handler.setFormatter(fmt)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def get_device(device_name: str) -> torch.device:
    if str(device_name).lower() == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def write_json(path: Path, data: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def file_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    return path.stat().st_size / 1024 / 1024


# =========================================================
# checkpoint 加载
# =========================================================

def check_export_dependencies(logger: logging.Logger) -> List[str]:
    """检查导出依赖是否齐全，返回缺失列表。"""
    missing = []
    if not _ONNX_AVAILABLE:
        missing.append("onnx")
    if not _ONNXRT_AVAILABLE:
        missing.append("onnxruntime")
    if not _TF_AVAILABLE:
        missing.append("tensorflow")
    return missing


def safe_torch_load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def strip_module_prefix(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    out = OrderedDict()

    for k, v in state_dict.items():
        nk = k[7:] if k.startswith("module.") else k
        out[nk] = v

    return out


def extract_state_dict(loaded: Any) -> Dict[str, Any]:
    if isinstance(loaded, dict):
        preferred_keys = [
            "model",
            "state_dict",
            "model_state_dict",
            "net",
            "network",
            "ema_state_dict",
        ]

        for key in preferred_keys:
            value = loaded.get(key, None)
            if isinstance(value, dict) and len(value) > 0:
                return value

        if all(isinstance(k, str) for k in loaded.keys()):
            return loaded

    raise ValueError("无法从 checkpoint 中提取 state_dict。")


def infer_num_classes_from_state_dict(state_dict: Dict[str, Any]) -> Optional[int]:
    possible_keys = [
        "classifier.3.weight",
        "module.classifier.3.weight",
    ]

    for key in possible_keys:
        if key in state_dict:
            return int(state_dict[key].shape[0])

    for key, value in state_dict.items():
        if key.endswith("classifier.3.weight") and hasattr(value, "shape"):
            return int(value.shape[0])

    return None


def normalize_idx_to_class(idx_to_class_raw: Dict[Any, Any]) -> Dict[int, str]:
    result = {}

    for k, v in idx_to_class_raw.items():
        result[int(k)] = str(v)

    return result


def get_class_mapping_from_checkpoint(
    loaded: Any,
    data_dir: Path,
    cfg: Dict[str, Any],
    logger: logging.Logger
) -> Tuple[Dict[str, int], Dict[int, str]]:
    if isinstance(loaded, dict):
        class_to_idx = loaded.get("class_to_idx", None)
        idx_to_class = loaded.get("idx_to_class", None)

        if isinstance(class_to_idx, dict) and class_to_idx:
            class_to_idx = {str(k): int(v) for k, v in class_to_idx.items()}

            if isinstance(idx_to_class, dict) and idx_to_class:
                idx_to_class = normalize_idx_to_class(idx_to_class)
            else:
                idx_to_class = {idx: name for name, idx in class_to_idx.items()}

            return class_to_idx, idx_to_class

    logger.warning("checkpoint 中没有 class_to_idx，将从 test/train 文件夹名自动推断类别。")

    split_dir = data_dir / str(cfg.get("test_split", "test"))

    if not split_dir.exists():
        split_dir = data_dir / "train"

    classes = sorted([
        p.name for p in split_dir.iterdir()
        if p.is_dir()
    ])

    if not classes:
        raise ValueError("无法从数据集目录推断类别，请检查 data_dir。")

    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}

    return class_to_idx, idx_to_class


def get_model_name_from_checkpoint(loaded: Any, cfg: Dict[str, Any]) -> str:
    cfg_model = str(cfg.get("model_name", "auto")).lower()

    if cfg_model != "auto":
        return cfg_model

    if isinstance(loaded, dict):
        ckpt_model = loaded.get("model_name", None)
        if ckpt_model:
            return str(ckpt_model).lower()

        extra = loaded.get("extra", None)
        if isinstance(extra, dict) and extra.get("model_name", None):
            return str(extra["model_name"]).lower()

    return "mobilenet_v3_small"


def get_num_classes(
    loaded: Any,
    state_dict: Dict[str, Any],
    class_to_idx: Dict[str, int],
    cfg: Dict[str, Any]
) -> int:
    cfg_num_classes = cfg.get("num_classes", "auto")

    if str(cfg_num_classes).lower() != "auto":
        return int(cfg_num_classes)

    if class_to_idx:
        return len(class_to_idx)

    if isinstance(loaded, dict):
        ckpt_num = loaded.get("num_classes", None)
        if ckpt_num is not None:
            return int(ckpt_num)

    inferred = infer_num_classes_from_state_dict(state_dict)

    if inferred is not None:
        return inferred

    raise ValueError("无法确定 num_classes，请在 CONFIG 中手动设置。")


# =========================================================
# 模型构建
# =========================================================

def build_model(model_name: str, num_classes: int) -> nn.Module:
    model_name = str(model_name).lower()

    if model_name == "mobilenet_v3_small":
        try:
            model = models.mobilenet_v3_small(weights=None, num_classes=num_classes)
        except TypeError:
            model = models.mobilenet_v3_small(weights=None)
            model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        return model

    if model_name == "mobilenet_v3_large":
        try:
            model = models.mobilenet_v3_large(weights=None, num_classes=num_classes)
        except TypeError:
            model = models.mobilenet_v3_large(weights=None)
            model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        return model

    raise ValueError("model_name 只能是 mobilenet_v3_small、mobilenet_v3_large 或 auto。")


def load_model_from_checkpoint(
    cfg: Dict[str, Any],
    device: torch.device,
    data_dir: Path,
    logger: logging.Logger
):
    ckpt_path = Path(cfg["ckpt"])

    if not ckpt_path.exists():
        raise FileNotFoundError(f"找不到 checkpoint：{ckpt_path.resolve()}")

    loaded = safe_torch_load(ckpt_path, device)
    state_dict = strip_module_prefix(extract_state_dict(loaded))

    class_to_idx, idx_to_class = get_class_mapping_from_checkpoint(
        loaded,
        data_dir,
        cfg,
        logger
    )

    model_name = get_model_name_from_checkpoint(loaded, cfg)
    num_classes = get_num_classes(loaded, state_dict, class_to_idx, cfg)

    logger.info(f"model_name = {model_name}")
    logger.info(f"num_classes = {num_classes}")
    logger.info(f"class_to_idx = {class_to_idx}")

    model = build_model(model_name, num_classes)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing:
        logger.warning(f"checkpoint missing keys: {len(missing)}")
    if unexpected:
        logger.warning(f"checkpoint unexpected keys: {len(unexpected)}")

    model.to(device)
    model.eval()

    return model, loaded, state_dict, model_name, num_classes, class_to_idx, idx_to_class


# =========================================================
# 数据集读取
# =========================================================

def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMG_EXTS


def list_samples_from_split(
    data_dir: Path,
    split: str,
    class_to_idx: Dict[str, int],
    limit: int = 0,
    seed: int = 42,
    balanced: bool = True
) -> List[Dict[str, Any]]:
    """
    从某个 split 中列出图片样本。

    关键修复：
    - 原版本在 limit>0 时先把所有路径排序再截断，Windows/中文路径排序会导致
      verify_samples=100 只覆盖部分类别，验证报告失真。
    - 新版本默认做分层均衡抽样，尽量让每个类别都参与验证。
    """
    split_dir = data_dir / split

    if not split_dir.exists():
        raise FileNotFoundError(f"找不到数据集 split 目录：{split_dir.resolve()}")

    rng = random.Random(int(seed))
    per_class: Dict[str, List[Dict[str, Any]]] = OrderedDict()

    for class_name, class_idx in sorted(class_to_idx.items(), key=lambda kv: kv[1]):
        class_dir = split_dir / class_name
        class_samples = []

        if class_dir.exists():
            image_paths = [
                p for p in class_dir.rglob("*")
                if is_image_file(p)
            ]
            image_paths.sort()

            for p in image_paths:
                class_samples.append({
                    "image_path": p,
                    "label": int(class_idx),
                    "class_name": class_name,
                    "split": split,
                })

        per_class[class_name] = class_samples

    all_samples = []
    for samples in per_class.values():
        all_samples.extend(samples)

    if not all_samples:
        raise ValueError(f"{split_dir.resolve()} 中没有找到有效图片。")

    if not limit or limit <= 0 or limit >= len(all_samples):
        return all_samples

    if not balanced:
        samples = list(all_samples)
        rng.shuffle(samples)
        return samples[:limit]

    # 分层均衡抽样：先给每类分配 base，再把余量分给仍有剩余样本的类别。
    class_items = list(per_class.items())
    n_classes = len(class_items)
    base = max(1, limit // max(n_classes, 1))
    selected: List[Dict[str, Any]] = []
    leftovers: List[Dict[str, Any]] = []

    for class_name, samples in class_items:
        shuffled = list(samples)
        rng.shuffle(shuffled)

        take = min(base, len(shuffled))
        selected.extend(shuffled[:take])
        leftovers.extend(shuffled[take:])

    if len(selected) < limit and leftovers:
        rng.shuffle(leftovers)
        selected.extend(leftovers[:limit - len(selected)])

    selected = selected[:limit]
    selected.sort(key=lambda x: (int(x["label"]), str(x["image_path"])))
    return selected


def get_calibration_samples(
    data_dir: Path,
    class_to_idx: Dict[str, int],
    cfg: Dict[str, Any],
    logger: logging.Logger
) -> List[Dict[str, Any]]:
    calib_split = str(cfg.get("calib_split", "calibration"))
    fallback_split = str(cfg.get("calib_fallback_split", "train"))
    limit = int(cfg.get("calib_limit", 500))

    if (data_dir / calib_split).exists():
        logger.info(f"使用 calibration split：{calib_split}")
        return list_samples_from_split(data_dir, calib_split, class_to_idx, limit=limit, seed=int(cfg.get("seed", 42)), balanced=True)

    logger.warning(f"没有找到 {calib_split}，改用 {fallback_split} 作为 INT8 校准集。")
    return list_samples_from_split(data_dir, fallback_split, class_to_idx, limit=limit, seed=int(cfg.get("seed", 42)), balanced=True)


def save_dataset_summary(
    run_dir: Path,
    test_samples: List[Dict[str, Any]],
    calib_samples: List[Dict[str, Any]],
    idx_to_class: Dict[int, str]
):
    rows = []

    for split_name, samples in [
        ("test", test_samples),
        ("calibration", calib_samples),
    ]:
        counter = Counter([s["label"] for s in samples])

        for idx in sorted(idx_to_class.keys()):
            rows.append({
                "split": split_name,
                "class_idx": idx,
                "class_name": idx_to_class[idx],
                "count": counter.get(idx, 0)
            })

    path = run_dir / "dataset_summary.csv"

    write_csv(
        path,
        rows,
        ["split", "class_idx", "class_name", "count"]
    )

    return path


# =========================================================
# 图像预处理
# =========================================================

def resize_keep_ratio(img: Image.Image, shorter_side: int) -> Image.Image:
    w, h = img.size

    if w <= 0 or h <= 0:
        raise ValueError("图片宽高异常。")

    if w < h:
        new_w = shorter_side
        new_h = int(round(h * shorter_side / w))
    else:
        new_h = shorter_side
        new_w = int(round(w * shorter_side / h))

    return img.resize((new_w, new_h), Image.BILINEAR)


def center_crop(img: Image.Image, crop_size: int) -> Image.Image:
    w, h = img.size

    left = max(0, int(round((w - crop_size) / 2)))
    top = max(0, int(round((h - crop_size) / 2)))
    right = left + crop_size
    bottom = top + crop_size

    return img.crop((left, top, right, bottom))


def load_image_as_pil(path: Path, cfg: Dict[str, Any]) -> Image.Image:
    img = Image.open(path)

    if cfg.get("rgb", True):
        img = img.convert("RGB")
    else:
        img = img.convert("L")

    return img


def pil_to_hwc_float32(img: Image.Image, cfg: Dict[str, Any]) -> np.ndarray:
    img_size = int(cfg.get("img_size", 224))
    resize_size = int(cfg.get("resize_size", 256))
    eval_preprocess = str(cfg.get("eval_preprocess", "resize_center_crop"))

    if eval_preprocess == "direct_resize":
        img = img.resize((img_size, img_size), Image.BILINEAR)
    else:
        img = resize_keep_ratio(img, resize_size)
        img = center_crop(img, img_size)

    if cfg.get("rgb", True):
        img = img.convert("RGB")
    else:
        img = img.convert("L")

    x = np.asarray(img, dtype=np.float32)

    if x.ndim == 2:
        x = np.expand_dims(x, axis=-1)

    x = x / 255.0

    if cfg.get("normalize", True):
        mean = np.asarray(cfg.get("mean", [0.485, 0.456, 0.406]), dtype=np.float32).reshape(1, 1, -1)
        std = np.asarray(cfg.get("std", [0.229, 0.224, 0.225]), dtype=np.float32).reshape(1, 1, -1)
        x = (x - mean) / std

    return x.astype(np.float32)


def hwc_to_nchw_batch(x_hwc: np.ndarray) -> np.ndarray:
    x_chw = np.transpose(x_hwc, (2, 0, 1))
    return np.expand_dims(x_chw, axis=0).astype(np.float32)


def hwc_to_nhwc_batch(x_hwc: np.ndarray) -> np.ndarray:
    return np.expand_dims(x_hwc, axis=0).astype(np.float32)


def preprocess_for_pytorch_onnx(img: Image.Image, cfg: Dict[str, Any]) -> np.ndarray:
    x_hwc = pil_to_hwc_float32(img, cfg)
    return hwc_to_nchw_batch(x_hwc)


def preprocess_for_layout(img: Image.Image, cfg: Dict[str, Any], layout: str) -> np.ndarray:
    x_hwc = pil_to_hwc_float32(img, cfg)

    layout = str(layout).upper()

    if layout == "NCHW":
        return hwc_to_nchw_batch(x_hwc)

    return hwc_to_nhwc_batch(x_hwc)


# =========================================================
# ONNX 导出与验证
# =========================================================

def export_onnx(
    model: nn.Module,
    onnx_path: Path,
    cfg: Dict[str, Any],
    device: torch.device,
    logger: logging.Logger
) -> Path:
    model.eval()

    img_size = int(cfg.get("img_size", 224))
    dummy = torch.randn(1, 3, img_size, img_size, dtype=torch.float32, device=device)

    input_name = str(cfg.get("input_name", "input"))
    output_name = str(cfg.get("output_name", "output"))

    dynamic_axes = None

    if cfg.get("onnx_dynamic_batch", True):
        dynamic_axes = {
            input_name: {0: "batch"},
            output_name: {0: "batch"},
        }

    logger.info(f"[ONNX] exporting: {onnx_path}")

    export_kwargs = dict(
        args=dummy,
        f=str(onnx_path),
        input_names=[input_name],
        output_names=[output_name],
        dynamic_axes=dynamic_axes,
        opset_version=int(cfg.get("opset", 13)),
        do_constant_folding=True,
    )

    try:
        with torch.no_grad():
            torch.onnx.export(
                model,
                **export_kwargs,
                dynamo=False
            )
    except TypeError:
        with torch.no_grad():
            torch.onnx.export(
                model,
                **export_kwargs
            )

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    logger.info("[ONNX] export and checker passed.")

    return onnx_path


def simplify_onnx_model(onnx_path: Path, logger: logging.Logger) -> Path:
    simplified_path = onnx_path.with_name(onnx_path.stem + "_simplified.onnx")

    logger.info("[ONNX-SIM] simplifying...")

    model = onnx.load(str(onnx_path))
    model_simplified, ok = simplify(model)

    if not ok:
        raise RuntimeError("onnxsim simplify failed.")

    onnx.save(model_simplified, str(simplified_path))
    onnx.checker.check_model(onnx.load(str(simplified_path)))

    logger.info(f"[ONNX-SIM] saved: {simplified_path}")

    return simplified_path


def create_ort_session(onnx_path: Path) -> ort.InferenceSession:
    return ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"]
    )


def run_onnx_session(sess: ort.InferenceSession, x: np.ndarray) -> np.ndarray:
    input_name = sess.get_inputs()[0].name
    y = sess.run(None, {input_name: x.astype(np.float32)})[0]
    return np.asarray(y)


def check_onnx_with_ort(onnx_path: Path, sample: Dict[str, Any], cfg: Dict[str, Any], logger: logging.Logger):
    sess = create_ort_session(onnx_path)

    img = load_image_as_pil(sample["image_path"], cfg)
    x = preprocess_for_pytorch_onnx(img, cfg)

    y = run_onnx_session(sess, x)

    logger.info(f"[ORT] ONNX quick check ok. output_shape={y.shape}, dtype={y.dtype}")


# =========================================================
# SavedModel 与 TFLite 转换
# =========================================================

def find_direct_tflite_files(folder: Path) -> List[Path]:
    """查找 onnx2tf 直接生成的 TFLite 文件。"""
    if not folder.exists():
        return []
    return sorted([p for p in folder.rglob("*.tflite") if p.is_file()])


def select_direct_tflite(files: List[Path], quant_type: str) -> Optional[Path]:
    """从 onnx2tf 直接输出中选择最匹配的 tflite。"""
    if not files:
        return None

    qt = str(quant_type).lower()
    scored = []

    for p in files:
        name = p.name.lower()
        score = 0

        if qt in ("fp16", "float16"):
            if "float16" in name or "fp16" in name:
                score += 100
            if "int8" in name or "integer" in name:
                score -= 50
        elif qt == "float32":
            if "float32" in name or "fp32" in name:
                score += 100
            if "float16" in name or "fp16" in name or "int8" in name or "integer" in name:
                score -= 50
        elif qt == "int8":
            if "int8" in name or "integer" in name:
                score += 100
            if "float16" in name or "fp16" in name:
                score -= 30

        # 更小的文件通常是量化版本；float32 通常更大。
        scored.append((score, p.stat().st_size, p))

    if qt == "float32":
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    else:
        scored.sort(key=lambda x: (x[0], -x[1]), reverse=True)

    best = scored[0]
    return best[2] if best[0] > -50 else None


def convert_onnx_to_saved_model(
    onnx_path: Path,
    saved_model_dir: Path,
    logger: logging.Logger
) -> Path:
    """
    ONNX -> SavedModel/TFLite。

    兼容修复：
    新版 onnx2tf 在某些配置下不会生成 saved_model.pb，而是直接写出 .tflite。
    旧脚本强制检查 saved_model.pb，因此会误报失败。这里不再把缺少
    saved_model.pb 视为立即失败；如果发现直接生成的 .tflite，后续流程会直接使用它。
    """
    try:
        import onnx2tf
    except Exception as e:
        raise ImportError(
            "缺少 onnx2tf，无法执行 ONNX -> SavedModel/TFLite。"
            "请先安装：pip install onnx2tf"
        ) from e

    if saved_model_dir.exists():
        shutil.rmtree(saved_model_dir)

    logger.info(f"[onnx2tf] converting ONNX -> SavedModel/TFLite: {saved_model_dir}")

    onnx2tf.convert(
        input_onnx_file_path=str(onnx_path),
        output_folder_path=str(saved_model_dir),
        copy_onnx_input_output_names_to_tflite=True,
        non_verbose=True,
        output_signaturedefs=True,
    )

    pb_path = saved_model_dir / "saved_model.pb"

    if pb_path.exists():
        logger.info("[onnx2tf] SavedModel export done.")
        return saved_model_dir

    direct_tflites = find_direct_tflite_files(saved_model_dir)

    if direct_tflites:
        logger.warning(
            "onnx2tf 未生成 saved_model.pb，但检测到直接生成的 TFLite 文件；"
            "将跳过 SavedModel -> TFLite 二次转换，直接使用这些文件。"
        )
        for p in direct_tflites:
            logger.info(f"[onnx2tf-direct] {p} ({file_size_mb(p):.3f} MB)")
        return saved_model_dir

    raise FileNotFoundError(
        f"onnx2tf 结束后既没有找到 saved_model.pb，也没有找到 .tflite：{saved_model_dir}"
    )


def tf_dtype_from_name(name: str):
    name = str(name).lower()

    if name == "int8":
        return tf.int8

    if name == "uint8":
        return tf.uint8

    return tf.float32


def get_saved_model_input_info(saved_model_dir: Path) -> Tuple[str, str]:
    loaded = tf.saved_model.load(str(saved_model_dir))

    if "serving_default" not in loaded.signatures:
        raise RuntimeError("SavedModel 中没有 serving_default signature。")

    fn = loaded.signatures["serving_default"]
    _, kw = fn.structured_input_signature

    if len(kw) != 1:
        raise RuntimeError(f"期望 SavedModel 只有一个输入，实际为：{list(kw.keys())}")

    input_name = list(kw.keys())[0]
    spec = kw[input_name]
    shape = list(spec.shape)

    layout = infer_layout_from_shape(shape)

    return input_name, layout


def infer_layout_from_shape(shape: List[int]) -> str:
    if len(shape) != 4:
        return "NHWC"

    # [N, C, H, W]
    if shape[1] in (1, 3):
        return "NCHW"

    # [N, H, W, C]
    if shape[-1] in (1, 3):
        return "NHWC"

    return "NHWC"


def make_representative_dataset(
    input_name: str,
    layout: str,
    calib_samples: List[Dict[str, Any]],
    cfg: Dict[str, Any]
):
    def representative_dataset():
        for sample in calib_samples:
            img = load_image_as_pil(sample["image_path"], cfg)
            x = preprocess_for_layout(img, cfg, layout).astype(np.float32)
            yield {input_name: x}

    return representative_dataset


def convert_saved_model_to_tflite(
    saved_model_dir: Path,
    out_path: Path,
    quant_type: str,
    cfg: Dict[str, Any],
    calib_samples: List[Dict[str, Any]],
    logger: logging.Logger
) -> Path:
    quant_type = str(quant_type).lower()

    input_name, layout = get_saved_model_input_info(saved_model_dir)

    logger.info(f"[TFLite] converting quant={quant_type}, savedmodel_input={input_name}, layout={layout}")

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))

    if quant_type == "float32":
        pass

    elif quant_type == "fp16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

    elif quant_type == "int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.representative_dataset = make_representative_dataset(
            input_name=input_name,
            layout=layout,
            calib_samples=calib_samples,
            cfg=cfg
        )

        converter.inference_input_type = tf_dtype_from_name(cfg.get("int8_input_type", "float32"))
        converter.inference_output_type = tf_dtype_from_name(cfg.get("int8_output_type", "float32"))

    else:
        raise ValueError("quant_type 只能是 float32 / fp16 / int8。")

    tflite_model = converter.convert()

    with open(out_path, "wb") as f:
        f.write(tflite_model)

    logger.info(f"[TFLite] saved: {out_path}")

    return out_path


# =========================================================
# 推理函数
# =========================================================

def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - np.max(x, axis=-1, keepdims=True)
    exp = np.exp(x)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def run_pytorch(model: nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        xt = torch.from_numpy(x).to(device=device, dtype=torch.float32)
        y = model(xt)

        if isinstance(y, (tuple, list)):
            y = y[0]

        if isinstance(y, dict):
            y = next(iter(y.values()))

        return y.detach().cpu().numpy()


def load_tflite_interpreter(tflite_path: Path) -> tf.lite.Interpreter:
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    return interpreter


def quantize_tensor_if_needed(x_float32: np.ndarray, input_detail: Dict[str, Any]) -> np.ndarray:
    dtype = input_detail["dtype"]
    scale, zero_point = input_detail.get("quantization", (0.0, 0))

    if dtype == np.float32 or not scale:
        return x_float32.astype(np.float32)

    q = np.round(x_float32 / float(scale) + int(zero_point))

    if dtype == np.int8:
        return np.clip(q, -128, 127).astype(np.int8)

    if dtype == np.uint8:
        return np.clip(q, 0, 255).astype(np.uint8)

    return q.astype(dtype)


def dequantize_tensor_if_needed(y: np.ndarray, output_detail: Dict[str, Any]) -> np.ndarray:
    dtype = output_detail["dtype"]
    scale, zero_point = output_detail.get("quantization", (0.0, 0))

    if dtype == np.float32 or not scale:
        return y.astype(np.float32)

    return (y.astype(np.float32) - int(zero_point)) * float(scale)


def run_tflite_interpreter(
    interpreter: tf.lite.Interpreter,
    x_float32: np.ndarray
) -> np.ndarray:
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]

    x_feed = quantize_tensor_if_needed(x_float32, input_detail)

    interpreter.set_tensor(input_detail["index"], x_feed)
    interpreter.invoke()

    y = interpreter.get_tensor(output_detail["index"])
    y = dequantize_tensor_if_needed(y, output_detail)

    return y.astype(np.float32)


def get_tflite_input_layout(interpreter: tf.lite.Interpreter) -> str:
    input_detail = interpreter.get_input_details()[0]
    shape = list(input_detail["shape"])
    return infer_layout_from_shape(shape)


# =========================================================
# 评估与报告
# =========================================================

def predict_from_logits(logits: np.ndarray) -> Tuple[int, float]:
    logits = np.asarray(logits).reshape(1, -1)
    probs = softmax_np(logits)[0]
    pred = int(np.argmax(probs))
    conf = float(np.max(probs))
    return pred, conf


def evaluate_pytorch(
    model: nn.Module,
    samples: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    device: torch.device,
    idx_to_class: Dict[int, str],
    logger: logging.Logger
):
    records = []
    y_true = []
    y_pred = []

    start_all = time.perf_counter()

    for i, sample in enumerate(samples):
        img = load_image_as_pil(sample["image_path"], cfg)
        x = preprocess_for_pytorch_onnx(img, cfg)

        t0 = time.perf_counter()
        logits = run_pytorch(model, x, device)
        latency_ms = (time.perf_counter() - t0) * 1000

        pred, conf = predict_from_logits(logits)
        label = int(sample["label"])

        y_true.append(label)
        y_pred.append(pred)

        records.append({
            "index": i,
            "backend": "pytorch",
            "image_path": str(sample["image_path"]),
            "true_idx": label,
            "true_class": idx_to_class[label],
            "pred_idx": pred,
            "pred_class": idx_to_class[pred],
            "confidence": conf,
            "correct": int(pred == label),
            "latency_ms": latency_ms,
        })

    total_time = time.perf_counter() - start_all

    logger.info(f"[Eval] PyTorch done. samples={len(samples)}, time={total_time:.2f}s")

    return y_true, y_pred, records


def evaluate_onnx(
    onnx_path: Path,
    samples: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    idx_to_class: Dict[int, str],
    logger: logging.Logger
):
    sess = create_ort_session(onnx_path)

    records = []
    y_true = []
    y_pred = []

    start_all = time.perf_counter()

    for i, sample in enumerate(samples):
        img = load_image_as_pil(sample["image_path"], cfg)
        x = preprocess_for_pytorch_onnx(img, cfg)

        t0 = time.perf_counter()
        logits = run_onnx_session(sess, x)
        latency_ms = (time.perf_counter() - t0) * 1000

        pred, conf = predict_from_logits(logits)
        label = int(sample["label"])

        y_true.append(label)
        y_pred.append(pred)

        records.append({
            "index": i,
            "backend": "onnx",
            "image_path": str(sample["image_path"]),
            "true_idx": label,
            "true_class": idx_to_class[label],
            "pred_idx": pred,
            "pred_class": idx_to_class[pred],
            "confidence": conf,
            "correct": int(pred == label),
            "latency_ms": latency_ms,
        })

    total_time = time.perf_counter() - start_all

    logger.info(f"[Eval] ONNX done. samples={len(samples)}, time={total_time:.2f}s")

    return y_true, y_pred, records


def evaluate_tflite(
    tflite_path: Path,
    backend_name: str,
    samples: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    idx_to_class: Dict[int, str],
    logger: logging.Logger
):
    interpreter = load_tflite_interpreter(tflite_path)
    layout = get_tflite_input_layout(interpreter)

    logger.info(f"[Eval] {backend_name} input layout = {layout}")

    records = []
    y_true = []
    y_pred = []

    start_all = time.perf_counter()

    for i, sample in enumerate(samples):
        img = load_image_as_pil(sample["image_path"], cfg)
        x = preprocess_for_layout(img, cfg, layout)

        t0 = time.perf_counter()
        logits = run_tflite_interpreter(interpreter, x)
        latency_ms = (time.perf_counter() - t0) * 1000

        pred, conf = predict_from_logits(logits)
        label = int(sample["label"])

        y_true.append(label)
        y_pred.append(pred)

        records.append({
            "index": i,
            "backend": backend_name,
            "image_path": str(sample["image_path"]),
            "true_idx": label,
            "true_class": idx_to_class[label],
            "pred_idx": pred,
            "pred_class": idx_to_class[pred],
            "confidence": conf,
            "correct": int(pred == label),
            "latency_ms": latency_ms,
        })

    total_time = time.perf_counter() - start_all

    logger.info(f"[Eval] {backend_name} done. samples={len(samples)}, time={total_time:.2f}s")

    return y_true, y_pred, records


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


def save_confusion_matrix_csv(path: Path, cm: np.ndarray, class_names: List[str]):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["真实\\预测"] + class_names)

        for i, row in enumerate(cm):
            writer.writerow([class_names[i]] + row.tolist())


def save_confusion_matrix_png(path: Path, cm: np.ndarray, class_names: List[str], title: str):
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


def save_backend_report(
    run_dir: Path,
    backend_name: str,
    y_true: List[int],
    y_pred: List[int],
    records: List[Dict[str, Any]],
    idx_to_class: Dict[int, str],
    cfg: Dict[str, Any]
):
    class_names = [idx_to_class[i] for i in sorted(idx_to_class.keys())]
    labels = sorted(idx_to_class.keys())

    acc = float(accuracy_score(y_true, y_pred))
    avg_latency = float(np.mean([r["latency_ms"] for r in records])) if records else 0.0

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

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    pred_csv = run_dir / f"{backend_name}_per_image_predictions.csv"
    report_txt = run_dir / f"{backend_name}_classification_report.txt"
    report_json = run_dir / f"{backend_name}_classification_report.json"
    cm_csv = run_dir / f"{backend_name}_confusion_matrix.csv"
    cm_png = run_dir / f"{backend_name}_confusion_matrix.png"
    metrics_json = run_dir / f"{backend_name}_metrics.json"

    if cfg.get("save_per_image_predictions", True):
        write_csv(
            pred_csv,
            records,
            [
                "index",
                "backend",
                "image_path",
                "true_idx",
                "true_class",
                "pred_idx",
                "pred_class",
                "confidence",
                "correct",
                "latency_ms",
            ]
        )

    with open(report_txt, "w", encoding="utf-8") as f:
        f.write(report_text)

    write_json(report_json, report_dict)

    save_confusion_matrix_csv(cm_csv, cm, class_names)

    if cfg.get("save_confusion_matrix_png", True):
        save_confusion_matrix_png(cm_png, cm, class_names, f"{backend_name} Confusion Matrix")

    metrics = {
        "backend": backend_name,
        "accuracy": acc,
        "avg_latency_ms": avg_latency,
        "num_samples": len(y_true),
        "classification_report_txt": str(report_txt.resolve()),
        "classification_report_json": str(report_json.resolve()),
        "confusion_matrix_csv": str(cm_csv.resolve()),
        "confusion_matrix_png": str(cm_png.resolve()),
        "per_image_predictions_csv": str(pred_csv.resolve()) if cfg.get("save_per_image_predictions", True) else None,
    }

    write_json(metrics_json, metrics)

    return metrics, report_text


def compare_prediction_consistency(
    run_dir: Path,
    all_prediction_records: Dict[str, List[Dict[str, Any]]]
):
    backends = list(all_prediction_records.keys())

    if len(backends) < 2:
        return None

    base_backend = backends[0]
    base_records = all_prediction_records[base_backend]

    rows = []

    for i in range(len(base_records)):
        row = {
            "index": i,
            "image_path": base_records[i]["image_path"],
            "true_class": base_records[i]["true_class"],
        }

        preds = {}

        for backend in backends:
            rec = all_prediction_records[backend][i]
            preds[backend] = rec["pred_class"]
            row[f"{backend}_pred"] = rec["pred_class"]
            row[f"{backend}_conf"] = rec["confidence"]
            row[f"{backend}_correct"] = rec["correct"]

        row["all_same_pred"] = int(len(set(preds.values())) == 1)

        rows.append(row)

    path = run_dir / "backend_prediction_consistency.csv"

    fieldnames = ["index", "image_path", "true_class"]

    for backend in backends:
        fieldnames.extend([
            f"{backend}_pred",
            f"{backend}_conf",
            f"{backend}_correct",
        ])

    fieldnames.append("all_same_pred")

    write_csv(path, rows, fieldnames)

    return path


def get_class_mapping_from_config_file(config_path: Path, logger: logging.Logger) -> Tuple[Dict[str, int], Dict[int, str]]:
    """从外部 class_mapping JSON 文件加载类别映射。"""
    if not config_path.exists():
        raise FileNotFoundError(f"class config 文件不存在：{config_path.resolve()}")

    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    class_to_idx = data.get("class_to_idx", {})
    if not class_to_idx:
        raise ValueError(f"class config 文件中没有 class_to_idx：{config_path}")

    class_to_idx = {str(k): int(v) for k, v in class_to_idx.items()}
    idx_to_class = {int(v): str(k) for k, v in class_to_idx.items()}

    logger.info(f"从外部配置加载类别映射：{class_to_idx}")
    return class_to_idx, idx_to_class


def dry_run_checks(
    cfg: Dict[str, Any],
    data_dir: Path,
    ckpt_path: Path,
    logger: logging.Logger
) -> bool:
    """Dry-run：检查所有预条件，不实际导出。"""
    checks = []

    logger.info("=" * 60)
    logger.info("DRY-RUN 预检开始")
    logger.info("=" * 60)

    check_ok = ckpt_path.exists()
    checks.append(("PyTorch checkpoint 存在", check_ok, str(ckpt_path.resolve())))
    logger.info(f"[{'OK' if check_ok else 'FAIL'}] checkpoint: {ckpt_path.resolve()}")

    try:
        device = torch.device("cpu")
        loaded = safe_torch_load(ckpt_path, device)
        checks.append(("checkpoint 可加载", True, ""))
        logger.info("[OK] checkpoint 可加载")
    except Exception as e:
        checks.append(("checkpoint 可加载", False, str(e)))
        _print_checks(checks)
        return False

    try:
        state_dict = strip_module_prefix(extract_state_dict(loaded))
        checks.append(("state_dict 可提取", True, f"{len(state_dict)} keys"))
        logger.info(f"[OK] state_dict: {len(state_dict)} keys")
    except Exception as e:
        checks.append(("state_dict 可提取", False, str(e)))
        _print_checks(checks)
        return False

    num_classes_ok = False
    class_config_path_str = cfg.get("class_config_path", "")
    try:
        if class_config_path_str:
            config_file = Path(class_config_path_str)
            if not config_file.is_absolute():
                config_file = PROJECT_ROOT / config_file
            c2i, i2c = get_class_mapping_from_config_file(config_file, logger)
        else:
            c2i, i2c = get_class_mapping_from_checkpoint(loaded, data_dir, cfg, logger)
        num_classes = len(c2i)
        num_classes_ok = num_classes == 5
        checks.append(("num_classes = 5", num_classes_ok, f"实际: {num_classes}"))
        logger.info(f"[{'OK' if num_classes_ok else 'FAIL'}] num_classes = {num_classes}")
    except Exception as e:
        checks.append(("num_classes = 5", False, str(e)))
        c2i, i2c = {}, {}

    try:
        if num_classes_ok:
            model_name = get_model_name_from_checkpoint(loaded, cfg)
            model = build_model(model_name, num_classes)
            model.load_state_dict(state_dict, strict=False)
            out_features = model.classifier[3].out_features
            out_ok = out_features == num_classes
            checks.append(("输出层 = num_classes", out_ok, f"实际: {out_features}"))
            logger.info(f"[{'OK' if out_ok else 'FAIL'}] classifier[3].out_features = {out_features}")
    except Exception as e:
        checks.append(("输出层 = num_classes", False, str(e)))

    try:
        expected_order = ["待分拣", "其他", "厨余", "可回收", "有害"]
        actual_order = [i2c.get(i, "?") for i in range(min(num_classes, len(i2c)))]
        mapping_ok = actual_order[:num_classes] == expected_order[:num_classes]
        checks.append(("类别顺序正确", mapping_ok, f"实际: {actual_order}"))
        logger.info(f"[{'OK' if mapping_ok else 'FAIL'}] 类别顺序: {actual_order}")
    except Exception as e:
        checks.append(("类别顺序正确", False, str(e)))

    try:
        out_root = Path(cfg["outdir"])
        out_root.mkdir(parents=True, exist_ok=True)
        checks.append(("output-dir 可创建", True, str(out_root.resolve())))
    except Exception as e:
        checks.append(("output-dir 可创建", False, str(e)))

    img_size = int(cfg.get("img_size", 224))
    checks.append(("image_size = 224", True, str(img_size)))

    dep_list = check_export_dependencies(logger)
    dep_ok = len(dep_list) == 0
    checks.append(("导出依赖齐全", dep_ok, f"缺失: {dep_list}" if dep_list else "全部就绪"))

    _print_checks(checks)
    return all(ok for _, ok, _ in checks)


def _print_checks(checks):
    print()
    print("=" * 60)
    print("DRY-RUN 预检结果")
    print("=" * 60)
    for name, ok, detail in checks:
        icon = "OK" if ok else "FAIL"
        line = f"  [{icon}] {name}"
        if detail:
            line += f": {detail}"
        print(line)
    n_ok = sum(1 for _, ok, _ in checks if ok)
    n_total = len(checks)
    print(f"\n结果: {n_ok}/{n_total} 通过")
    print("=" * 60)


# =========================================================
# 主程序
# =========================================================

def main():
    warnings.filterwarnings("ignore", category=UserWarning)

    # ── 命令行参数解析 ──
    parser = argparse.ArgumentParser(
        description="MobileNetV3 垃圾分类五分类模型导出 — PyTorch → ONNX → TFLite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例:
  # Dry-run 预检
  python model_export_tflite.py --model-path models/vision_trigger_5class_mobilenetv3/latest_mobilenetv3_best.pt --output-dir models/vision_trigger_5class_tflite --class-config 09_Vision_Trigger_5Class_System/config/class_mapping_5class.json --data-dir garbage_dataset --num-classes auto --dry-run

  # 实际导出 + float16 + 验证
  python model_export_tflite.py --model-path models/vision_trigger_5class_mobilenetv3/latest_mobilenetv3_best.pt --output-dir models/vision_trigger_5class_tflite --class-config 09_Vision_Trigger_5Class_System/config/class_mapping_5class.json --data-dir garbage_dataset --num-classes auto --quantize-float16 --verify --verify-samples 100
"""
    )
    parser.add_argument("--model-path", type=str, default=None, help="PyTorch checkpoint .pt 文件路径")
    parser.add_argument("--output-dir", type=str, default=None, help="量化导出根目录")
    parser.add_argument("--class-config", type=str, default=None, help="外部 class_mapping JSON 文件路径")
    parser.add_argument("--data-dir", type=str, default=None, help="数据集目录 (train/val/test)")
    parser.add_argument("--model-name", type=str, default="auto", choices=["auto", "mobilenet_v3_small", "mobilenet_v3_large"], help="模型名称 (默认 auto)")
    parser.add_argument("--num-classes", type=str, default="auto", help="类别数 (默认 auto 从 checkpoint/class config 推断)")
    parser.add_argument("--image-size", type=int, default=224, help="输入图像尺寸 (默认 224)")
    parser.add_argument("--batch-size", type=int, default=1, help="推理批大小 (默认 1)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="device (默认 auto)")
    parser.add_argument("--quantize-float16", action="store_true", default=False, help="导出 float16 TFLite")
    parser.add_argument("--quantize-int8", action="store_true", default=False, help="导出 INT8 TFLite（默认关闭；需要校准集）")
    parser.add_argument("--verify", action="store_true", default=False, help="导出后验证 PyTorch vs TFLite top1 一致率")
    parser.add_argument("--verify-samples", type=int, default=0, help="验证样本数 (默认 0 表示全量 test)")
    parser.add_argument("--dry-run", action="store_true", default=False, help="预检模式：只检查不导出")

    args = parser.parse_args()
    cfg = merge_cli_to_config(args)

    # ── 路径解析 ──
    cfg["data_dir"] = str((PROJECT_ROOT / cfg["data_dir"]).resolve())
    cfg["ckpt"] = str((PROJECT_ROOT / cfg["ckpt"]).resolve())
    cfg["outdir"] = str((PROJECT_ROOT / cfg["outdir"]).resolve())
    set_seed(int(cfg.get("seed", 42)))

    data_dir = Path(cfg["data_dir"])
    ckpt_path = Path(cfg["ckpt"])

    # ── Dry-run 模式：只检查不导出 ──
    if cfg.get("run_dry_run", False):
        logger_temp = logging.getLogger("dry_run")
        logger_temp.setLevel(logging.INFO)
        logger_temp.handlers.clear()
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter(fmt="%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        logger_temp.addHandler(ch)

        ok = dry_run_checks(cfg, data_dir, ckpt_path, logger_temp)
        sys.exit(0 if ok else 1)

    # 延迟导入可选依赖 — sklearn/matplotlib 仅导出时需要
    global classification_report, confusion_matrix, accuracy_score, plt
    try:
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        pass  # sklearn/matplotlib 可选，不影响核心导出流程

    # 触发 ONNX/TF 惰性导入
    _lazy_import_onnx()
    _lazy_import_ort()
    _lazy_import_onnxsim()
    _lazy_import_tf()

    # ── 正常导出模式 ──
    if not data_dir.exists():
        raise FileNotFoundError(f"数据集目录不存在：{data_dir.resolve()}")

    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint 不存在：{ckpt_path.resolve()}")

    # ── 导出依赖检查 ──
    run_dir = prepare_run_dir(cfg)
    logger = setup_logger(run_dir)
    missing_deps = check_export_dependencies(logger)
    if missing_deps:
        logger.error(f"缺少导出依赖: {', '.join(missing_deps)}")
        logger.error("")
        logger.error("请安装以下依赖后重试:")
        logger.error(f"  pip install {' '.join(missing_deps)}")
        logger.error("")
        logger.error("如果 tensorflow 报 numpy 兼容性错误，请先降级 numpy:")
        logger.error("  pip install \"numpy<2\"")
        logger.error("")
        logger.error("或者使用以下完整安装命令:")
        logger.error("  pip install onnx onnxruntime onnxsim tensorflow \"numpy<2\"")
        logger.error("")
        logger.info(f"已生成空运行目录: {run_dir}")
        # 写入依赖缺失报告
        deps_report = {
            "status": "blocked",
            "missing_dependencies": missing_deps,
            "fix_commands": [
                "pip install onnx onnxruntime onnxsim",
                "pip install \"numpy<2\"",
                "pip install tensorflow",
            ],
        }
        write_json(run_dir / "export_summary.json", deps_report)
        logger.info(f"依赖报告: {run_dir / 'export_summary.json'}")
        sys.exit(1)

    device = get_device(str(cfg.get("device", "cpu")))

    logger.info("========== MobileNetV3 垃圾分类模型量化导出开始 ==========")
    logger.info(f"data_dir = {data_dir.resolve()}")
    logger.info(f"ckpt = {ckpt_path.resolve()}")
    logger.info(f"run_dir = {run_dir.resolve()}")
    logger.info(f"device = {device}")

    config_path = run_dir / "config.json"
    write_json(config_path, cfg)

    model, loaded_ckpt, state_dict, model_name, num_classes, class_to_idx, idx_to_class = load_model_from_checkpoint(
        cfg=cfg,
        device=device,
        data_dir=data_dir,
        logger=logger
    )

    class_mapping_path = run_dir / "class_mapping.json"
    write_json(
        class_mapping_path,
        {
            "class_to_idx": class_to_idx,
            "idx_to_class": {str(k): v for k, v in idx_to_class.items()},
            "model_name": model_name,
            "num_classes": num_classes,
        }
    )

    logger.info(f"class mapping saved: {class_mapping_path}")

    # ── 外部类别配置覆盖 ──
    class_config_path_str = cfg.get("class_config_path", "")
    if class_config_path_str:
        config_file = Path(class_config_path_str)
        if not config_file.is_absolute():
            config_file = PROJECT_ROOT / config_file
        if config_file.exists():
            external_c2i, external_i2c = get_class_mapping_from_config_file(config_file, logger)
            logger.info(f"外部类别配置加载: {external_c2i}")
            class_to_idx = external_c2i
            idx_to_class = external_i2c
            num_classes = len(class_to_idx)
            logger.info(f"已覆盖类别映射，num_classes = {num_classes}")

    # ── 类别顺序验证 ──
    expected_order = ["待分拣", "其他", "厨余", "可回收", "有害"]
    actual_order = [idx_to_class.get(i, "?") for i in range(min(num_classes, len(idx_to_class)))]
    if actual_order[:5] == expected_order:
        logger.info(f"✓ 类别顺序正确: {actual_order}")
    else:
        logger.warning(f"⚠ 类别顺序不匹配！期望: {expected_order}, 实际: {actual_order}")

    # ── class_names.json ──
    class_names_path = run_dir / "class_names.json"
    write_json(class_names_path, [idx_to_class[i] for i in range(num_classes)])

    # ── export_config.json ──
    export_config_path = run_dir / "export_config.json"
    write_json(export_config_path, {
        "model_type": model_name,
        "num_classes": num_classes,
        "class_names": [idx_to_class[i] for i in range(num_classes)],
        "image_size": int(cfg.get("img_size", 224)),
        "normalize": {"mean": cfg["mean"], "std": cfg["std"]},
        "pytorch_ckpt": str(ckpt_path.resolve()),
    })

    test_split = str(cfg.get("test_split", "test"))
    verify_limit = int(cfg.get("verify_limit", 0))

    test_samples = list_samples_from_split(
        data_dir,
        test_split,
        class_to_idx,
        limit=verify_limit,
        seed=int(cfg.get("seed", 42)),
        balanced=True
    )

    calib_samples = get_calibration_samples(
        data_dir,
        class_to_idx,
        cfg,
        logger
    )

    dataset_summary_path = save_dataset_summary(
        run_dir,
        test_samples,
        calib_samples,
        idx_to_class
    )

    logger.info(f"test samples = {len(test_samples)}")
    logger.info(f"calibration samples = {len(calib_samples)}")
    logger.info(f"dataset summary saved: {dataset_summary_path}")

    # =====================================================
    # 1. PyTorch FP32 评估
    # =====================================================

    all_metrics = []
    all_prediction_records = {}

    y_true_pt, y_pred_pt, records_pt = evaluate_pytorch(
        model=model,
        samples=test_samples,
        cfg=cfg,
        device=device,
        idx_to_class=idx_to_class,
        logger=logger
    )

    metrics_pt, report_pt = save_backend_report(
        run_dir,
        "pytorch_fp32",
        y_true_pt,
        y_pred_pt,
        records_pt,
        idx_to_class,
        cfg
    )

    all_metrics.append(metrics_pt)
    all_prediction_records["pytorch_fp32"] = records_pt

    logger.info("========== PyTorch FP32 Report ==========")
    logger.info("\n" + report_pt)

    # =====================================================
    # 2. 导出 ONNX
    # =====================================================

    onnx_path = run_dir / "model_float32.onnx"

    export_onnx(
        model=model,
        onnx_path=onnx_path,
        cfg=cfg,
        device=device,
        logger=logger
    )

    final_onnx_path = onnx_path

    if cfg.get("onnx_simplify", True):
        try:
            final_onnx_path = simplify_onnx_model(onnx_path, logger)
        except Exception as e:
            logger.warning(f"ONNX simplify 失败，将继续使用原始 ONNX：{repr(e)}")
            final_onnx_path = onnx_path

    if cfg.get("check_onnx_with_ort", True):
        check_onnx_with_ort(
            final_onnx_path,
            test_samples[0],
            cfg,
            logger
        )

    # =====================================================
    # 3. ONNX Runtime 评估
    # =====================================================

    y_true_onnx, y_pred_onnx, records_onnx = evaluate_onnx(
        onnx_path=final_onnx_path,
        samples=test_samples,
        cfg=cfg,
        idx_to_class=idx_to_class,
        logger=logger
    )

    metrics_onnx, report_onnx = save_backend_report(
        run_dir,
        "onnx_fp32",
        y_true_onnx,
        y_pred_onnx,
        records_onnx,
        idx_to_class,
        cfg
    )

    all_metrics.append(metrics_onnx)
    all_prediction_records["onnx_fp32"] = records_onnx

    logger.info("========== ONNX FP32 Report ==========")
    logger.info("\n" + report_onnx)

    # =====================================================
    # 4. ONNX -> SavedModel
    # =====================================================

    saved_model_dir = run_dir / "saved_model"

    convert_onnx_to_saved_model(
        onnx_path=final_onnx_path,
        saved_model_dir=saved_model_dir,
        logger=logger
    )

    # =====================================================
    # 5. TFLite 导出与评估
    # =====================================================

    tflite_paths = {}
    saved_model_pb = saved_model_dir / "saved_model.pb"

    if saved_model_pb.exists():
        if cfg.get("export_tflite_float32", True):
            path = run_dir / "garbage_mobilenetv3_5class_float32.tflite"
            tflite_paths["tflite_float32"] = convert_saved_model_to_tflite(
                saved_model_dir=saved_model_dir,
                out_path=path,
                quant_type="float32",
                cfg=cfg,
                calib_samples=calib_samples,
                logger=logger
            )

        if cfg.get("export_tflite_fp16", True):
            path = run_dir / "garbage_mobilenetv3_5class_float16.tflite"
            tflite_paths["tflite_fp16"] = convert_saved_model_to_tflite(
                saved_model_dir=saved_model_dir,
                out_path=path,
                quant_type="fp16",
                cfg=cfg,
                calib_samples=calib_samples,
                logger=logger
            )

        if cfg.get("export_tflite_int8", False):
            path = run_dir / "model_int8.tflite"
            tflite_paths["tflite_int8"] = convert_saved_model_to_tflite(
                saved_model_dir=saved_model_dir,
                out_path=path,
                quant_type="int8",
                cfg=cfg,
                calib_samples=calib_samples,
                logger=logger
            )
    else:
        direct_tflites = find_direct_tflite_files(saved_model_dir)
        logger.info("[TFLite] SavedModel 不存在，使用 onnx2tf 直接生成的 TFLite 文件。")

        if cfg.get("export_tflite_float32", True):
            src = select_direct_tflite(direct_tflites, "float32")
            if src is not None:
                dst = run_dir / "garbage_mobilenetv3_5class_float32.tflite"
                if src.resolve() != dst.resolve():
                    shutil.copy2(src, dst)
                tflite_paths["tflite_float32"] = dst
                logger.info(f"[TFLite-direct] float32 -> {dst}")

        if cfg.get("export_tflite_fp16", True):
            src = select_direct_tflite(direct_tflites, "float16")
            if src is not None:
                dst = run_dir / "garbage_mobilenetv3_5class_float16.tflite"
                if src.resolve() != dst.resolve():
                    shutil.copy2(src, dst)
                tflite_paths["tflite_fp16"] = dst
                logger.info(f"[TFLite-direct] float16 -> {dst}")

        if cfg.get("export_tflite_int8", False):
            src = select_direct_tflite(direct_tflites, "int8")
            if src is not None:
                dst = run_dir / "model_int8.tflite"
                if src.resolve() != dst.resolve():
                    shutil.copy2(src, dst)
                tflite_paths["tflite_int8"] = dst
                logger.info(f"[TFLite-direct] int8 -> {dst}")

        if not tflite_paths:
            raise RuntimeError(
                "onnx2tf 没有生成可用的 TFLite 文件，无法继续评估。"
            )

    failed_tflite_backends = {}

    for backend_name, tflite_path in tflite_paths.items():
        try:
            y_true_tf, y_pred_tf, records_tf = evaluate_tflite(
                tflite_path=tflite_path,
                backend_name=backend_name,
                samples=test_samples,
                cfg=cfg,
                idx_to_class=idx_to_class,
                logger=logger
            )

            metrics_tf, report_tf = save_backend_report(
                run_dir,
                backend_name,
                y_true_tf,
                y_pred_tf,
                records_tf,
                idx_to_class,
                cfg
            )

            metrics_tf["status"] = "ok"
            all_metrics.append(metrics_tf)
            all_prediction_records[backend_name] = records_tf

            logger.info(f"========== {backend_name} Report ==========")
            logger.info("\n" + report_tf)

        except Exception as e:
            error_text = repr(e)
            logger.error(f"[Eval] {backend_name} 评估失败，但不会中断整个导出流程。")
            logger.error(f"[Eval] {backend_name} error: {error_text}")
            logger.error("[Eval] 如果这是 float16 TFLite，在普通 CPU Interpreter 上不支持部分 FP16 CONV_2D 是常见情况；优先使用已通过验证的 float32 TFLite。")

            failed_tflite_backends[backend_name] = {
                "path": str(Path(tflite_path).resolve()),
                "error": error_text,
                "status": "eval_failed",
            }

            all_metrics.append({
                "backend": backend_name,
                "status": "eval_failed",
                "accuracy": None,
                "avg_latency_ms": None,
                "num_samples": len(test_samples),
                "error": error_text,
                "classification_report_txt": None,
                "confusion_matrix_csv": None,
                "confusion_matrix_png": None,
                "per_image_predictions_csv": None,
            })
            continue

    # =====================================================
    # 6. 后处理：体积、指标汇总、一致性分析
    # =====================================================

    model_size_rows = [
        {
            "name": "onnx_original",
            "path": str(onnx_path.resolve()),
            "size_mb": file_size_mb(onnx_path)
        },
        {
            "name": "onnx_final",
            "path": str(final_onnx_path.resolve()),
            "size_mb": file_size_mb(final_onnx_path)
        },
    ]

    for backend_name, path in tflite_paths.items():
        model_size_rows.append({
            "name": backend_name,
            "path": str(path.resolve()),
            "size_mb": file_size_mb(path)
        })

    size_report_path = run_dir / "model_size_report.csv"

    write_csv(
        size_report_path,
        model_size_rows,
        ["name", "path", "size_mb"]
    )

    metrics_rows = []

    baseline_acc = metrics_pt["accuracy"]

    for item in all_metrics:
        acc = item.get("accuracy")
        acc_drop = None if acc is None else baseline_acc - acc

        metrics_rows.append({
            "backend": item.get("backend"),
            "status": item.get("status", "ok"),
            "accuracy": acc,
            "acc_drop_vs_pytorch_fp32": acc_drop,
            "avg_latency_ms": item.get("avg_latency_ms"),
            "num_samples": item.get("num_samples"),
            "error": item.get("error", ""),
            "classification_report_txt": item.get("classification_report_txt"),
            "confusion_matrix_csv": item.get("confusion_matrix_csv"),
            "confusion_matrix_png": item.get("confusion_matrix_png"),
            "per_image_predictions_csv": item.get("per_image_predictions_csv"),
        })

    metrics_summary_path = run_dir / "metrics_summary.csv"

    write_csv(
        metrics_summary_path,
        metrics_rows,
        [
            "backend",
            "status",
            "accuracy",
            "acc_drop_vs_pytorch_fp32",
            "avg_latency_ms",
            "num_samples",
            "error",
            "classification_report_txt",
            "confusion_matrix_csv",
            "confusion_matrix_png",
            "per_image_predictions_csv",
        ]
    )

    consistency_path = compare_prediction_consistency(
        run_dir,
        all_prediction_records
    )

    # 复制一份最常用模型到 outdir 根目录，方便部署
    out_root = Path(cfg["outdir"])
    out_root.mkdir(parents=True, exist_ok=True)

    latest_files = {}

    for backend_name, path in tflite_paths.items():
        latest_path = out_root / f"latest_{backend_name}.tflite"
        shutil.copy2(path, latest_path)
        latest_files[backend_name] = str(latest_path.resolve())

    latest_onnx = out_root / "latest_model.onnx"
    shutil.copy2(final_onnx_path, latest_onnx)
    latest_files["onnx"] = str(latest_onnx.resolve())

    summary = {
        "created_time": now_str(),
        "data_dir": str(data_dir.resolve()),
        "ckpt": str(ckpt_path.resolve()),
        "run_dir": str(run_dir.resolve()),
        "model_name": model_name,
        "num_classes": num_classes,
        "class_to_idx": class_to_idx,
        "idx_to_class": {str(k): v for k, v in idx_to_class.items()},
        "test_samples": len(test_samples),
        "calibration_samples": len(calib_samples),
        "baseline_pytorch_fp32_acc": baseline_acc,
        "metrics": metrics_rows,
        "failed_tflite_backends": failed_tflite_backends,
        "files": {
            "config": str(config_path.resolve()),
            "class_mapping": str(class_mapping_path.resolve()),
            "dataset_summary": str(dataset_summary_path.resolve()),
            "onnx": str(final_onnx_path.resolve()),
            "saved_model": str(saved_model_dir.resolve()),
            "metrics_summary": str(metrics_summary_path.resolve()),
            "model_size_report": str(size_report_path.resolve()),
            "prediction_consistency": str(consistency_path.resolve()) if consistency_path else None,
            "console_log": str((run_dir / "console.log").resolve()),
            "latest_files": latest_files,
        }
    }

    summary_json_path = run_dir / "export_quant_summary.json"
    write_json(summary_json_path, summary)

    summary_txt_path = run_dir / "export_quant_summary.txt"

    lines = []
    lines.append("========== MobileNetV3 垃圾分类模型量化导出报告 ==========")
    lines.append(f"生成时间：{now_str()}")
    lines.append("")
    lines.append(f"数据集目录：{data_dir.resolve()}")
    lines.append(f"checkpoint：{ckpt_path.resolve()}")
    lines.append(f"输出目录：{run_dir.resolve()}")
    lines.append(f"模型：{model_name}")
    lines.append(f"类别数：{num_classes}")
    lines.append("")
    lines.append("类别映射：")
    for idx in sorted(idx_to_class.keys()):
        lines.append(f"{idx}: {idx_to_class[idx]}")
    lines.append("")
    lines.append("评估结果：")
    for row in metrics_rows:
        if row.get("accuracy") is None:
            lines.append(
                f"{row['backend']} | status={row.get('status')} | "
                f"error={row.get('error', '')}"
            )
        else:
            lines.append(
                f"{row['backend']} | "
                f"acc={row['accuracy']:.6f} | "
                f"drop_vs_pytorch={row['acc_drop_vs_pytorch_fp32']:.6f} | "
                f"avg_latency_ms={row['avg_latency_ms']:.3f}"
            )
    lines.append("")
    lines.append("模型体积：")
    for row in model_size_rows:
        lines.append(
            f"{row['name']} | {row['size_mb']:.3f} MB | {row['path']}"
        )
    lines.append("")
    lines.append("关键文件：")
    lines.append(f"metrics_summary.csv：{metrics_summary_path.resolve()}")
    lines.append(f"model_size_report.csv：{size_report_path.resolve()}")
    lines.append(f"export_quant_summary.json：{summary_json_path.resolve()}")
    lines.append(f"console.log：{(run_dir / 'console.log').resolve()}")

    with open(summary_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info("========== 量化导出全部完成 ==========")
    logger.info(f"summary json: {summary_json_path}")
    logger.info(f"summary txt: {summary_txt_path}")
    logger.info(f"metrics summary: {metrics_summary_path}")
    logger.info(f"model size report: {size_report_path}")

    for row in metrics_rows:
        if row.get("accuracy") is None:
            logger.info(
                f"[RESULT] {row['backend']} | "
                f"status={row.get('status')} | "
                f"error={row.get('error', '')}"
            )
        else:
            logger.info(
                f"[RESULT] {row['backend']} | "
                f"acc={row['accuracy']:.6f} | "
                f"drop={row['acc_drop_vs_pytorch_fp32']:.6f} | "
                f"latency={row['avg_latency_ms']:.3f}ms"
            )


if __name__ == "__main__":
    main()