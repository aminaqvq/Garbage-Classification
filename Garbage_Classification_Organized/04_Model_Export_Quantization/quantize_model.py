import os
import csv
import json
import time
import shutil
import logging
import warnings
from pathlib import Path
from datetime import datetime
from collections import OrderedDict, Counter
PROJECT_ROOT = Path(__file__).resolve().parent.parent
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from PIL import Image

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import torch
import torch.nn as nn
from torchvision import models

import onnx
import onnxruntime as ort
from onnxsim import simplify

import tensorflow as tf

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================================================
# 配置区：主要改这里
# =========================================================

CONFIG: Dict[str, Any] = {
    # 项目路径
    # DATA_DIR 下应该有 train / val / test
    # 如果你生成了 calibration 目录，也会优先使用 calibration
    "data_dir": "garbage_dataset",

    # 训练好的 best 模型
    # 如果你用的是我前面给你的完整训练脚本，通常是：
    # outputs/latest_mobilenetv3_best.pt
    "ckpt": "outputs/latest_mobilenetv3_best.pt",

    # 量化导出目录
    "outdir": "export",

    # 模型名称：
    # auto 会优先读取 checkpoint 里的 model_name
    # 如果读不到，默认 mobilenet_v3_small
    # 可选：mobilenet_v3_small / mobilenet_v3_large / auto
    "model_name": "auto",

    # 类别数：
    # auto 会优先根据 checkpoint 里的 class_to_idx 判断
    "num_classes": "auto",

    # 设备
    # 导出和量化建议 cpu；如果只验证 PyTorch 也可以 cuda
    "device": "cpu",

    # 图像预处理
    "img_size": 224,
    "resize_size": 256,
    "eval_preprocess": "resize_center_crop",  # resize_center_crop / direct_resize
    "normalize": True,
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "rgb": True,

    # ONNX 导出
    "opset": 13,
    "input_name": "input",
    "output_name": "output",
    "onnx_simplify": True,
    "onnx_dynamic_batch": True,
    "check_onnx_with_ort": True,

    # TFLite 导出
    "export_tflite_float32": True,
    "export_tflite_fp16": True,
    "export_tflite_int8": True,

    # INT8 TFLite 输入输出类型
    # 推荐默认 float32：模型内部 int8，但输入输出仍然 float32，部署更简单
    # 如果你要纯 int8 输入输出，可改成 int8
    "int8_input_type": "float32",   # float32 / int8 / uint8
    "int8_output_type": "float32",  # float32 / int8 / uint8

    # 校准集
    # 如果 data_dir/calibration 存在，就使用 calibration
    # 如果不存在，就用 train
    "calib_split": "calibration",
    "calib_fallback_split": "train",
    "calib_limit": 500,

    # 测试集评估
    "test_split": "test",
    # 0 表示测试集全量评估；如果想快速试跑，可以改成 50 或 100
    "verify_limit": 0,

    # 日志与覆盖
    "overwrite_outdir": False,
    "save_per_image_predictions": True,
    "save_confusion_matrix_png": True,

    # 随机种子
    "seed": 42,
}


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

    run_dir = out_root / f"quant_run_{timestamp_str()}"
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
    limit: int = 0
) -> List[Dict[str, Any]]:
    split_dir = data_dir / split

    if not split_dir.exists():
        raise FileNotFoundError(f"找不到数据集 split 目录：{split_dir.resolve()}")

    samples = []

    for class_name, class_idx in class_to_idx.items():
        class_dir = split_dir / class_name

        if not class_dir.exists():
            continue

        image_paths = [
            p for p in class_dir.rglob("*")
            if is_image_file(p)
        ]

        image_paths.sort()

        for p in image_paths:
            samples.append({
                "image_path": p,
                "label": int(class_idx),
                "class_name": class_name,
                "split": split,
            })

    samples.sort(key=lambda x: str(x["image_path"]))

    if limit and limit > 0:
        samples = samples[:limit]

    if not samples:
        raise ValueError(f"{split_dir.resolve()} 中没有找到有效图片。")

    return samples


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
        return list_samples_from_split(data_dir, calib_split, class_to_idx, limit=limit)

    logger.warning(f"没有找到 {calib_split}，改用 {fallback_split} 作为 INT8 校准集。")
    return list_samples_from_split(data_dir, fallback_split, class_to_idx, limit=limit)


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

def convert_onnx_to_saved_model(
    onnx_path: Path,
    saved_model_dir: Path,
    logger: logging.Logger
) -> Path:
    try:
        import onnx2tf
    except Exception as e:
        raise ImportError(
            "缺少 onnx2tf，无法执行 ONNX -> SavedModel。"
            "请先安装：pip install onnx2tf"
        ) from e

    if saved_model_dir.exists():
        shutil.rmtree(saved_model_dir)

    logger.info(f"[onnx2tf] converting ONNX -> SavedModel: {saved_model_dir}")

    onnx2tf.convert(
        input_onnx_file_path=str(onnx_path),
        output_folder_path=str(saved_model_dir),
        copy_onnx_input_output_names_to_tflite=True,
        non_verbose=True,
        output_signaturedefs=True,
    )

    pb_path = saved_model_dir / "saved_model.pb"

    if not pb_path.exists():
        raise FileNotFoundError(f"onnx2tf 结束后没有找到 saved_model.pb：{pb_path}")

    logger.info("[onnx2tf] SavedModel export done.")

    return saved_model_dir


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


# =========================================================
# 主程序
# =========================================================

def main():
    warnings.filterwarnings("ignore", category=UserWarning)

    cfg = dict(CONFIG)
    cfg["data_dir"] = str((PROJECT_ROOT / cfg["data_dir"]).resolve())
    cfg["ckpt"] = str((PROJECT_ROOT / cfg["ckpt"]).resolve())
    cfg["outdir"] = str((PROJECT_ROOT / cfg["outdir"]).resolve())
    set_seed(int(cfg.get("seed", 42)))

    data_dir = Path(cfg["data_dir"])
    ckpt_path = Path(cfg["ckpt"])

    if not data_dir.exists():
        raise FileNotFoundError(f"数据集目录不存在：{data_dir.resolve()}")

    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint 不存在：{ckpt_path.resolve()}")

    run_dir = prepare_run_dir(cfg)
    logger = setup_logger(run_dir)

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

    test_split = str(cfg.get("test_split", "test"))
    verify_limit = int(cfg.get("verify_limit", 0))

    test_samples = list_samples_from_split(
        data_dir,
        test_split,
        class_to_idx,
        limit=verify_limit
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

    onnx_path = run_dir / "model.onnx"

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

    if cfg.get("export_tflite_float32", True):
        path = run_dir / "model_float32.tflite"
        tflite_paths["tflite_float32"] = convert_saved_model_to_tflite(
            saved_model_dir=saved_model_dir,
            out_path=path,
            quant_type="float32",
            cfg=cfg,
            calib_samples=calib_samples,
            logger=logger
        )

    if cfg.get("export_tflite_fp16", True):
        path = run_dir / "model_fp16.tflite"
        tflite_paths["tflite_fp16"] = convert_saved_model_to_tflite(
            saved_model_dir=saved_model_dir,
            out_path=path,
            quant_type="fp16",
            cfg=cfg,
            calib_samples=calib_samples,
            logger=logger
        )

    if cfg.get("export_tflite_int8", True):
        path = run_dir / "model_int8.tflite"
        tflite_paths["tflite_int8"] = convert_saved_model_to_tflite(
            saved_model_dir=saved_model_dir,
            out_path=path,
            quant_type="int8",
            cfg=cfg,
            calib_samples=calib_samples,
            logger=logger
        )

    for backend_name, tflite_path in tflite_paths.items():
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

        all_metrics.append(metrics_tf)
        all_prediction_records[backend_name] = records_tf

        logger.info(f"========== {backend_name} Report ==========")
        logger.info("\n" + report_tf)

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
        acc_drop = baseline_acc - item["accuracy"]

        metrics_rows.append({
            "backend": item["backend"],
            "accuracy": item["accuracy"],
            "acc_drop_vs_pytorch_fp32": acc_drop,
            "avg_latency_ms": item["avg_latency_ms"],
            "num_samples": item["num_samples"],
            "classification_report_txt": item["classification_report_txt"],
            "confusion_matrix_csv": item["confusion_matrix_csv"],
            "confusion_matrix_png": item["confusion_matrix_png"],
            "per_image_predictions_csv": item["per_image_predictions_csv"],
        })

    metrics_summary_path = run_dir / "metrics_summary.csv"

    write_csv(
        metrics_summary_path,
        metrics_rows,
        [
            "backend",
            "accuracy",
            "acc_drop_vs_pytorch_fp32",
            "avg_latency_ms",
            "num_samples",
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
        logger.info(
            f"[RESULT] {row['backend']} | "
            f"acc={row['accuracy']:.6f} | "
            f"drop={row['acc_drop_vs_pytorch_fp32']:.6f} | "
            f"latency={row['avg_latency_ms']:.3f}ms"
        )


if __name__ == "__main__":
    main()