#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
树莓派上位机：AI 垃圾识别 + 单字符串口控制舵机下位机
========================================================

适配的下位机固件：firmware/mcu_full/main.c

下位机串口协议非常简单：
    树莓派只发送 1 个 ASCII 字符，下位机收到后改变 angle_pwm：
        'R' -> angle_pwm = 8   -> 可回收
        'H' -> angle_pwm = 19  -> 有害
        'K' -> angle_pwm = 29  -> 厨余
        'O' -> angle_pwm = 36  -> 其他

注意：
1. 这个下位机版本不会主动发送 0xA1 触发信号。
2. 这个下位机版本不会返回 0xCC ACK 或 0xDD DONE。
3. 因此本脚本不再等待握手，而是在 AI 识别稳定后直接发送 R/H/K/O。
4. 默认 auto 模式会自动发送；为了防止重复动作，加入了发送冷却时间。
5. 如果你想更安全地调试，可以使用 --mode manual，在画面中按空格发送当前稳定分类。

推荐运行：
    python3 rpi/servo_char_control.py --mode auto

安全调试：
    python3 rpi/servo_char_control.py --mode manual

直接测试串口和舵机：
    python3 rpi/servo_char_control.py --test-char R
    python3 rpi/servo_char_control.py --test-char K
    python3 rpi/servo_char_control.py --test-char H
    python3 rpi/servo_char_control.py --test-char O
"""

# =========================================================
# 0. 环境变量：必须放在 import cv2 之前
# =========================================================

import os

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


# =========================================================
# 1. 标准库与第三方库
# =========================================================

import argparse
import csv
import json
import logging
import sys
import time
import traceback
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import serial
from PIL import Image, ImageDraw, ImageFont

try:
    from tflite_runtime.interpreter import Interpreter
    TFLITE_BACKEND = "tflite_runtime"
except ImportError:
    try:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
        TFLITE_BACKEND = "tensorflow"
    except ImportError as exc:
        raise RuntimeError(
            "未找到 TFLite 解释器。树莓派建议安装：\n"
            "python3 -m pip install tflite-runtime\n"
            "如果使用完整 TensorFlow：python3 -m pip install tensorflow"
        ) from exc


# =========================================================
# 2. 默认配置
# =========================================================

PROJECT_ROOT_DEFAULT = Path(__file__).resolve().parents[2]
MODEL_RELATIVE_PATH = Path("export") / "latest_tflite_fp16.tflite"
CLASS_MAPPING_FILENAME = "config/class_mapping.json"

SERIAL_PORT_DEFAULT = "/dev/ttyAMA0"
BAUDRATE_DEFAULT = 9600
SERIAL_TIMEOUT = 0.02

CAMERA_INDEX_DEFAULT = 0
CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 480
WINDOW_NAME = "Garbage Sorting - Servo Char Protocol"

# 模型预处理参数：保持与你已有 FP16 TFLite 导出流程一致
RESIZE_SIZE = 256
CROP_SIZE = 224
RGB_INPUT = True
NORMALIZE = True
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# ROI 引导框比例
ROI_W_RATIO = 0.56
ROI_H_RATIO = 0.70

# AI 稳定判断
CONF_THRESHOLD_DEFAULT = 0.80
STABLE_FRAMES_DEFAULT = 3
FRAME_INTERVAL_SEC = 0.03

# 自动发送冷却。下位机没有 DONE 反馈，所以只能靠时间防重复。
SEND_COOLDOWN_SEC_DEFAULT = 5.0

# 默认类别映射：必须与你训练导出时的 class_to_idx 保持一致
DEFAULT_IDX_TO_CLASS = {
    0: "其他",
    1: "厨余",
    2: "可回收",
    3: "有害",
}

CLASS_DISPLAY_NAME = {
    "可回收": "可回收垃圾",
    "厨余": "厨余垃圾",
    "有害": "有害垃圾",
    "其他": "其他垃圾",
}

# 关键：适配 firmware/mcu_full/main.c 的单字符协议
# 注意厨余是 K，不是 C，因为你的 C 代码里判断的是 ch == 'K'
CLASS_TO_SERVO_CHAR = {
    "可回收": "R",  # Recyclable
    "厨余": "K",    # Kitchen
    "有害": "H",    # Harmful
    "其他": "O",    # Other
}

VALID_TEST_CHARS = {"R", "H", "K", "O"}

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
]

FONT_SIZE_MAIN = 24
FONT_SIZE_SMALL = 19


# =========================================================
# 3. 通用工具
# =========================================================

def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def timestamp_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def get_display_name(raw_class: str) -> str:
    return CLASS_DISPLAY_NAME.get(raw_class, raw_class)


def hex_str(data: bytes) -> str:
    return " ".join(f"{b:02X}" for b in data)


# =========================================================
# 4. 日志
# =========================================================

def setup_logger(log_dir: Path) -> logging.Logger:
    ensure_dirs(log_dir)

    logger = logging.getLogger("ServoCharSorting")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    console.setLevel(logging.INFO)

    file_handler = RotatingFileHandler(
        log_dir / "servo_char_runtime.log",
        maxBytes=2 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.INFO)

    logger.addHandler(console)
    logger.addHandler(file_handler)

    logger.info("上位机启动：AI + 单字符舵机控制协议 + 满载保护")
    logger.info("TFLite 后端: %s", TFLITE_BACKEND)

    return logger


def append_csv(csv_path: Path, row: Dict) -> None:
    ensure_dirs(csv_path.parent)
    exists = csv_path.exists()

    CSV_FIELDNAMES = [
        "time",
        "round_id",
        "mode",
        "event",
        "pred_id",
        "raw_class",
        "display_class",
        "confidence",
        "stable_count",
        "is_stable",
        "servo_char",
        "tx_hex",
        "rx_hex",
        "mcu_event",
        "system_state",
        "snapshot_path",
        "message",
    ]

    with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in CSV_FIELDNAMES})


# =========================================================
# 5. 类别映射
# =========================================================

def load_idx_to_class(mapping_path: Path, logger: logging.Logger) -> Dict[int, str]:
    """
    优先读取 class_mapping.json。
    支持两种格式：
        {"idx_to_class": {"0": "其他", ...}}
        {"class_to_idx": {"其他": 0, ...}}
    没有文件时使用 DEFAULT_IDX_TO_CLASS。
    """
    if mapping_path.exists():
        try:
            with open(mapping_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            idx_to_class = data.get("idx_to_class")
            if isinstance(idx_to_class, dict):
                result = {int(k): str(v) for k, v in idx_to_class.items()}
                logger.info("已读取 idx_to_class: %s", result)
                return result

            class_to_idx = data.get("class_to_idx")
            if isinstance(class_to_idx, dict):
                result = {int(v): str(k) for k, v in class_to_idx.items()}
                logger.info("已读取 class_to_idx 并反转: %s", result)
                return result

        except Exception as exc:
            logger.warning("读取 class_mapping.json 失败，将使用默认映射: %s", exc)

    logger.info("使用默认类别映射: %s", DEFAULT_IDX_TO_CLASS)
    return DEFAULT_IDX_TO_CLASS.copy()


# =========================================================
# 6. 图像预处理与画面显示
# =========================================================

def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def infer_layout_from_shape(shape: List[int]) -> str:
    if len(shape) != 4:
        return "NHWC"
    if shape[1] in (1, 3) and shape[-1] not in (1, 3):
        return "NCHW"
    return "NHWC"


def quantize_tensor_if_needed(x_float32: np.ndarray, detail) -> np.ndarray:
    dtype = detail["dtype"]
    scale, zero_point = detail.get("quantization", (0.0, 0))

    if dtype == np.float32 or scale in (0.0, None):
        return x_float32.astype(np.float32)

    q = np.round(x_float32 / float(scale) + int(zero_point))

    if dtype == np.int8:
        return np.clip(q, -128, 127).astype(np.int8)
    if dtype == np.uint8:
        return np.clip(q, 0, 255).astype(np.uint8)

    return q.astype(dtype)


def dequantize_tensor_if_needed(y: np.ndarray, detail) -> np.ndarray:
    dtype = detail["dtype"]
    scale, zero_point = detail.get("quantization", (0.0, 0))

    if dtype == np.float32 or scale in (0.0, None):
        return y.astype(np.float32)

    return (y.astype(np.float32) - int(zero_point)) * float(scale)


def resize_keep_ratio(img_rgb: np.ndarray, shorter_side: int) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError("输入图像尺寸无效")

    if w < h:
        new_w = shorter_side
        new_h = int(round(h * shorter_side / w))
    else:
        new_h = shorter_side
        new_w = int(round(w * shorter_side / h))

    return cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def center_crop(img_rgb: np.ndarray, crop_size: int) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    left = max(0, int(round((w - crop_size) / 2)))
    top = max(0, int(round((h - crop_size) / 2)))

    crop = img_rgb[top:top + crop_size, left:left + crop_size]

    if crop.shape[0] != crop_size or crop.shape[1] != crop_size:
        crop = cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)

    return crop


def get_center_roi(frame_bgr: np.ndarray) -> Tuple[int, int, int, int]:
    h, w = frame_bgr.shape[:2]
    roi_w = int(w * ROI_W_RATIO)
    roi_h = int(h * ROI_H_RATIO)

    x1 = max((w - roi_w) // 2, 0)
    y1 = max((h - roi_h) // 2, 0)
    x2 = min(x1 + roi_w, w)
    y2 = min(y1 + roi_h, h)
    return x1, y1, x2, y2


def get_font(font_size: int) -> Optional[ImageFont.FreeTypeFont]:
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, font_size)
            except Exception:
                continue
    return None


FONT_MAIN = get_font(FONT_SIZE_MAIN)
FONT_SMALL = get_font(FONT_SIZE_SMALL)


def draw_text_pil(
    frame_bgr: np.ndarray,
    text_lines: List[str],
    x: int,
    y: int,
    color_rgb=(255, 255, 255),
    line_gap: int = 30,
    font: Optional[ImageFont.FreeTypeFont] = None,
) -> np.ndarray:
    if font is None:
        font = FONT_MAIN

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(pil_img)

    py = y
    for line in text_lines:
        draw.text((x, py), line, font=font, fill=color_rgb)
        py += line_gap

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def draw_overlay(
    frame_bgr: np.ndarray,
    roi_box: Tuple[int, int, int, int],
    status_text: str,
    class_text: str,
    conf_text: str,
    stable_text: str,
    serial_text: str,
    mode_text: str,
    round_text: str,
    fps_text: str,
    color=(0, 255, 0),
) -> np.ndarray:
    x1, y1, x2, y2 = roi_box
    show = frame_bgr.copy()

    cv2.rectangle(show, (x1, y1), (x2, y2), color, 2)

    overlay = show.copy()
    cv2.rectangle(overlay, (10, 10), (630, 280), (0, 0, 0), -1)
    show = cv2.addWeighted(overlay, 0.45, show, 0.55, 0)

    lines = [
        f"状态：{status_text}",
        f"类别：{class_text}",
        f"置信度：{conf_text}",
        f"稳定帧：{stable_text}",
        f"串口：{serial_text}",
        f"模式：{mode_text}",
        f"轮次：{round_text}    FPS：{fps_text}",
        "按键：q退出，s截图，空格发送当前稳定结果，r/k/h/o直接测试",
    ]

    if FONT_MAIN is None:
        y = 34
        for line in lines:
            safe_line = line.encode("ascii", errors="replace").decode("ascii")
            cv2.putText(show, safe_line, (22, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            y += 30
        cv2.putText(show, "ROI", (x1 + 8, y1 - 8 if y1 > 25 else y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.70, color, 2)
        return show

    show = draw_text_pil(
        show,
        lines,
        x=22,
        y=22,
        color_rgb=(255, 255, 255),
        line_gap=30,
        font=FONT_SMALL or FONT_MAIN,
    )

    show = draw_text_pil(
        show,
        ["ROI"],
        x=x1 + 8,
        y=y1 - 30 if y1 > 35 else y1 + 8,
        color_rgb=(color[2], color[1], color[0]),
        line_gap=30,
        font=FONT_SMALL or FONT_MAIN,
    )

    return show


# =========================================================
# 7. TFLite 分类器
# =========================================================

class GarbageClassifierTFLite:
    def __init__(self, model_path: Path, idx_to_class: Dict[int, str], logger: logging.Logger):
        if not model_path.exists():
            raise FileNotFoundError(f"TFLite 模型不存在: {model_path}")

        self.model_path = model_path
        self.idx_to_class = idx_to_class
        self.logger = logger

        self.logger.info("加载模型: %s", model_path)

        self.interpreter = Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()

        self.input_detail = self.interpreter.get_input_details()[0]
        self.output_detail = self.interpreter.get_output_details()[0]

        self.input_shape = list(self.input_detail["shape"])
        self.output_shape = list(self.output_detail["shape"])
        self.input_dtype = self.input_detail["dtype"]
        self.output_dtype = self.output_detail["dtype"]
        self.input_layout = infer_layout_from_shape(self.input_shape)

        self.logger.info("模型输入 shape: %s", self.input_shape)
        self.logger.info("模型输出 shape: %s", self.output_shape)
        self.logger.info("输入 dtype: %s", self.input_dtype)
        self.logger.info("输出 dtype: %s", self.output_dtype)
        self.logger.info("输入布局: %s", self.input_layout)
        self.logger.info("类别映射: %s", self.idx_to_class)

        self.warmup()

    def warmup(self) -> None:
        self.logger.info("模型预热中...")

        if self.input_layout == "NCHW":
            dummy = np.zeros((1, 3, CROP_SIZE, CROP_SIZE), dtype=np.float32)
        else:
            dummy = np.zeros((1, CROP_SIZE, CROP_SIZE, 3), dtype=np.float32)

        dummy = quantize_tensor_if_needed(dummy, self.input_detail)
        self.interpreter.set_tensor(self.input_detail["index"], dummy)
        self.interpreter.invoke()
        _ = self.interpreter.get_tensor(self.output_detail["index"])

        self.logger.info("模型预热完成")

    def preprocess(self, roi_bgr: np.ndarray) -> np.ndarray:
        if RGB_INPUT:
            img = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        else:
            img = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        img = resize_keep_ratio(img, RESIZE_SIZE)
        img = center_crop(img, CROP_SIZE)

        x = img.astype(np.float32) / 255.0

        if NORMALIZE:
            mean = np.array(MEAN, dtype=np.float32).reshape(1, 1, 3)
            std = np.array(STD, dtype=np.float32).reshape(1, 1, 3)
            x = (x - mean) / std

        if self.input_layout == "NCHW":
            x = np.transpose(x, (2, 0, 1))
            x = np.expand_dims(x, axis=0)
        else:
            x = np.expand_dims(x, axis=0)

        return quantize_tensor_if_needed(x.astype(np.float32), self.input_detail)

    def predict(self, roi_bgr: np.ndarray) -> Dict:
        x = self.preprocess(roi_bgr)

        self.interpreter.set_tensor(self.input_detail["index"], x)
        self.interpreter.invoke()

        y = self.interpreter.get_tensor(self.output_detail["index"])
        y = dequantize_tensor_if_needed(y, self.output_detail)

        logits = y.reshape(-1).astype(np.float32)
        probs = softmax_np(logits)

        pred_id = int(np.argmax(probs))
        confidence = float(probs[pred_id])
        raw_class = self.idx_to_class.get(pred_id, str(pred_id))
        display_class = get_display_name(raw_class)

        return {
            "pred_id": pred_id,
            "raw_class": raw_class,
            "display_class": display_class,
            "confidence": confidence,
            "probs": probs.tolist(),
        }


# =========================================================
# 8. 稳定识别器
# =========================================================

class StablePredictor:
    def __init__(self, conf_threshold: float, stable_frames: int):
        self.conf_threshold = conf_threshold
        self.stable_frames = stable_frames
        self.reset()

    def reset(self) -> None:
        self.last_pred_id = None
        self.stable_count = 0
        self.best_result = None
        self.best_conf = -1.0

    def update(self, result: Dict) -> Dict:
        pred_id = result["pred_id"]
        conf = result["confidence"]

        if conf > self.best_conf:
            self.best_conf = conf
            self.best_result = result

        if conf < self.conf_threshold:
            self.stable_count = 0
            self.last_pred_id = None
            return {
                "is_stable": False,
                "stable_count": 0,
                "status": "低置信度",
                "color": (0, 165, 255),
            }

        if pred_id == self.last_pred_id:
            self.stable_count += 1
        else:
            self.last_pred_id = pred_id
            self.stable_count = 1

        is_stable = self.stable_count >= self.stable_frames
        return {
            "is_stable": is_stable,
            "stable_count": self.stable_count,
            "status": "识别稳定" if is_stable else "识别中",
            "color": (0, 255, 0) if is_stable else (255, 255, 0),
        }


# =========================================================
# 9. 串口管理：单字符协议
# =========================================================

class ServoCharSerial:
    def __init__(self, port: str, baudrate: int, logger: logging.Logger):
        self.port = port
        self.baudrate = baudrate
        self.logger = logger
        self.ser = None

    def open(self) -> None:
        self.logger.info("打开串口: %s, baud=%d", self.port, self.baudrate)
        self.ser = serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=SERIAL_TIMEOUT,
            xonxoff=False,
            rtscts=False,
            dsrdtr=False,
        )
        time.sleep(1.0)
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()
        self.logger.info("串口打开成功")

    def send_char(self, ch: str) -> bytes:
        if ch not in VALID_TEST_CHARS:
            raise ValueError(f"非法下位机控制字符: {ch!r}，只能是 R/H/K/O")

        if self.ser is None or not self.ser.is_open:
            raise RuntimeError("串口尚未打开")

        data = ch.encode("ascii")
        self.ser.write(data)
        self.ser.flush()
        self.logger.info("发送舵机控制字符: %s -> %s", ch, hex_str(data))
        return data

    def send_class(self, raw_class: str) -> bytes:
        if raw_class not in CLASS_TO_SERVO_CHAR:
            raise ValueError(f"未知分类，无法映射为 R/H/K/O: {raw_class}")
        return self.send_char(CLASS_TO_SERVO_CHAR[raw_class])

    def read_available(self) -> bytes:
        if self.ser is None or not self.ser.is_open:
            return b""
        try:
            n = self.ser.in_waiting
            if n > 0:
                data = self.ser.read(n)
                self.logger.info("串口 RX %d 字节: %s", n, hex_str(data))
                return data
        except Exception as exc:
            self.logger.debug("串口读取异常: %s", exc)
        return b""
    def close(self) -> None:
        if self.ser is not None and self.ser.is_open:
            self.ser.close()
            self.logger.info("串口已关闭")


# =========================================================
# 10. 摄像头管理
# =========================================================

class CameraManager:
    def __init__(self, camera_index: int, logger: logging.Logger):
        self.camera_index = camera_index
        self.logger = logger
        self.cap = None

    def open(self) -> None:
        self.logger.info("打开摄像头 index=%d", self.camera_index)
        self.cap = cv2.VideoCapture(self.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)

        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开摄像头: {self.camera_index}")

        # 预读几帧，让摄像头自动曝光稳定
        for _ in range(10):
            self.cap.read()
            time.sleep(0.03)

        self.logger.info("摄像头打开成功")

    def read(self) -> np.ndarray:
        if self.cap is None:
            raise RuntimeError("摄像头尚未打开")

        ret, frame = self.cap.read()
        if not ret or frame is None:
            raise RuntimeError("摄像头读取失败")
        return frame

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.logger.info("摄像头已关闭")


# =========================================================
# 11. 主系统
# =========================================================

class ServoCharSortingApp:
    def __init__(self, args):
        self.args = args
        self.project_root = Path(args.project_root).expanduser().resolve()
        self.model_path = Path(args.model_path).expanduser().resolve() if args.model_path else self.project_root / MODEL_RELATIVE_PATH
        self.mapping_path = Path(args.class_mapping).expanduser().resolve() if args.class_mapping else self.project_root / CLASS_MAPPING_FILENAME
        self.log_dir = self.project_root / "Logs"
        self.capture_dir = self.project_root / "Captures_Servo_Char"
        self.csv_path = self.log_dir / "servo_char_rounds.csv"

        ensure_dirs(self.log_dir, self.capture_dir)
        self.logger = setup_logger(self.log_dir)

        self.logger.info("PROJECT_ROOT: %s", self.project_root)
        self.logger.info("MODEL_PATH: %s", self.model_path)
        self.logger.info("CLASS_MAPPING: %s", self.mapping_path)
        self.logger.info("串口协议: 单字符 R/H/K/O + 满载反馈 F/N")
        self.logger.info("运行模式: %s", self.args.mode)

        idx_to_class = load_idx_to_class(self.mapping_path, self.logger)

        self.classifier = GarbageClassifierTFLite(self.model_path, idx_to_class, self.logger)
        self.stable_predictor = StablePredictor(args.conf_threshold, args.stable_frames)
        self.serial_mgr = ServoCharSerial(args.serial_port, args.baudrate, self.logger)
        self.camera_mgr = CameraManager(args.camera_index, self.logger)


        self.paused_by_full = False
        self.full_message = ""
        self.last_mcu_rx_text = "未收到"
        self.full_since_time = None
        self.awaiting_manual_resume = False
        self.running = True
        self.round_id = 0
        self.last_send_time = 0.0
        self.last_frame_time = time.time()
        self.fps = 0.0

        self.latest_result = None
        self.latest_stable_info = None
        self.latest_show_frame = None

        self.status_text = "初始化中"
        self.class_text = "-"
        self.conf_text = "-"
        self.stable_text = "-"
        self.serial_text = "未发送"

    def update_fps(self) -> None:
        now = time.time()
        dt = now - self.last_frame_time
        self.last_frame_time = now
        if dt > 0:
            current = 1.0 / dt
            self.fps = 0.9 * self.fps + 0.1 * current if self.fps > 0 else current

    def save_snapshot(self, frame: np.ndarray, result: Optional[Dict], tag: str) -> str:
        ensure_dirs(self.capture_dir)
        ts = timestamp_str()

        if result:
            raw_class = result.get("raw_class", "unknown")
            conf = result.get("confidence", 0.0)
            filename = f"round_{self.round_id:04d}_{raw_class}_{conf:.3f}_{tag}_{ts}.jpg"
        else:
            filename = f"round_{self.round_id:04d}_{tag}_{ts}.jpg"

        path = self.capture_dir / filename
        cv2.imwrite(str(path), frame)
        self.logger.info("保存截图: %s", path)
        return str(path)


    def system_state_str(self) -> str:
        if self.paused_by_full:
            return "FULL-PAUSED"
        if self.awaiting_manual_resume:
            return "AWAITING-MANUAL-RESUME"
        return "NORMAL"

    def process_mcu_events(self) -> None:
        data = self.serial_mgr.read_available()
        if not data:
            return
        for byte_val in data:
            ch = chr(byte_val)
            if ch == 'F':
                if not self.paused_by_full:
                    self.paused_by_full = True
                    self.awaiting_manual_resume = False
                    self.full_message = "垃圾桶已满，分类已暂停，请清理"
                    self.full_since_time = time.time()
                    self.last_mcu_rx_text = f"RX 'F' / 0x{byte_val:02X}"
                    self.serial_text = self.last_mcu_rx_text
                    self.status_text = self.full_message
                    self.stable_predictor.reset()
                    self.logger.warning("[MCU] 收到 F，暂停自动分类")
                    append_csv(self.csv_path, {
                        "time": now_str(), "round_id": self.round_id,
                        "mode": self.args.mode, "event": "mcu_full",
                        "rx_hex": hex_str(bytes([byte_val])),
                        "mcu_event": "FULL", "system_state": self.system_state_str(),
                        "message": "下位机F，垃圾桶满载，分类已暂停",
                    })
            elif ch == 'N':
                self.last_mcu_rx_text = f"RX 'N' / 0x{byte_val:02X}"
                self.logger.info("[MCU] 收到 N（恢复正常）")
                if self.args.auto_resume_after_clear:
                    self.paused_by_full = False
                    self.awaiting_manual_resume = False
                    self.full_message = ""
                    self.full_since_time = None
                    self.status_text = "满载已解除，分类已自动恢复"
                    self.serial_text = self.last_mcu_rx_text
                    self.stable_predictor.reset()
                    self.logger.info("[MCU] 自动恢复分类")
                    append_csv(self.csv_path, {
                        "time": now_str(), "round_id": self.round_id,
                        "mode": self.args.mode, "event": "mcu_normal_auto_resume",
                        "rx_hex": hex_str(bytes([byte_val])),
                        "mcu_event": "NORMAL", "system_state": self.system_state_str(),
                        "message": "下位机N，已自动恢复分类",
                    })
                else:
                    self.awaiting_manual_resume = True
                    self.status_text = "满载已解除，按 c 继续分类"
                    self.serial_text = self.last_mcu_rx_text
                    self.logger.info("[MCU] 收到 N，等待用户按 c 手动恢复")
                    append_csv(self.csv_path, {
                        "time": now_str(), "round_id": self.round_id,
                        "mode": self.args.mode, "event": "mcu_normal_awaiting_manual",
                        "rx_hex": hex_str(bytes([byte_val])),
                        "mcu_event": "NORMAL", "system_state": self.system_state_str(),
                        "message": "下位机N，等待用户按c手动恢复",
                    })
    def send_result_to_mcu(self, result: Dict, event: str, frame_for_snapshot: Optional[np.ndarray] = None) -> bool:
        if self.paused_by_full and not getattr(self.args, 'allow_send_while_full', False):
            self.logger.warning("满载暂停，禁止发送分类命令")
            self.serial_text = "满载暂停，禁止发送"
            append_csv(self.csv_path, {
                "time": now_str(), "round_id": self.round_id,
                "mode": self.args.mode, "event": "send_blocked_full",
                "raw_class": result.get("raw_class"),
                "display_class": result.get("display_class"),
                "system_state": self.system_state_str(),
                "message": "满载暂停保护：禁止发送R/H/K/O",
            })
            return False
        if self.awaiting_manual_resume and not getattr(self.args, 'allow_send_while_full', False):
            self.logger.warning("等待手动恢复，禁止发送分类命令")
            self.serial_text = "等待手动恢复，禁止发送"
            return False
        raw_class = result["raw_class"]
        servo_char = CLASS_TO_SERVO_CHAR.get(raw_class)

        if servo_char is None:
            self.logger.error("分类 %s 无法映射到舵机字符", raw_class)
            self.serial_text = "映射失败"
            return False

        try:
            tx_data = self.serial_mgr.send_class(raw_class)
            self.round_id += 1
            self.last_send_time = time.time()
            self.serial_text = f"TX '{servo_char}' / 0x{tx_data[0]:02X}"
            self.status_text = "已发送舵机控制字符"

            snapshot_path = ""
            if frame_for_snapshot is not None:
                snapshot_path = self.save_snapshot(frame_for_snapshot, result, f"send_{servo_char}")

            append_csv(self.csv_path, {
                "time": now_str(),
                "round_id": self.round_id,
                "mode": self.args.mode,
                "event": event,
                "pred_id": result.get("pred_id"),
                "raw_class": raw_class,
                "display_class": result.get("display_class"),
                "confidence": f"{result.get('confidence', 0.0):.6f}",
                "stable_count": self.latest_stable_info.get("stable_count", "") if self.latest_stable_info else "",
                "is_stable": self.latest_stable_info.get("is_stable", "") if self.latest_stable_info else "",
                "servo_char": servo_char,
                "tx_hex": hex_str(tx_data),
                "snapshot_path": snapshot_path,
                "system_state": self.system_state_str(),
                "message": "single-char protocol for firmware/mcu_full/main.c",
            })

            return True

        except Exception as exc:
            self.logger.exception("发送失败: %s", exc)
            self.serial_text = f"发送失败: {exc}"
            append_csv(self.csv_path, {
                "time": now_str(),
                "round_id": self.round_id,
                "mode": self.args.mode,
                "event": "send_failed",
                "raw_class": raw_class,
                "display_class": result.get("display_class"),
                "confidence": f"{result.get('confidence', 0.0):.6f}",
                "servo_char": servo_char,
                "message": str(exc),
            })
            return False

    def should_auto_send(self, stable_info: Dict) -> bool:
        if self.args.mode != "auto":
            return False
        if not stable_info.get("is_stable", False):
            return False
        if self.latest_result is None:
            return False

        if self.paused_by_full or self.awaiting_manual_resume:
            return False
        now = time.time()
        return (now - self.last_send_time) >= self.args.send_cooldown

    def handle_key(self, key: int) -> None:
        if key == 255:
            return

        if key == ord("q"):
            self.running = False
        if key == ord("c"):
            if self.paused_by_full or self.awaiting_manual_resume:
                self.paused_by_full = False
                self.awaiting_manual_resume = False
                self.full_message = ""
                self.full_since_time = None
                self.status_text = "用户手动恢复分类"
                self.serial_text = self.last_mcu_rx_text
                self.stable_predictor.reset()
                self.logger.info("用户按c手动恢复分类")
                append_csv(self.csv_path, {
                    "time": now_str(), "round_id": self.round_id,
                    "mode": self.args.mode, "event": "user_manual_resume",
                    "system_state": self.system_state_str(),
                    "message": "用户按c手动恢复分类",
                })
            return
        if key == ord("p"):
            if self.paused_by_full:
                self.logger.info("当前为满载暂停，无法用p键切换")
                return
            if not hasattr(self, '_manual_paused'):
                self._manual_paused = False
            self._manual_paused = not self._manual_paused
            if self._manual_paused:
                self.status_text = "用户手动暂停"
            else:
                self.status_text = "已恢复"
                self.stable_predictor.reset()
            return
        if self.paused_by_full:
            self.logger.info("当前为满载暂停状态，禁止发送分类命令")
            return
        if self.awaiting_manual_resume:
            self.logger.info("等待手动恢复分类，请按c")
            return
            return

        if key == ord("s") and self.latest_show_frame is not None:
            self.save_snapshot(self.latest_show_frame, self.latest_result, "manual_snapshot")
            return

        # manual 模式下，空格发送当前稳定识别结果
        if key == ord(" "):
            if self.latest_result is not None and self.latest_stable_info and self.latest_stable_info.get("is_stable"):
                self.send_result_to_mcu(self.latest_result, "manual_space", self.latest_show_frame)
            else:
                self.logger.info("当前结果还不稳定，未发送")
            return

        # 直接测试字符：r/k/h/o
        lower_to_char = {
            ord("r"): "R",
            ord("k"): "K",
            ord("h"): "H",
            ord("o"): "O",
        }
        if key in lower_to_char:
            ch = lower_to_char[key]
            try:
                tx_data = self.serial_mgr.send_char(ch)
                self.round_id += 1
                self.last_send_time = time.time()
                self.serial_text = f"手动TX '{ch}' / 0x{tx_data[0]:02X}"
                self.status_text = "手动测试字符已发送"
                append_csv(self.csv_path, {
                    "time": now_str(),
                    "round_id": self.round_id,
                    "mode": self.args.mode,
                    "event": "manual_key_char",
                    "servo_char": ch,
                    "tx_hex": hex_str(tx_data),
                    "message": "keyboard direct test",
                })
            except Exception as exc:
                self.logger.exception("手动发送字符失败: %s", exc)
                self.serial_text = f"手动发送失败: {exc}"

    def run(self) -> None:
        self.serial_mgr.open()
        self.camera_mgr.open()

        if not self.args.no_window:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW_NAME, CAPTURE_WIDTH, CAPTURE_HEIGHT)

        self.status_text = "摄像头已打开，连续识别中"

        try:
            while self.running:
                self.process_mcu_events()
                frame = self.camera_mgr.read()
                self.update_fps()

                roi_box = get_center_roi(frame)
                x1, y1, x2, y2 = roi_box
                roi = frame[y1:y2, x1:x2]
                if self.paused_by_full or self.awaiting_manual_resume:
                    self.update_fps()
                    roi_box = get_center_roi(frame)
                    s = "FULL-PAUSED" if self.paused_by_full else "AWAITING-RESUME"
                    show = draw_overlay(
                        frame_bgr=frame, roi_box=roi_box,
                        status_text=self.status_text,
                        class_text="-", conf_text="-", stable_text="0",
                        serial_text=self.serial_text,
                        mode_text=f"{self.args.mode}, {s}",
                        round_text=str(self.round_id),
                        fps_text=f"{self.fps:.1f}",
                    )
                    self.latest_show_frame = show
                    if not self.args.no_window:
                        cv2.imshow(WINDOW_NAME, show)
                        key = cv2.waitKey(1) & 0xFF
                        self.handle_key(key)
                    time.sleep(FRAME_INTERVAL_SEC)
                    continue

                result = self.classifier.predict(roi)
                stable_info = self.stable_predictor.update(result)

                self.latest_result = result
                self.latest_stable_info = stable_info

                self.status_text = stable_info["status"]
                self.class_text = result["display_class"]
                self.conf_text = f"{result['confidence']:.3f}"
                self.stable_text = str(stable_info["stable_count"])

                # 自动模式：识别稳定后，直接发送 R/H/K/O
                if self.should_auto_send(stable_info):
                    self.send_result_to_mcu(result, "auto_stable", None)

                show = draw_overlay(
                    frame_bgr=frame,
                    roi_box=roi_box,
                    status_text=self.status_text,
                    class_text=self.class_text,
                    conf_text=self.conf_text,
                    stable_text=self.stable_text,
                    serial_text=self.serial_text,
                    mode_text=f"{self.args.mode}, cooldown={self.args.send_cooldown:.1f}s",
                    round_text=str(self.round_id),
                    fps_text=f"{self.fps:.1f}",
                    color=stable_info["color"],
                )
                self.latest_show_frame = show

                if not self.args.no_window:
                    cv2.imshow(WINDOW_NAME, show)
                    key = cv2.waitKey(1) & 0xFF
                    self.handle_key(key)
                else:
                    # 无窗口模式下，每隔一小段时间输出一次状态，避免刷屏
                    if int(time.time() * 2) % 2 == 0:
                        print(
                            f"\r{now_str()} | {self.status_text} | {self.class_text} "
                            f"conf={self.conf_text} stable={self.stable_text} serial={self.serial_text}",
                            end="",
                            flush=True,
                        )

                time.sleep(FRAME_INTERVAL_SEC)

        except KeyboardInterrupt:
            self.logger.info("用户中断，准备退出")
        except Exception:
            self.logger.error("主循环异常:\n%s", traceback.format_exc())
            raise
        finally:
            self.close()

    def close(self) -> None:
        self.camera_mgr.close()
        self.serial_mgr.close()
        if not self.args.no_window:
            cv2.destroyAllWindows()
        self.logger.info("程序已退出")


# =========================================================
# 12. 命令行参数
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="树莓派 AI 垃圾识别 + R/H/K/O 单字符舵机控制上位机"
    )

    parser.add_argument(
        "--project-root",
        default=str(PROJECT_ROOT_DEFAULT),
        help="项目根目录，默认自动检测项目根目录",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="TFLite 模型路径。默认使用 <project-root>/export/latest_tflite_fp16.tflite",
    )
    parser.add_argument(
        "--class-mapping",
        default=None,
        help="class_mapping.json 路径。默认使用 <project-root>/config/class_mapping.json",
    )
    parser.add_argument(
        "--serial-port",
        default=SERIAL_PORT_DEFAULT,
        help="串口设备，默认 /dev/ttyAMA0；USB-TTL 常见为 /dev/ttyUSB0",
    )
    parser.add_argument(
        "--baudrate",
        type=int,
        default=BAUDRATE_DEFAULT,
        help="波特率，必须与 firmware/mcu_full/main.c 一致，默认 9600",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=CAMERA_INDEX_DEFAULT,
        help="摄像头编号，默认 0",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "manual"],
        default="auto",
        help="auto=稳定后自动发送；manual=按空格发送当前稳定结果。默认 auto",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=CONF_THRESHOLD_DEFAULT,
        help="稳定识别置信度阈值，默认 0.80",
    )
    parser.add_argument(
        "--stable-frames",
        type=int,
        default=STABLE_FRAMES_DEFAULT,
        help="连续相同高置信度帧数，默认 3",
    )
    parser.add_argument(
        "--send-cooldown",
        type=float,
        default=SEND_COOLDOWN_SEC_DEFAULT,
        help="自动发送后的冷却时间；下位机没有 DONE 反馈，所以用它防止重复发送。默认 5 秒",
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="不显示 OpenCV 窗口，适合 SSH 运行",
    )
    parser.add_argument(
        "--test-char",
        choices=sorted(VALID_TEST_CHARS),
        default=None,
        help="只发送一个测试字符后退出。例如 --test-char R / K / H / O",
    )

    parser.add_argument(
        "--auto-resume-after-clear",
        action="store_true",
        default=False,
        help="满载解除后自动恢复分类。默认不自动，需按c键手动恢复",
    )
    parser.add_argument(
        "--allow-send-while-full",
        action="store_true",
        default=False,
        help="[调试用] 满载时仍允许发送分类命令。默认禁止。",
    )
    return parser.parse_args()


# =========================================================
# 13. 程序入口
# =========================================================

def main():
    args = parse_args()

    # 直接测试模式：只打开串口，发一个 R/H/K/O，然后退出。
    # 用于确认 firmware/mcu_full/main.c 是否能收到字符并改变舵机角度。
    if args.test_char:
        project_root = Path(args.project_root).expanduser().resolve()
        log_dir = project_root / "Logs"
        logger = setup_logger(log_dir)
        serial_mgr = ServoCharSerial(args.serial_port, args.baudrate, logger)
        try:
            serial_mgr.open()
            serial_mgr.send_char(args.test_char)
            logger.info("测试字符 %s 已发送，程序退出", args.test_char)
        finally:
            serial_mgr.close()
        return

    app = ServoCharSortingApp(args)
    app.run()


if __name__ == "__main__":
    main()
