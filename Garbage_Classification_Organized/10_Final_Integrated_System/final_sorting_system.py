#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
树莓派上位机最终版：垃圾识别 + 串口握手 + 52RC 分拣控制

流程：
1) 等待 52RC 单片机发送 0xA1，表示超声波检测到物体。
2) 摄像头采集 ROI，TFLite 模型连续识别，达到稳定帧数后确定分类。
3) 发送分类帧：AA 01/02/03/04 55。
4) 等待 52RC 返回 0xCC ACK 和 0xDD DONE。
5) 保存日志、统计文件、识别截图。

默认项目路径：自动检测（脚本所在目录的父目录）
默认模型路径：<项目路径>/export/latest_tflite_fp16.tflite
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
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

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

PROJECT_ROOT_DEFAULT = Path(__file__).resolve().parent.parent

SERIAL_PORT_DEFAULT = "/dev/ttyAMA0"
BAUDRATE = 9600
SERIAL_TIMEOUT = 0.02

CAMERA_INDEX_DEFAULT = 0
CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 480
WINDOW_NAME = "Garbage Final Sorting System"

# 模型预处理参数：保持与你当前 FP16 TFLite 导出一致
RESIZE_SIZE = 256
CROP_SIZE = 224
RGB_INPUT = True
NORMALIZE = True
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# 识别稳定策略
CONF_THRESHOLD_DEFAULT = 0.80
STABLE_FRAMES_DEFAULT = 3
PREDICT_TIMEOUT_SEC_DEFAULT = 4.0
MIN_FALLBACK_CONF_DEFAULT = 0.55
UNCERTAIN_POLICY_DEFAULT = "send_best"  # send_best / send_other / skip
FRAME_INTERVAL_SEC = 0.03
TRIGGER_DEBOUNCE_SEC = 0.8
ACK_TIMEOUT_SEC = 2.0
DONE_TIMEOUT_SEC = 12.0  # C 端含舵机和电机动作，给足时间

# ROI 引导框比例
ROI_W_RATIO = 0.56
ROI_H_RATIO = 0.70

# 串口协议
FRAME_HEAD = 0xAA
FRAME_TAIL = 0x55

# 52RC -> 树莓派
MCU_TRIGGER_READY = 0xA1
MCU_ACK_RECEIVED = 0xCC
MCU_DONE = 0xDD
MCU_ERROR = 0xEE

# 树莓派 -> 52RC
CLASS_TO_MCU_CODE = {
    "可回收": 0x01,
    "厨余": 0x02,
    "有害": 0x03,
    "其他": 0x04,
}

CLASS_DISPLAY_NAME = {
    "可回收": "可回收垃圾",
    "厨余": "厨余垃圾",
    "有害": "有害垃圾",
    "其他": "其他垃圾",
}

DEFAULT_IDX_TO_CLASS = {
    0: "其他",
    1: "厨余",
    2: "可回收",
    3: "有害",
}

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
]


# =========================================================
# 3. 工具函数
# =========================================================

def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def timestamp_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def hex_str(data: bytes) -> str:
    return " ".join(f"{b:02X}" for b in data)


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


FONT_MAIN = get_font(25)
FONT_SMALL = get_font(20)


def setup_logger(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("GarbageFinalSystem")
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
        log_dir / "final_runtime.log",
        maxBytes=2 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.INFO)

    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


def load_idx_to_class(mapping_path: Path, logger: logging.Logger) -> Dict[int, str]:
    if mapping_path.exists():
        try:
            data = json.loads(mapping_path.read_text(encoding="utf-8"))
            idx_to_class = data.get("idx_to_class")
            if isinstance(idx_to_class, dict):
                result = {int(k): str(v) for k, v in idx_to_class.items()}
                logger.info("从 class_mapping.json 读取 idx_to_class: %s", result)
                return result
            class_to_idx = data.get("class_to_idx")
            if isinstance(class_to_idx, dict):
                result = {int(v): str(k) for k, v in class_to_idx.items()}
                logger.info("从 class_mapping.json 读取 class_to_idx 并反转: %s", result)
                return result
        except Exception as exc:
            logger.warning("读取 class_mapping.json 失败，将使用默认映射: %s", exc)
    logger.info("使用默认类别映射: %s", DEFAULT_IDX_TO_CLASS)
    return DEFAULT_IDX_TO_CLASS.copy()


def get_display_name(raw_class: str) -> str:
    return CLASS_DISPLAY_NAME.get(raw_class, raw_class)


# =========================================================
# 4. 图像显示
# =========================================================

def draw_text_pil(frame_bgr: np.ndarray, lines: List[str], x: int, y: int,
                  font: Optional[ImageFont.FreeTypeFont], line_gap: int = 30,
                  color_rgb=(255, 255, 255)) -> np.ndarray:
    if font is None:
        return frame_bgr
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(img)
    yy = y
    for line in lines:
        draw.text((x, yy), line, font=font, fill=color_rgb)
        yy += line_gap
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def draw_overlay(frame_bgr: np.ndarray, roi_box, status_text: str, class_text: str,
                 conf_text: str, stable_text: str, serial_text: str,
                 round_text: str, fps_text: str, color=(0, 255, 0)) -> np.ndarray:
    x1, y1, x2, y2 = roi_box
    show = frame_bgr.copy()
    cv2.rectangle(show, (x1, y1), (x2, y2), color, 2)

    overlay = show.copy()
    cv2.rectangle(overlay, (10, 10), (620, 245), (0, 0, 0), -1)
    show = cv2.addWeighted(overlay, 0.45, show, 0.55, 0)

    lines = [
        f"状态：{status_text}",
        f"类别：{class_text}",
        f"置信度：{conf_text}",
        f"稳定帧：{stable_text}",
        f"串口：{serial_text}",
        f"轮次：{round_text}",
        f"FPS：{fps_text}",
    ]

    if FONT_SMALL is not None:
        show = draw_text_pil(show, lines, 22, 22, FONT_SMALL, 31)
        show = draw_text_pil(
            show, ["ROI"], x1 + 8, y1 - 28 if y1 > 35 else y1 + 8,
            FONT_SMALL, 30, color_rgb=(color[2], color[1], color[0])
        )
    else:
        y = 35
        for line in lines:
            safe = line.encode("ascii", errors="replace").decode("ascii")
            cv2.putText(show, safe, (22, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 0), 2)
            y += 30
        cv2.putText(show, "ROI", (x1 + 8, y1 - 8 if y1 > 25 else y1 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.70, color, 2)
    return show


# =========================================================
# 5. TFLite 分类器
# =========================================================

class GarbageClassifierTFLite:
    def __init__(self, model_path: Path, idx_to_class: Dict[int, str], logger: logging.Logger):
        if not model_path.exists():
            raise FileNotFoundError(f"TFLite 模型不存在: {model_path}")
        self.idx_to_class = idx_to_class
        self.logger = logger

        logger.info("加载 TFLite 模型: %s", model_path)
        self.interpreter = Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()
        self.input_detail = self.interpreter.get_input_details()[0]
        self.output_detail = self.interpreter.get_output_details()[0]
        self.input_shape = list(self.input_detail["shape"])
        self.output_shape = list(self.output_detail["shape"])
        self.input_layout = infer_layout_from_shape(self.input_shape)

        logger.info("TFLite 后端: %s", TFLITE_BACKEND)
        logger.info("模型输入 shape=%s, dtype=%s, layout=%s", self.input_shape, self.input_detail["dtype"], self.input_layout)
        logger.info("模型输出 shape=%s, dtype=%s", self.output_shape, self.output_detail["dtype"])
        logger.info("类别映射: %s", self.idx_to_class)
        self.warmup()

    def warmup(self):
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
        return {
            "pred_id": pred_id,
            "raw_class": raw_class,
            "display_class": get_display_name(raw_class),
            "confidence": confidence,
            "probs": probs.tolist(),
        }


# =========================================================
# 6. 稳定识别器
# =========================================================

class StablePredictor:
    def __init__(self, conf_threshold: float, stable_frames: int):
        self.conf_threshold = conf_threshold
        self.stable_frames = stable_frames
        self.reset()

    def reset(self):
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
            return {"is_stable": False, "stable_count": 0, "status": "低置信度", "color": (0, 165, 255)}
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
# 7. 串口与摄像头管理
# =========================================================

class SerialManager:
    def __init__(self, port: str, logger: logging.Logger):
        self.logger = logger
        logger.info("打开串口: %s, baud=%d, 8N1", port, BAUDRATE)
        self.ser = serial.Serial(
            port=port,
            baudrate=BAUDRATE,
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
        logger.info("串口打开成功")

    def read_byte(self) -> Optional[int]:
        data = self.ser.read(1)
        return data[0] if data else None

    def send_class(self, raw_class: str) -> bytes:
        if raw_class not in CLASS_TO_MCU_CODE:
            raise ValueError(f"未知分类，无法发送给 52RC: {raw_class}")
        code = CLASS_TO_MCU_CODE[raw_class]
        packet = bytes([FRAME_HEAD, code, FRAME_TAIL])
        self.ser.write(packet)
        self.ser.flush()
        self.logger.info("发送分类结果: %s -> %s", raw_class, hex_str(packet))
        return packet

    def close(self):
        if getattr(self, "ser", None) and self.ser.is_open:
            self.ser.close()
            self.logger.info("串口已关闭")


class CameraManager:
    def __init__(self, camera_index: int, logger: logging.Logger):
        self.logger = logger
        logger.info("打开摄像头 index=%d", camera_index)
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开摄像头: {camera_index}")
        for _ in range(10):
            self.cap.read()
            time.sleep(0.03)
        logger.info("摄像头打开成功")

    def read(self) -> np.ndarray:
        ret, frame = self.cap.read()
        if not ret or frame is None:
            raise RuntimeError("摄像头读取失败")
        return frame

    def close(self):
        if getattr(self, "cap", None):
            self.cap.release()
            self.logger.info("摄像头已关闭")


# =========================================================
# 8. 主系统
# =========================================================

class FinalGarbageSortingSystem:
    def __init__(self, args):
        self.project_root = Path(args.project_root).expanduser().resolve()
        self.model_path = Path(args.model_path).expanduser().resolve() if args.model_path else self.project_root / "export" / "latest_tflite_fp16.tflite"
        self.mapping_path = Path(args.mapping_path).expanduser().resolve() if args.mapping_path else self.project_root / "config" / "class_mapping.json"
        self.log_dir = self.project_root / "Logs"
        self.capture_dir = self.project_root / "Captures_Final"
        self.runtime_csv = self.log_dir / "final_rounds.csv"
        self.stats_file = self.log_dir / "final_stats.json"
        self.show_window = not args.no_window
        self.predict_timeout_sec = args.predict_timeout
        self.min_fallback_conf = args.min_fallback_conf
        self.uncertain_policy = args.uncertain_policy
        self.stable_frames = args.stable_frames
        self.conf_threshold = args.conf_threshold

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.capture_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger(self.log_dir)
        self.logger.info("PROJECT_ROOT: %s", self.project_root)
        self.logger.info("MODEL_PATH: %s", self.model_path)
        self.logger.info("CLASS_MAPPING_PATH: %s", self.mapping_path)

        idx_to_class = load_idx_to_class(self.mapping_path, self.logger)
        self.classifier = GarbageClassifierTFLite(self.model_path, idx_to_class, self.logger)
        self.serial_mgr = SerialManager(args.serial_port, self.logger)
        self.camera_mgr = CameraManager(args.camera_index, self.logger)
        self.stable_predictor = StablePredictor(self.conf_threshold, self.stable_frames)

        self.stats = self.load_stats()
        self.running = True
        self.round_id = 0
        self.last_trigger_time = 0.0
        self.fps = 0.0
        self.last_frame_time = time.time()

        self.status_text = "摄像头已打开，等待 52RC 发送 0xA1"
        self.class_text = "-"
        self.conf_text = "-"
        self.stable_text = "-"
        self.serial_text = "等待 0xA1"

        if self.show_window:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW_NAME, CAPTURE_WIDTH, CAPTURE_HEIGHT)

    def load_stats(self) -> Dict:
        if self.stats_file.exists():
            try:
                return json.loads(self.stats_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {
            "total_rounds": 0,
            "success_rounds": 0,
            "failed_rounds": 0,
            "uncertain_rounds": 0,
            "ack_success": 0,
            "done_success": 0,
            "class_counts": {"可回收垃圾": 0, "厨余垃圾": 0, "有害垃圾": 0, "其他垃圾": 0},
            "last_update": None,
        }

    def save_stats(self):
        self.stats["last_update"] = now_str()
        self.stats_file.write_text(json.dumps(self.stats, ensure_ascii=False, indent=4), encoding="utf-8")

    def append_round_log(self, row: Dict):
        file_exists = self.runtime_csv.exists()
        fieldnames = [
            "time", "round_id", "trigger_rx", "status", "pred_id", "raw_class",
            "display_class", "confidence", "stable_count", "is_stable", "mcu_code",
            "tx_packet", "ack_received", "done_received", "snapshot_path", "elapsed_sec",
            "probs", "message"
        ]
        with self.runtime_csv.open("a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def update_fps(self):
        now = time.time()
        dt = now - self.last_frame_time
        self.last_frame_time = now
        if dt > 0:
            current = 1.0 / dt
            self.fps = 0.9 * self.fps + 0.1 * current if self.fps > 0 else current

    def show_frame(self, frame: np.ndarray, roi_box, color=(0, 255, 0)):
        if not self.show_window:
            return
        show = draw_overlay(
            frame, roi_box, self.status_text, self.class_text, self.conf_text,
            self.stable_text, self.serial_text, str(self.round_id), f"{self.fps:.1f}", color
        )
        cv2.imshow(WINDOW_NAME, show)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            self.running = False
        elif key == ord("s"):
            self.save_snapshot(show, None, "manual")

    def save_snapshot(self, frame: np.ndarray, result: Optional[Dict], tag: str) -> str:
        ts = timestamp_str()
        if result:
            filename = f"round_{self.round_id:04d}_{result.get('raw_class', 'unknown')}_{result.get('confidence', 0.0):.3f}_{tag}_{ts}.jpg"
        else:
            filename = f"round_{self.round_id:04d}_{tag}_{ts}.jpg"
        path = self.capture_dir / filename
        cv2.imwrite(str(path), frame)
        self.logger.info("保存截图: %s", path)
        return str(path)

    def wait_for_byte_with_preview(self, target_byte: int, timeout_sec: float, wait_name: str) -> bool:
        start = time.time()
        while self.running and time.time() - start < timeout_sec:
            value = self.serial_mgr.read_byte()
            if value is not None:
                self.logger.info("收到串口字节: 0x%02X", value)
                self.serial_text = f"RX 0x{value:02X}"
                if value == target_byte:
                    return True
                if value == MCU_ERROR:
                    self.logger.warning("收到 52RC 错误字节 0xEE")
                    return False

            frame = self.camera_mgr.read()
            self.update_fps()
            roi_box = get_center_roi(frame)
            self.status_text = f"等待 {wait_name}"
            self.show_frame(frame, roi_box, color=(0, 255, 255))
            time.sleep(FRAME_INTERVAL_SEC)
        return False

    def classify_once_after_trigger(self) -> Tuple[Dict, Dict, np.ndarray, Tuple[int, int, int, int]]:
        self.stable_predictor.reset()
        start = time.time()
        latest_frame = None
        latest_roi_box = None

        while self.running and time.time() - start < self.predict_timeout_sec:
            loop_start = time.time()
            frame = self.camera_mgr.read()
            latest_frame = frame
            self.update_fps()
            roi_box = get_center_roi(frame)
            latest_roi_box = roi_box
            x1, y1, x2, y2 = roi_box
            roi = frame[y1:y2, x1:x2]

            result = self.classifier.predict(roi)
            stable_info = self.stable_predictor.update(result)

            self.status_text = stable_info["status"]
            self.class_text = f"{result['display_class']} ({result['raw_class']})"
            self.conf_text = f"{result['confidence']:.3f}"
            self.stable_text = f"{stable_info['stable_count']} / {self.stable_frames}"
            self.serial_text = "识别中"
            self.show_frame(frame, roi_box, color=stable_info["color"])

            if stable_info["is_stable"]:
                self.logger.info("识别稳定: %s, conf=%.3f", result["display_class"], result["confidence"])
                return result, stable_info, frame, roi_box

            sleep_time = FRAME_INTERVAL_SEC - (time.time() - loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

        best_result = self.stable_predictor.best_result
        self.logger.warning("识别超时，策略: %s", self.uncertain_policy)

        if self.uncertain_policy == "send_best" and best_result and best_result["confidence"] >= self.min_fallback_conf:
            return best_result, {
                "is_stable": False,
                "stable_count": self.stable_predictor.stable_count,
                "status": "超时发送最高置信度",
                "color": (0, 165, 255),
            }, latest_frame, latest_roi_box

        if self.uncertain_policy == "send_other":
            return {
                "pred_id": -1,
                "raw_class": "其他",
                "display_class": "其他垃圾",
                "confidence": 0.0 if best_result is None else best_result.get("confidence", 0.0),
                "probs": [] if best_result is None else best_result.get("probs", []),
            }, {
                "is_stable": False,
                "stable_count": 0,
                "status": "超时默认其他",
                "color": (0, 165, 255),
            }, latest_frame, latest_roi_box

        raise RuntimeError("识别超时且未发送分类结果")

    def run_one_round(self):
        self.round_id += 1
        round_start = time.time()
        self.stats["total_rounds"] += 1
        self.logger.info("========== Round %04d 开始 ==========", self.round_id)

        result = None
        stable_info = None
        snapshot_path = ""
        tx_packet = b""
        ack_received = False
        done_received = False
        status = "failed"
        message = ""

        try:
            self.status_text = "收到 0xA1，开始识别"
            self.serial_text = "RX 0xA1"
            self.class_text = "-"
            self.conf_text = "-"
            self.stable_text = "-"

            result, stable_info, frame, roi_box = self.classify_once_after_trigger()
            if frame is None or roi_box is None:
                raise RuntimeError("识别结束时没有有效画面")

            raw_class = result["raw_class"]
            display_class = result["display_class"]
            confidence = result["confidence"]
            if raw_class not in CLASS_TO_MCU_CODE:
                self.logger.warning("模型返回未知类别 %s，强制映射为其他", raw_class)
                raw_class = "其他"
                result["raw_class"] = "其他"
                result["display_class"] = "其他垃圾"
                display_class = "其他垃圾"

            self.status_text = "识别完成，准备发送"
            self.class_text = f"{display_class} ({raw_class})"
            self.conf_text = f"{confidence:.3f}"
            self.stable_text = f"{stable_info['stable_count']} / {self.stable_frames}"

            show = draw_overlay(
                frame, roi_box, self.status_text, self.class_text, self.conf_text,
                self.stable_text, "准备发送", str(self.round_id), f"{self.fps:.1f}", stable_info["color"]
            )
            snapshot_path = self.save_snapshot(show, result, "before_send")

            tx_packet = self.serial_mgr.send_class(raw_class)
            self.serial_text = f"TX {hex_str(tx_packet)}"

            ack_received = self.wait_for_byte_with_preview(MCU_ACK_RECEIVED, ACK_TIMEOUT_SEC, "0xCC ACK")
            if ack_received:
                self.stats["ack_success"] += 1
                self.logger.info("收到 0xCC：52RC 已确认分类结果")
            else:
                message = "ack timeout or error"
                raise RuntimeError(message)

            done_received = self.wait_for_byte_with_preview(MCU_DONE, DONE_TIMEOUT_SEC, "0xDD DONE")
            if done_received:
                self.stats["done_success"] += 1
                self.stats["success_rounds"] += 1
                status = "success"
                self.logger.info("收到 0xDD：52RC 分拣动作完成")
            else:
                message = "done timeout or error"
                raise RuntimeError(message)

            if not stable_info["is_stable"]:
                self.stats["uncertain_rounds"] += 1
            if display_class in self.stats["class_counts"]:
                self.stats["class_counts"][display_class] += 1

            self.status_text = "本轮结束，等待下一次 0xA1"
            self.serial_text = "等待 0xA1"

        except Exception as exc:
            self.logger.error("本轮异常: %s", exc)
            self.logger.error(traceback.format_exc())
            self.stats["failed_rounds"] += 1
            status = "failed"
            if not message:
                message = str(exc)

        finally:
            elapsed = time.time() - round_start
            self.save_stats()
            self.append_round_log({
                "time": now_str(),
                "round_id": self.round_id,
                "trigger_rx": "0xA1",
                "status": status,
                "pred_id": "" if result is None else result.get("pred_id", ""),
                "raw_class": "" if result is None else result.get("raw_class", ""),
                "display_class": "" if result is None else result.get("display_class", ""),
                "confidence": "" if result is None else f"{result.get('confidence', 0.0):.6f}",
                "stable_count": "" if stable_info is None else stable_info.get("stable_count", ""),
                "is_stable": "" if stable_info is None else int(stable_info.get("is_stable", False)),
                "mcu_code": "" if result is None else CLASS_TO_MCU_CODE.get(result.get("raw_class", ""), ""),
                "tx_packet": hex_str(tx_packet) if tx_packet else "",
                "ack_received": int(ack_received),
                "done_received": int(done_received),
                "snapshot_path": snapshot_path,
                "elapsed_sec": f"{elapsed:.3f}",
                "probs": "" if result is None else json.dumps(result.get("probs", []), ensure_ascii=False),
                "message": message,
            })
            self.logger.info("========== Round %04d 结束，status=%s, elapsed=%.2fs ==========", self.round_id, status, elapsed)

    def run(self):
        self.logger.info("========== 最终系统进入运行状态 ==========")
        self.logger.info("等待 52RC 发送 0xA1。窗口中按 q 退出，按 s 手动截图。")
        try:
            while self.running:
                frame = self.camera_mgr.read()
                self.update_fps()
                roi_box = get_center_roi(frame)

                value = self.serial_mgr.read_byte()
                if value is not None:
                    self.logger.info("收到串口字节: 0x%02X", value)
                    if value == MCU_TRIGGER_READY:
                        now = time.time()
                        if now - self.last_trigger_time < TRIGGER_DEBOUNCE_SEC:
                            self.logger.info("忽略过近的重复 0xA1")
                            continue
                        self.last_trigger_time = now
                        self.run_one_round()
                        continue
                    if value == MCU_ERROR:
                        self.serial_text = "RX 0xEE"
                        self.status_text = "52RC 报告错误"
                    else:
                        self.serial_text = f"RX 0x{value:02X}"
                        self.status_text = "收到未知串口字节"
                else:
                    self.status_text = "摄像头预览中，等待 52RC 发送 0xA1"
                    self.serial_text = "等待 0xA1"

                self.class_text = "-"
                self.conf_text = "-"
                self.stable_text = "-"
                self.show_frame(frame, roi_box, color=(255, 255, 0))
                time.sleep(FRAME_INTERVAL_SEC)

        except KeyboardInterrupt:
            self.logger.info("用户 Ctrl+C 退出")
        except Exception as exc:
            self.logger.error("系统异常: %s", exc)
            self.logger.error(traceback.format_exc())
        finally:
            self.close()

    def close(self):
        try:
            self.save_stats()
        except Exception:
            pass
        try:
            self.camera_mgr.close()
        except Exception:
            pass
        try:
            self.serial_mgr.close()
        except Exception:
            pass
        if self.show_window:
            cv2.destroyAllWindows()
        self.logger.info("最终系统已关闭")
        self.logger.info("轮次日志: %s", self.runtime_csv)
        self.logger.info("统计文件: %s", self.stats_file)


# =========================================================
# 9. 启动入口
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser(description="树莓派垃圾分类最终上位机脚本")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT_DEFAULT), help="项目根目录")
    parser.add_argument("--model-path", default="", help="TFLite 模型路径；为空则使用 project-root/export/latest_tflite_fp16.tflite")
    parser.add_argument("--mapping-path", default="", help="class_mapping.json 路径；为空则使用 project-root/config/class_mapping.json")
    parser.add_argument("--serial-port", default=SERIAL_PORT_DEFAULT, help="树莓派串口，例如 /dev/ttyAMA0 或 /dev/ttyUSB0")
    parser.add_argument("--camera-index", type=int, default=CAMERA_INDEX_DEFAULT, help="摄像头编号")
    parser.add_argument("--no-window", action="store_true", help="无桌面/SSH 环境使用，不显示 OpenCV 窗口")
    parser.add_argument("--conf-threshold", type=float, default=CONF_THRESHOLD_DEFAULT, help="稳定识别置信度阈值")
    parser.add_argument("--stable-frames", type=int, default=STABLE_FRAMES_DEFAULT, help="连续稳定帧数")
    parser.add_argument("--predict-timeout", type=float, default=PREDICT_TIMEOUT_SEC_DEFAULT, help="每轮识别最长时间，秒")
    parser.add_argument("--uncertain-policy", choices=["send_best", "send_other", "skip"], default=UNCERTAIN_POLICY_DEFAULT, help="识别不稳定时的处理策略")
    parser.add_argument("--min-fallback-conf", type=float, default=MIN_FALLBACK_CONF_DEFAULT, help="send_best 策略下最低可接受置信度")
    return parser.parse_args()


def main():
    args = parse_args()
    system = FinalGarbageSortingSystem(args)
    system.run()


if __name__ == "__main__":
    main()
