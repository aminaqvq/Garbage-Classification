import os

# =========================================================
# 0. 环境设置：必须放在 import cv2 之前
# =========================================================

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


import cv2
import csv
import json
import time
import serial
import logging
import traceback
import numpy as np

from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont


# =========================================================
# 1. TFLite 解释器导入
# =========================================================

try:
    from tflite_runtime.interpreter import Interpreter
    TFLITE_BACKEND = "tflite_runtime"
except ImportError:
    try:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
        TFLITE_BACKEND = "tensorflow"
    except ImportError as e:
        raise RuntimeError(
            "未找到 TFLite 解释器。\n"
            "树莓派推荐安装：python3 -m pip install tflite-runtime"
        ) from e


# =========================================================
# 2. 项目路径配置：全部使用绝对路径
# =========================================================

PROJECT_ROOT = Path("/home/amina/workspaces/Garbage Classification")

MODEL_PATH = PROJECT_ROOT / "export" / "latest_tflite_fp16.tflite"
CLASS_MAPPING_PATH = PROJECT_ROOT / "class_mapping.json"

LOG_DIR = PROJECT_ROOT / "Logs"
CAPTURE_DIR = PROJECT_ROOT / "Captures_Final"

RUNTIME_LOG_CSV = LOG_DIR / "final_rounds.csv"
STATS_FILE = LOG_DIR / "final_stats.json"


# =========================================================
# 3. 硬件配置
# =========================================================

SERIAL_PORT = "/dev/ttyAMA0"
BAUDRATE = 9600
SERIAL_TIMEOUT = 0.02

CAMERA_INDEX = 0
CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 480

SHOW_WINDOW = True
WINDOW_NAME = "Garbage Final Sorting System"


# =========================================================
# 4. 模型预处理配置：跟当前 FP16 导出模型保持一致
# =========================================================

RESIZE_SIZE = 256
CROP_SIZE = 224
RGB_INPUT = True

NORMALIZE = True
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# =========================================================
# 5. 识别流程配置
# =========================================================

CONF_THRESHOLD = 0.80
STABLE_FRAMES = 3
PREDICT_TIMEOUT_SEC = 4.0
FRAME_INTERVAL_SEC = 0.03

# 如果超时仍不稳定：
# send_best：发送当前最高置信度结果
# send_other：发送“其他”
# skip：不发送，让 52RC 超时
UNCERTAIN_POLICY = "send_best"

# 低于这个置信度才允许被认为完全不可信
MIN_FALLBACK_CONF = 0.55

TRIGGER_DEBOUNCE_SEC = 0.8
ACK_TIMEOUT_SEC = 2.0
DONE_TIMEOUT_SEC = 8.0

# ROI 引导框比例
ROI_W_RATIO = 0.56
ROI_H_RATIO = 0.70


# =========================================================
# 6. 串口协议
# =========================================================

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


# =========================================================
# 7. 字体配置
# =========================================================

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]

FONT_SIZE_MAIN = 26
FONT_SIZE_SMALL = 21


# =========================================================
# 8. 日志系统
# =========================================================

def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def timestamp_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def setup_logger():
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    log_path = LOG_DIR / f"final_runtime_{timestamp_str()}.log"

    logger = logging.getLogger("GarbageFinalSystem")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.info("最终系统启动")
    logger.info("日志文件: %s", log_path)
    logger.info("TFLite 后端: %s", TFLITE_BACKEND)

    return logger


logger = setup_logger()


def hex_str(data: bytes) -> str:
    return " ".join(f"{b:02X}" for b in data)


def append_round_log(row: Dict):
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    file_exists = RUNTIME_LOG_CSV.exists()

    fieldnames = [
        "time",
        "round_id",
        "trigger_rx",
        "status",
        "pred_id",
        "raw_class",
        "display_class",
        "confidence",
        "stable_count",
        "is_stable",
        "mcu_code",
        "tx_packet",
        "ack_received",
        "done_received",
        "snapshot_path",
        "elapsed_sec",
        "probs",
        "message"
    ]

    with open(RUNTIME_LOG_CSV, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


# =========================================================
# 9. 统计文件
# =========================================================

def load_stats():
    if STATS_FILE.exists():
        try:
            with open(STATS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    return {
        "total_rounds": 0,
        "success_rounds": 0,
        "failed_rounds": 0,
        "uncertain_rounds": 0,
        "ack_success": 0,
        "done_success": 0,
        "class_counts": {
            "可回收垃圾": 0,
            "厨余垃圾": 0,
            "有害垃圾": 0,
            "其他垃圾": 0
        },
        "last_update": None
    }


def save_stats(stats):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stats["last_update"] = now_str()

    with open(STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=4)


# =========================================================
# 10. 类别映射
# =========================================================

def load_idx_to_class() -> Dict[int, str]:
    if CLASS_MAPPING_PATH.exists():
        try:
            with open(CLASS_MAPPING_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)

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

        except Exception as e:
            logger.warning("读取 class_mapping.json 失败，将使用默认映射: %s", e)

    logger.info("使用默认类别映射: %s", DEFAULT_IDX_TO_CLASS)
    return DEFAULT_IDX_TO_CLASS.copy()


def get_display_name(raw_class: str) -> str:
    return CLASS_DISPLAY_NAME.get(raw_class, raw_class)


# =========================================================
# 11. 图像与 TFLite 工具
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
    line_gap: int = 32,
    font: Optional[ImageFont.FreeTypeFont] = None
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
    roi_box,
    status_text: str,
    class_text: str,
    conf_text: str,
    stable_text: str,
    serial_text: str,
    round_text: str,
    fps_text: str,
    color=(0, 255, 0)
) -> np.ndarray:
    x1, y1, x2, y2 = roi_box

    show = frame_bgr.copy()

    cv2.rectangle(show, (x1, y1), (x2, y2), color, 2)

    overlay = show.copy()
    cv2.rectangle(overlay, (10, 10), (610, 245), (0, 0, 0), -1)
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

    if FONT_MAIN is None:
        y = 35
        for line in lines:
            safe_line = line.encode("ascii", errors="replace").decode("ascii")
            cv2.putText(show, safe_line, (22, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 0), 2)
            y += 30

        cv2.putText(show, "ROI", (x1 + 8, y1 - 8 if y1 > 25 else y1 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.70, color, 2)
        return show

    show = draw_text_pil(
        show,
        lines,
        x=22,
        y=22,
        color_rgb=(255, 255, 255),
        line_gap=31,
        font=FONT_SMALL or FONT_MAIN
    )

    show = draw_text_pil(
        show,
        ["ROI"],
        x=x1 + 8,
        y=y1 - 30 if y1 > 35 else y1 + 8,
        color_rgb=(color[2], color[1], color[0]),
        line_gap=30,
        font=FONT_SMALL or FONT_MAIN
    )

    return show


# =========================================================
# 12. 分类器
# =========================================================

class GarbageClassifierTFLite:
    def __init__(self, model_path: Path, idx_to_class: Dict[int, str]):
        if not model_path.exists():
            raise FileNotFoundError(f"TFLite 模型不存在: {model_path}")

        self.idx_to_class = idx_to_class

        logger.info("加载 TFLite 模型: %s", model_path)

        self.interpreter = Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()

        self.input_detail = self.interpreter.get_input_details()[0]
        self.output_detail = self.interpreter.get_output_details()[0]

        self.input_shape = list(self.input_detail["shape"])
        self.output_shape = list(self.output_detail["shape"])
        self.input_dtype = self.input_detail["dtype"]
        self.output_dtype = self.output_detail["dtype"]
        self.input_layout = infer_layout_from_shape(self.input_shape)

        logger.info("模型输入 shape: %s", self.input_shape)
        logger.info("模型输出 shape: %s", self.output_shape)
        logger.info("输入 dtype: %s", self.input_dtype)
        logger.info("输出 dtype: %s", self.output_dtype)
        logger.info("输入布局: %s", self.input_layout)
        logger.info("类别映射: %s", self.idx_to_class)

        self.warmup()

    def warmup(self):
        logger.info("模型预热中...")

        if self.input_layout == "NCHW":
            dummy = np.zeros((1, 3, CROP_SIZE, CROP_SIZE), dtype=np.float32)
        else:
            dummy = np.zeros((1, CROP_SIZE, CROP_SIZE, 3), dtype=np.float32)

        dummy = quantize_tensor_if_needed(dummy, self.input_detail)

        self.interpreter.set_tensor(self.input_detail["index"], dummy)
        self.interpreter.invoke()
        _ = self.interpreter.get_tensor(self.output_detail["index"])

        logger.info("模型预热完成")

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
            "probs": probs.tolist()
        }


# =========================================================
# 13. 稳定识别器
# =========================================================

class StablePredictor:
    def __init__(self):
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

        if conf < CONF_THRESHOLD:
            self.stable_count = 0
            self.last_pred_id = None
            return {
                "is_stable": False,
                "stable_count": 0,
                "status": "低置信度",
                "color": (0, 165, 255)
            }

        if pred_id == self.last_pred_id:
            self.stable_count += 1
        else:
            self.last_pred_id = pred_id
            self.stable_count = 1

        is_stable = self.stable_count >= STABLE_FRAMES

        return {
            "is_stable": is_stable,
            "stable_count": self.stable_count,
            "status": "识别稳定" if is_stable else "识别中",
            "color": (0, 255, 0) if is_stable else (255, 255, 0)
        }


# =========================================================
# 14. 串口管理
# =========================================================

class SerialManager:
    def __init__(self):
        logger.info("打开串口: %s, baud=%d", SERIAL_PORT, BAUDRATE)

        self.ser = serial.Serial(
            port=SERIAL_PORT,
            baudrate=BAUDRATE,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=SERIAL_TIMEOUT,
            xonxoff=False,
            rtscts=False,
            dsrdtr=False
        )

        time.sleep(1.0)
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()

        logger.info("串口打开成功")

    def read_byte(self) -> Optional[int]:
        data = self.ser.read(1)
        if data:
            return data[0]
        return None

    def send_class(self, raw_class: str) -> bytes:
        if raw_class not in CLASS_TO_MCU_CODE:
            raise ValueError(f"未知分类，无法发送给 52RC: {raw_class}")

        code = CLASS_TO_MCU_CODE[raw_class]
        packet = bytes([FRAME_HEAD, code, FRAME_TAIL])

        self.ser.write(packet)
        self.ser.flush()

        logger.info("发送分类结果: %s -> %s", raw_class, hex_str(packet))

        return packet

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
            logger.info("串口已关闭")


# =========================================================
# 15. 摄像头管理
# =========================================================

class CameraManager:
    def __init__(self):
        logger.info("打开摄像头 index=%d", CAMERA_INDEX)

        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)

        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开摄像头: {CAMERA_INDEX}")

        for _ in range(10):
            self.cap.read()
            time.sleep(0.03)

        logger.info("摄像头打开成功")

    def read(self):
        ret, frame = self.cap.read()

        if not ret or frame is None:
            raise RuntimeError("摄像头读取失败")

        return frame

    def close(self):
        if self.cap:
            self.cap.release()
            logger.info("摄像头已关闭")


# =========================================================
# 16. 主系统
# =========================================================

class FinalGarbageSortingSystem:
    def __init__(self):
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        CAPTURE_DIR.mkdir(parents=True, exist_ok=True)

        logger.info("PROJECT_ROOT: %s", PROJECT_ROOT)
        logger.info("MODEL_PATH: %s", MODEL_PATH)
        logger.info("CLASS_MAPPING_PATH: %s", CLASS_MAPPING_PATH)
        logger.info("NORMALIZE: %s", NORMALIZE)

        idx_to_class = load_idx_to_class()

        self.classifier = GarbageClassifierTFLite(MODEL_PATH, idx_to_class)
        self.serial_mgr = SerialManager()
        self.camera_mgr = CameraManager()
        self.stable_predictor = StablePredictor()

        self.stats = load_stats()

        self.running = True
        self.round_id = 0
        self.last_trigger_time = 0.0

        self.status_text = "摄像头已打开，等待 52RC 发送 0xA1"
        self.class_text = "-"
        self.conf_text = "-"
        self.stable_text = "-"
        self.serial_text = "等待 0xA1"
        self.fps = 0.0
        self.last_frame_time = time.time()

        if SHOW_WINDOW:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW_NAME, CAPTURE_WIDTH, CAPTURE_HEIGHT)

        if FONT_MAIN is None:
            logger.warning("未找到中文字体，画面显示可能降级为 ASCII")
        else:
            logger.info("中文字体加载成功")

    def update_fps(self):
        now = time.time()
        dt = now - self.last_frame_time
        self.last_frame_time = now

        if dt > 0:
            current_fps = 1.0 / dt
            self.fps = 0.9 * self.fps + 0.1 * current_fps if self.fps > 0 else current_fps

    def show_frame(self, frame, roi_box, color=(0, 255, 0)):
        if not SHOW_WINDOW:
            return

        show = draw_overlay(
            frame_bgr=frame,
            roi_box=roi_box,
            status_text=self.status_text,
            class_text=self.class_text,
            conf_text=self.conf_text,
            stable_text=self.stable_text,
            serial_text=self.serial_text,
            round_text=str(self.round_id),
            fps_text=f"{self.fps:.1f}",
            color=color
        )

        cv2.imshow(WINDOW_NAME, show)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            self.running = False

        if key == ord("s"):
            self.save_snapshot(show, None, "manual")

    def save_snapshot(self, frame, result: Optional[Dict], tag: str) -> str:
        CAPTURE_DIR.mkdir(parents=True, exist_ok=True)

        ts = timestamp_str()

        if result:
            raw_class = result.get("raw_class", "unknown")
            conf = result.get("confidence", 0.0)
            filename = f"round_{self.round_id:04d}_{raw_class}_{conf:.3f}_{tag}_{ts}.jpg"
        else:
            filename = f"round_{self.round_id:04d}_{tag}_{ts}.jpg"

        path = CAPTURE_DIR / filename
        cv2.imwrite(str(path), frame)

        logger.info("保存截图: %s", path)

        return str(path)

    def wait_for_byte_with_preview(self, target_byte: int, timeout_sec: float, wait_name: str) -> bool:
        start = time.time()

        while self.running and time.time() - start < timeout_sec:
            value = self.serial_mgr.read_byte()

            if value is not None:
                logger.info("收到串口字节: 0x%02X", value)
                self.serial_text = f"RX 0x{value:02X}"

                if value == target_byte:
                    return True

                if value == MCU_ERROR:
                    logger.warning("收到 52RC 错误字节 0xEE")
                    return False

            frame = self.camera_mgr.read()
            self.update_fps()
            roi_box = get_center_roi(frame)

            self.status_text = f"等待 {wait_name}"
            self.show_frame(frame, roi_box, color=(0, 255, 255))

        return False

    def classify_once_after_trigger(self) -> Tuple[Dict, Dict, np.ndarray, Tuple[int, int, int, int]]:
        self.stable_predictor.reset()

        start = time.time()
        latest_frame = None
        latest_roi_box = None
        latest_result = None
        latest_stable_info = None

        while self.running and time.time() - start < PREDICT_TIMEOUT_SEC:
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

            latest_result = result
            latest_stable_info = stable_info

            self.status_text = stable_info["status"]
            self.class_text = f"{result['display_class']} ({result['raw_class']})"
            self.conf_text = f"{result['confidence']:.3f}"
            self.stable_text = f"{stable_info['stable_count']} / {STABLE_FRAMES}"
            self.serial_text = "识别中"

            self.show_frame(frame, roi_box, color=stable_info["color"])

            if stable_info["is_stable"]:
                logger.info(
                    "识别稳定: %s, conf=%.3f",
                    result["display_class"],
                    result["confidence"]
                )
                return result, stable_info, frame, roi_box

            elapsed = time.time() - loop_start
            sleep_time = FRAME_INTERVAL_SEC - elapsed

            if sleep_time > 0:
                time.sleep(sleep_time)

        logger.warning("识别超时，进入不稳定策略: %s", UNCERTAIN_POLICY)

        best_result = self.stable_predictor.best_result

        if UNCERTAIN_POLICY == "send_best" and best_result and best_result["confidence"] >= MIN_FALLBACK_CONF:
            fallback_info = {
                "is_stable": False,
                "stable_count": self.stable_predictor.stable_count,
                "status": "超时发送最高置信度",
                "color": (0, 165, 255)
            }
            return best_result, fallback_info, latest_frame, latest_roi_box

        if UNCERTAIN_POLICY == "send_other":
            fallback_result = {
                "pred_id": -1,
                "raw_class": "其他",
                "display_class": "其他垃圾",
                "confidence": 0.0 if best_result is None else best_result["confidence"],
                "probs": [] if best_result is None else best_result.get("probs", [])
            }
            fallback_info = {
                "is_stable": False,
                "stable_count": 0,
                "status": "超时默认其他",
                "color": (0, 165, 255)
            }
            return fallback_result, fallback_info, latest_frame, latest_roi_box

        raise RuntimeError("识别超时且策略为 skip，本轮不发送分类结果")

    def run_one_round(self):
        self.round_id += 1
        round_start = time.time()

        self.stats["total_rounds"] += 1

        logger.info("========== Round %04d 开始 ==========", self.round_id)

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

            raw_class = result["raw_class"]
            display_class = result["display_class"]
            confidence = result["confidence"]

            self.status_text = "识别完成，准备发送"
            self.class_text = f"{display_class} ({raw_class})"
            self.conf_text = f"{confidence:.3f}"
            self.stable_text = f"{stable_info['stable_count']} / {STABLE_FRAMES}"

            show = draw_overlay(
                frame_bgr=frame,
                roi_box=roi_box,
                status_text=self.status_text,
                class_text=self.class_text,
                conf_text=self.conf_text,
                stable_text=self.stable_text,
                serial_text="准备发送",
                round_text=str(self.round_id),
                fps_text=f"{self.fps:.1f}",
                color=stable_info["color"]
            )

            snapshot_path = self.save_snapshot(show, result, "before_send")

            tx_packet = self.serial_mgr.send_class(raw_class)
            self.serial_text = f"TX {hex_str(tx_packet)}"

            ack_received = self.wait_for_byte_with_preview(
                MCU_ACK_RECEIVED,
                ACK_TIMEOUT_SEC,
                "0xCC"
            )

            if ack_received:
                logger.info("收到 0xCC：52RC 已确认收到分类结果")
                self.stats["ack_success"] += 1
            else:
                logger.warning("未收到 0xCC")
                message = "ack timeout"

            done_received = self.wait_for_byte_with_preview(
                MCU_DONE,
                DONE_TIMEOUT_SEC,
                "0xDD"
            )

            if done_received:
                logger.info("收到 0xDD：52RC 动作完成")
                self.stats["done_success"] += 1
                status = "success"
                self.stats["success_rounds"] += 1
            else:
                logger.warning("未收到 0xDD")
                message = "done timeout"
                self.stats["failed_rounds"] += 1

            if not stable_info["is_stable"]:
                self.stats["uncertain_rounds"] += 1

            if display_class in self.stats["class_counts"]:
                self.stats["class_counts"][display_class] += 1

            self.status_text = "本轮结束，等待下一次 0xA1"
            self.serial_text = "等待 0xA1"

        except Exception as e:
            logger.error("本轮异常: %s", e)
            logger.error(traceback.format_exc())
            self.stats["failed_rounds"] += 1
            message = str(e)
            status = "failed"

        finally:
            elapsed = time.time() - round_start
            save_stats(self.stats)

            append_round_log({
                "time": now_str(),
                "round_id": self.round_id,
                "trigger_rx": "0xA1",
                "status": status,
                "pred_id": "" if result is None else result.get("pred_id", ""),
                "raw_class": "" if result is None else result.get("raw_class", ""),
                "display_class": "" if result is None else result.get("display_class", ""),
                "confidence": "" if result is None else result.get("confidence", ""),
                "stable_count": "" if stable_info is None else stable_info.get("stable_count", ""),
                "is_stable": "" if stable_info is None else int(stable_info.get("is_stable", False)),
                "mcu_code": "" if result is None else CLASS_TO_MCU_CODE.get(result.get("raw_class", ""), ""),
                "tx_packet": hex_str(tx_packet) if tx_packet else "",
                "ack_received": int(ack_received),
                "done_received": int(done_received),
                "snapshot_path": snapshot_path,
                "elapsed_sec": f"{elapsed:.3f}",
                "probs": "" if result is None else json.dumps(result.get("probs", []), ensure_ascii=False),
                "message": message
            })

            logger.info("========== Round %04d 结束，status=%s, elapsed=%.2fs ==========",
                        self.round_id, status, elapsed)

    def run(self):
        logger.info("========== 最终系统进入运行状态 ==========")
        logger.info("等待 52RC 发送 0xA1。按 q 退出，按 s 手动截图。")

        try:
            while self.running:
                frame = self.camera_mgr.read()
                self.update_fps()
                roi_box = get_center_roi(frame)

                value = self.serial_mgr.read_byte()

                if value is not None:
                    logger.info("收到串口字节: 0x%02X", value)

                    if value == MCU_TRIGGER_READY:
                        now = time.time()

                        if now - self.last_trigger_time < TRIGGER_DEBOUNCE_SEC:
                            logger.info("忽略过近的重复 0xA1")
                            continue

                        self.last_trigger_time = now
                        self.run_one_round()
                        continue

                    elif value == MCU_ERROR:
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
            logger.info("用户 Ctrl+C 退出")

        except Exception as e:
            logger.error("系统异常: %s", e)
            logger.error(traceback.format_exc())

        finally:
            self.close()

    def close(self):
        save_stats(self.stats)

        try:
            self.camera_mgr.close()
        except Exception:
            pass

        try:
            self.serial_mgr.close()
        except Exception:
            pass

        if SHOW_WINDOW:
            cv2.destroyAllWindows()

        logger.info("最终系统已关闭")
        logger.info("轮次日志: %s", RUNTIME_LOG_CSV)
        logger.info("统计文件: %s", STATS_FILE)


def main():
    system = FinalGarbageSortingSystem()
    system.run()


if __name__ == "__main__":
    main()