from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 尽量放在 import cv2 前，减少树莓派桌面 Qt 插件问题
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

try:
    import serial
except ImportError as exc:
    raise RuntimeError("缺少 pyserial，请先执行：python3 -m pip install pyserial") from exc

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:
    raise RuntimeError("缺少 Pillow，请先执行：python3 -m pip install pillow") from exc


# =========================================================
# 1. 默认配置
# =========================================================

PROJECT_ROOT = Path("/home/amina/workspaces/Garbage Classification")
DEFAULT_MODEL_PATH = PROJECT_ROOT / "export" / "latest_tflite_fp16.tflite"
DEFAULT_CLASS_MAPPING_PATH = PROJECT_ROOT / "class_mapping.json"
DEFAULT_LOG_DIR = PROJECT_ROOT / "Logs"
DEFAULT_CAPTURE_DIR = PROJECT_ROOT / "Captures_Locked_Trigger"

DEFAULT_SERIAL_PORT = "/dev/ttyAMA0"
DEFAULT_BAUDRATE = 9600
DEFAULT_SERIAL_TIMEOUT = 0.02

DEFAULT_CAMERA_INDEX = 0
DEFAULT_CAPTURE_WIDTH = 640
DEFAULT_CAPTURE_HEIGHT = 480

# 与你之前模型保持一致
RESIZE_SIZE = 256
CROP_SIZE = 224
RGB_INPUT = True
NORMALIZE = True
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

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

CLASS_TO_SERVO_CHAR = {
    "可回收": "R",
    "厨余": "K",
    "有害": "H",
    "其他": "O",
}

VALID_TX_CHARS = {"R", "K", "H", "O"}
MCU_TRIGGER_CHAR = "T"
MCU_ACK_CHAR = "A"
MCU_DONE_CHAR = "D"
MCU_ERROR_CHAR = "E"

WINDOW_NAME = "Garbage Sorting - Locked Trigger v4"

# 中文字体候选：解决 OpenCV 中文显示问号。窗口中文字全部用 PIL 绘制。
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/arphic/ukai.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
]


# =========================================================
# 2. 通用工具函数
# =========================================================

def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def timestamp_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_idx_to_class(path: Path) -> Dict[int, str]:
    """优先读取 class_mapping.json；失败则使用默认映射。"""
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            idx_to_class = data.get("idx_to_class")
            if isinstance(idx_to_class, dict):
                return {int(k): str(v) for k, v in idx_to_class.items()}

            class_to_idx = data.get("class_to_idx")
            if isinstance(class_to_idx, dict):
                return {int(v): str(k) for k, v in class_to_idx.items()}

        except Exception as exc:
            print(f"[WARN] 读取类别映射失败，将使用默认映射：{exc}")

    return DEFAULT_IDX_TO_CLASS.copy()


def get_display_name(raw_class: str) -> str:
    return CLASS_DISPLAY_NAME.get(raw_class, raw_class)


def class_to_char(raw_class: str) -> str:
    if raw_class not in CLASS_TO_SERVO_CHAR:
        raise ValueError(f"未知类别，无法转换为 R/K/H/O：{raw_class}")
    return CLASS_TO_SERVO_CHAR[raw_class]


def append_csv_log(log_file: Path, row: Dict) -> None:
    ensure_dir(log_file.parent)
    exists = log_file.exists()
    fieldnames = [
        "time", "event", "state", "mode", "round_id", "frame_id",
        "pred_id", "raw_class", "display_class", "confidence",
        "stable_count", "is_stable", "tx_char", "rx_char",
        "serial_port", "snapshot_path", "message",
    ]
    with open(log_file, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def load_chinese_font(font_path: Optional[str], size: int) -> Optional[ImageFont.FreeTypeFont]:
    candidates: List[str] = []
    if font_path:
        candidates.append(font_path)
    candidates.extend(FONT_CANDIDATES)
    for path in candidates:
        if path and os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return None


def hex_of_char(ch: str) -> str:
    return f"0x{ord(ch):02X}" if ch else ""


# =========================================================
# 3. 串口管理
# =========================================================

class LockedCharSerial:
    def __init__(self, port: str, baudrate: int, timeout: float):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser: Optional[serial.Serial] = None

    def open(self) -> None:
        print(f"[SERIAL] 打开串口：{self.port}, baud={self.baudrate}")
        self.ser = serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=self.timeout,
            xonxoff=False,
            rtscts=False,
            dsrdtr=False,
        )
        time.sleep(1.0)
        self.reset_buffers()
        print("[SERIAL] 串口打开成功，已清空输入/输出缓冲区")

    def reset_buffers(self) -> None:
        if self.ser is None:
            return
        try:
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
        except Exception:
            pass

    def drain_input(self, duration_sec: float = 0.08) -> List[str]:
        """清掉动作期间或冷却期间可能残留的重复 T/噪声。"""
        drained: List[str] = []
        start = time.time()
        while time.time() - start < duration_sec:
            ch = self.read_char()
            if ch is None:
                continue
            drained.append(ch)
        if drained:
            print(f"[SERIAL] 已清理残留字符：{drained}")
        return drained

    def read_char(self) -> Optional[str]:
        if self.ser is None or not self.ser.is_open:
            raise RuntimeError("串口未打开")
        data = self.ser.read(1)
        if not data:
            return None
        try:
            ch = data.decode("ascii", errors="ignore")
        except Exception:
            return None
        return ch if ch else None

    def send_char(self, ch: str) -> None:
        ch = ch.upper().strip()
        if ch not in VALID_TX_CHARS:
            raise ValueError(f"只能发送 R/K/H/O，当前为：{ch}")
        if self.ser is None or not self.ser.is_open:
            raise RuntimeError("串口未打开")
        data = ch.encode("ascii")
        self.ser.write(data)
        self.ser.flush()
        print(f"[TX] 发送一次分类字符：{ch}  hex={data[0]:02X}")

    def close(self) -> None:
        if self.ser is not None and self.ser.is_open:
            self.ser.close()
            print("[SERIAL] 串口已关闭")


# =========================================================
# 4. TFLite 与图像预处理
# =========================================================

def lazy_import_cv2_np_tflite():
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("缺少 OpenCV 或 numpy，请先安装：python3 -m pip install opencv-python numpy") from exc

    try:
        from tflite_runtime.interpreter import Interpreter
        backend = "tflite_runtime"
    except ImportError:
        try:
            import tensorflow as tf # pyright: ignore[reportMissingModuleSource]
            Interpreter = tf.lite.Interpreter
            backend = "tensorflow"
        except ImportError as exc:
            raise RuntimeError(
                "没有找到 TFLite 解释器。\n"
                "树莓派建议：python3 -m pip install tflite-runtime\n"
                "或安装完整 TensorFlow：python3 -m pip install tensorflow"
            ) from exc
    return cv2, np, Interpreter, backend


def softmax_np(np, x):
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


def quantize_tensor_if_needed(np, x_float32, detail):
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


def dequantize_tensor_if_needed(np, y, detail):
    dtype = detail["dtype"]
    scale, zero_point = detail.get("quantization", (0.0, 0))
    if dtype == np.float32 or scale in (0.0, None):
        return y.astype(np.float32)
    return (y.astype(np.float32) - int(zero_point)) * float(scale)


class GarbageClassifierTFLite:
    def __init__(self, model_path: Path, idx_to_class: Dict[int, str]):
        self.cv2, self.np, Interpreter, backend = lazy_import_cv2_np_tflite()
        self.backend = backend
        if not model_path.exists():
            raise FileNotFoundError(f"TFLite 模型不存在：{model_path}")
        self.model_path = model_path
        self.idx_to_class = idx_to_class

        print(f"[AI] 加载模型：{model_path}")
        print(f"[AI] TFLite 后端：{backend}")
        print(f"[AI] 类别映射：{self.idx_to_class}")

        self.interpreter = Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()
        self.input_detail = self.interpreter.get_input_details()[0]
        self.output_detail = self.interpreter.get_output_details()[0]
        self.input_shape = list(self.input_detail["shape"])
        self.output_shape = list(self.output_detail["shape"])
        self.input_layout = infer_layout_from_shape(self.input_shape)
        print(f"[AI] input_shape={self.input_shape}, output_shape={self.output_shape}, layout={self.input_layout}")
        self.warmup()

    def warmup(self) -> None:
        np = self.np
        if self.input_layout == "NCHW":
            dummy = np.zeros((1, 3, CROP_SIZE, CROP_SIZE), dtype=np.float32)
        else:
            dummy = np.zeros((1, CROP_SIZE, CROP_SIZE, 3), dtype=np.float32)
        dummy = quantize_tensor_if_needed(np, dummy, self.input_detail)
        self.interpreter.set_tensor(self.input_detail["index"], dummy)
        self.interpreter.invoke()
        _ = self.interpreter.get_tensor(self.output_detail["index"])
        print("[AI] 模型预热完成")

    def resize_keep_ratio(self, img_rgb, shorter_side: int):
        cv2 = self.cv2
        h, w = img_rgb.shape[:2]
        if w < h:
            new_w = shorter_side
            new_h = int(round(h * shorter_side / w))
        else:
            new_h = shorter_side
            new_w = int(round(w * shorter_side / h))
        return cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    def center_crop(self, img_rgb, crop_size: int):
        cv2 = self.cv2
        h, w = img_rgb.shape[:2]
        left = max(0, int(round((w - crop_size) / 2)))
        top = max(0, int(round((h - crop_size) / 2)))
        crop = img_rgb[top:top + crop_size, left:left + crop_size]
        if crop.shape[0] != crop_size or crop.shape[1] != crop_size:
            crop = cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
        return crop

    def preprocess(self, frame_bgr):
        cv2 = self.cv2
        np = self.np
        if RGB_INPUT:
            img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        else:
            img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = self.resize_keep_ratio(img, RESIZE_SIZE)
        img = self.center_crop(img, CROP_SIZE)
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
        return quantize_tensor_if_needed(np, x.astype(np.float32), self.input_detail)

    def predict(self, frame_bgr) -> Dict:
        np = self.np
        x = self.preprocess(frame_bgr)
        self.interpreter.set_tensor(self.input_detail["index"], x)
        self.interpreter.invoke()
        y = self.interpreter.get_tensor(self.output_detail["index"])
        y = dequantize_tensor_if_needed(np, y, self.output_detail)
        logits = y.reshape(-1).astype(np.float32)
        probs = softmax_np(np, logits)
        pred_id = int(np.argmax(probs))
        confidence = float(probs[pred_id])
        raw_class = self.idx_to_class.get(pred_id, str(pred_id))
        display_class = get_display_name(raw_class)
        tx_char = CLASS_TO_SERVO_CHAR.get(raw_class, "?")
        return {
            "pred_id": pred_id,
            "raw_class": raw_class,
            "display_class": display_class,
            "confidence": confidence,
            "tx_char": tx_char,
            "probs": probs.tolist(),
        }


# =========================================================
# 5. 稳定帧判断
# =========================================================

@dataclass
class StableState:
    is_stable: bool
    stable_count: int
    status: str


class StablePredictor:
    def __init__(self, conf_threshold: float, stable_frames: int):
        self.conf_threshold = conf_threshold
        self.stable_frames = stable_frames
        self.last_pred_id: Optional[int] = None
        self.stable_count = 0
        self.best_result: Optional[Dict] = None
        self.best_conf = -1.0

    def reset(self) -> None:
        self.last_pred_id = None
        self.stable_count = 0
        self.best_result = None
        self.best_conf = -1.0

    def update(self, result: Dict) -> StableState:
        pred_id = int(result["pred_id"])
        conf = float(result["confidence"])
        if conf > self.best_conf:
            self.best_conf = conf
            self.best_result = result
        if conf < self.conf_threshold:
            self.last_pred_id = None
            self.stable_count = 0
            return StableState(False, 0, "低置信度")
        if pred_id == self.last_pred_id:
            self.stable_count += 1
        else:
            self.last_pred_id = pred_id
            self.stable_count = 1
        is_stable = self.stable_count >= self.stable_frames
        return StableState(is_stable, self.stable_count, "识别稳定" if is_stable else "识别中")


# =========================================================
# 6. 主应用：严格锁定状态机
# =========================================================

class LockedTriggerApp:
    def __init__(self, args):
        self.args = args
        self.log_file = Path(args.log_dir) / "locked_trigger_runtime_log.csv"
        self.capture_dir = Path(args.capture_dir)
        ensure_dir(Path(args.log_dir))
        ensure_dir(self.capture_dir)

        self.serial_mgr = LockedCharSerial(args.serial_port, args.baudrate, args.serial_timeout)
        self.serial_mgr.open()

        idx_to_class = load_idx_to_class(Path(args.class_mapping_path))
        self.classifier = GarbageClassifierTFLite(Path(args.model_path), idx_to_class)
        self.cv2 = self.classifier.cv2
        self.np = self.classifier.np
        self.stable_predictor = StablePredictor(args.conf_threshold, args.stable_frames)

        self.font_main = load_chinese_font(args.font_path, args.font_size)
        self.font_small = load_chinese_font(args.font_path, max(16, args.font_size - 6))
        if self.font_main is None:
            print("[WARN] 未找到中文字体，画面中文可能显示异常。建议安装：sudo apt install fonts-wqy-zenhei fonts-noto-cjk")
        else:
            print("[UI] 中文字体加载成功，画面中文不会再用 cv2.putText 绘制")

        self.cap = None
        self.running = True
        self.round_id = 0
        self.frame_id = 0
        self.state = "WAIT_TRIGGER"
        self.status_text = "等待超声波 T，不进行 AI 识别"
        self.last_rx_char = ""
        self.last_tx_char = ""
        self.last_result: Optional[Dict] = None
        self.last_state: Optional[StableState] = None
        self.last_done_status = ""

    # ------------------------- 摄像头 -------------------------

    def open_camera(self) -> None:
        print(f"[CAMERA] 打开摄像头 index={self.args.camera_index}")
        self.cap = self.cv2.VideoCapture(self.args.camera_index)
        self.cap.set(self.cv2.CAP_PROP_FRAME_WIDTH, self.args.capture_width)
        self.cap.set(self.cv2.CAP_PROP_FRAME_HEIGHT, self.args.capture_height)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开摄像头：{self.args.camera_index}")
        for _ in range(8):
            self.cap.read()
            time.sleep(0.03)
        if not self.args.no_window:
            self.cv2.namedWindow(WINDOW_NAME, self.cv2.WINDOW_NORMAL)
            self.cv2.resizeWindow(WINDOW_NAME, self.args.capture_width, self.args.capture_height)
        print("[CAMERA] 摄像头打开成功")

    def read_frame(self):
        if self.cap is None:
            raise RuntimeError("摄像头未打开")
        ret, frame = self.cap.read()
        if not ret or frame is None:
            raise RuntimeError("摄像头读取失败")
        self.frame_id += 1
        return frame

    # ------------------------- 显示 -------------------------

    def draw_chinese_lines(self, frame_bgr, lines: List[str], x: int, y: int, font, color=(255, 255, 255), gap: int = 30):
        cv2 = self.cv2
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(img)
        py = y
        for line in lines:
            draw.text((x, py), line, font=font, fill=color)
            py += gap
        return cv2.cvtColor(self.np.array(img), cv2.COLOR_RGB2BGR)

    def draw_overlay(self, frame_bgr, result: Optional[Dict] = None, stable: Optional[StableState] = None, extra: str = ""):
        cv2 = self.cv2
        show = frame_bgr.copy()
        h, w = show.shape[:2]

        # ROI 提示框，只用于指导摆放，不影响识别；识别仍使用整帧中心预处理
        roi_w = int(w * 0.56)
        roi_h = int(h * 0.70)
        x1 = max((w - roi_w) // 2, 0)
        y1 = max((h - roi_h) // 2, 0)
        x2 = min(x1 + roi_w, w)
        y2 = min(y1 + roi_h, h)
        cv2.rectangle(show, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 不绘制背景面板，文本直接透明叠加到画面上

        if result:
            cls = result.get("display_class", "-")
            conf = f"{float(result.get('confidence', 0.0)):.3f}"
            tx = result.get("tx_char", "-")
        else:
            cls = "-"
            conf = "-"
            tx = "-"

        stable_text = "-"
        if stable:
            stable_text = f"{stable.status} / {stable.stable_count}/{self.args.stable_frames}"

        lines = [
            f"状态：{self.status_text}",
            f"状态机：{self.state}",
            f"类别：{cls}    置信度：{conf}    待发：{tx}",
            f"稳定：{stable_text}",
            f"串口：RX={self.last_rx_char or '-'}  TX={self.last_tx_char or '-'}  完成={self.last_done_status or '-'}",
            f"轮次：{self.round_id}",
        ]
        if extra:
            lines.append(f"提示：{extra}")

        if self.font_main:
            show = self.draw_chinese_lines(show, lines, 22, 22, self.font_small or self.font_main, gap=30)
            show = self.draw_chinese_lines(show, ["ROI"], x1 + 8, y1 - 28 if y1 > 35 else y1 + 8, self.font_small or self.font_main, color=(0, 255, 0), gap=24)
        else:
            # 兜底：不再显示中文，避免问号刷屏
            ascii_lines = [
                f"STATE: {self.state}",
                f"CLASS: {result.get('raw_class', '-') if result else '-'} CONF: {conf} TX: {tx}",
                f"SERIAL RX={self.last_rx_char or '-'} TX={self.last_tx_char or '-'} DONE={self.last_done_status or '-'}",
            ]
            y = 36
            for line in ascii_lines:
                cv2.putText(show, line, (22, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                y += 32
        return show

    def show_and_handle_key(self, show, result: Optional[Dict] = None) -> None:
        if self.args.no_window:
            return
        self.cv2.imshow(WINDOW_NAME, show)
        key = self.cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            self.running = False
        elif key == ord("s"):
            self.save_snapshot(show, result, "manual")

    # ------------------------- 日志/截图 -------------------------

    def save_snapshot(self, frame, result: Optional[Dict], tag: str) -> str:
        ensure_dir(self.capture_dir)
        if result:
            raw = result.get("raw_class", "unknown")
            conf = float(result.get("confidence", 0.0))
            name = f"round_{self.round_id:04d}_{raw}_{conf:.3f}_{tag}_{timestamp_str()}.jpg"
        else:
            name = f"round_{self.round_id:04d}_{tag}_{timestamp_str()}.jpg"
        path = self.capture_dir / name
        self.cv2.imwrite(str(path), frame)
        print(f"[CAPTURE] 保存截图：{path}")
        return str(path)

    def log_event(self, event: str, message: str = "", result: Optional[Dict] = None, stable: Optional[StableState] = None, snapshot_path: str = "") -> None:
        append_csv_log(self.log_file, {
            "time": now_str(),
            "event": event,
            "state": self.state,
            "mode": self.args.mode,
            "round_id": self.round_id,
            "frame_id": self.frame_id,
            "pred_id": result.get("pred_id") if result else "",
            "raw_class": result.get("raw_class") if result else "",
            "display_class": result.get("display_class") if result else "",
            "confidence": f"{float(result.get('confidence', 0.0)):.4f}" if result else "",
            "stable_count": stable.stable_count if stable else "",
            "is_stable": "1" if stable and stable.is_stable else "0",
            "tx_char": result.get("tx_char") if result else self.last_tx_char,
            "rx_char": self.last_rx_char,
            "serial_port": self.args.serial_port,
            "snapshot_path": snapshot_path,
            "message": message,
        })

    # ------------------------- 核心流程 -------------------------

    def choose_fallback_result(self) -> Optional[Dict]:
        best = self.stable_predictor.best_result
        if best is None:
            return None
        if self.args.uncertain_policy == "skip":
            return None
        if self.args.uncertain_policy == "other":
            return {
                "pred_id": -1,
                "raw_class": "其他",
                "display_class": get_display_name("其他"),
                "confidence": 0.0,
                "tx_char": "O",
                "probs": [],
            }
        # best：置信度足够就发 best，否则发其他，避免完全无结果
        if float(best.get("confidence", 0.0)) >= self.args.min_fallback_conf:
            return best
        return {
            "pred_id": -1,
            "raw_class": "其他",
            "display_class": get_display_name("其他"),
            "confidence": 0.0,
            "tx_char": "O",
            "probs": [],
        }

    def wait_done_locked(self) -> str:
        """发送后暂停识别，只等下位机 D/E。期间只显示画面，不做 AI 推理。"""
        self.state = "SENT_WAIT_DONE"
        self.status_text = "已发送分类，暂停识别，等待下位机 D/E"
        start = time.time()
        while self.running and time.time() - start < self.args.done_timeout:
            ch = self.serial_mgr.read_char()
            if ch is not None:
                self.last_rx_char = ch
                print(f"[RX] 等待完成期间收到：{repr(ch)} {hex_of_char(ch)}")
                if ch == MCU_DONE_CHAR:
                    self.last_done_status = "D 完成"
                    self.log_event("done", "received D")
                    return "done"
                if ch == MCU_ERROR_CHAR:
                    self.last_done_status = "E 错误"
                    self.log_event("mcu_error", "received E")
                    return "error"
                if ch == MCU_ACK_CHAR:
                    self.last_done_status = "A 已确认"
                    # ACK 不结束本轮，继续等 D
                elif ch == MCU_TRIGGER_CHAR:
                    # 动作期间重复 T 一律忽略，避免误触发下一轮
                    self.log_event("ignored_trigger_while_busy", "ignored T while waiting done")
                else:
                    self.log_event("ignored_char_while_busy", f"ignored {repr(ch)} while waiting done")

            try:
                frame = self.read_frame()
                show = self.draw_overlay(frame, self.last_result, self.last_state, extra="等待 D/E 期间不做识别")
                self.show_and_handle_key(show, self.last_result)
            except Exception:
                # 无窗口或偶发摄像头问题时，仍继续等串口完成
                pass
            time.sleep(self.args.frame_interval)

        self.last_done_status = "等待 D 超时"
        self.log_event("done_timeout", "timeout waiting D/E")
        return "timeout"

    def run_prediction_round(self) -> None:
        """收到 T 后只执行一轮识别，只发送一次分类字符。"""
        self.round_id += 1
        self.state = "PREDICTING"
        self.status_text = "收到 T，正在进行本轮识别"
        self.last_done_status = ""
        self.last_tx_char = ""
        self.last_result = None
        self.last_state = None
        self.stable_predictor.reset()
        self.serial_mgr.drain_input(0.05)

        print(f"\n[ROUND {self.round_id}] 收到 T，开始一轮识别")
        self.log_event("round_start", "trigger T received")

        start = time.time()
        final_result: Optional[Dict] = None
        final_state: Optional[StableState] = None
        sent = False

        while self.running:
            elapsed = time.time() - start
            if elapsed > self.args.max_predict_sec:
                print(f"[ROUND {self.round_id}] 识别超时，使用 fallback 策略：{self.args.uncertain_policy}")
                final_result = self.choose_fallback_result()
                final_state = StableState(False, self.stable_predictor.stable_count, "超时 fallback")
                break

            frame = self.read_frame()
            result = self.classifier.predict(frame)
            state = self.stable_predictor.update(result)
            self.last_result = result
            self.last_state = state

            self.status_text = "本轮识别中"
            show = self.draw_overlay(frame, result, state)
            self.show_and_handle_key(show, result)

            print(
                f"[AI] round={self.round_id} class={result['display_class']} "
                f"conf={result['confidence']:.3f} stable={state.stable_count}/{self.args.stable_frames} tx={result['tx_char']}"
            )

            # 同时满足：置信度/稳定帧达标 + 最短观察时间到达，才发送
            if state.is_stable and elapsed >= self.args.min_predict_sec:
                final_result = result
                final_state = state
                break

            time.sleep(self.args.frame_interval)

        if final_result is None:
            self.status_text = "本轮未获得可发送结果，跳过"
            print(f"[ROUND {self.round_id}] 未获得有效结果，本轮不发送")
            self.log_event("round_skip", "no valid result", self.last_result, self.last_state)
            self.state = "COOLDOWN"
            self.serial_mgr.drain_input(self.args.cooldown_sec)
            self.state = "WAIT_TRIGGER"
            self.status_text = "等待超声波 T，不进行 AI 识别"
            return

        tx_char = final_result.get("tx_char", "?")
        if tx_char not in VALID_TX_CHARS:
            # 极端异常兜底
            tx_char = "O"
            final_result = dict(final_result)
            final_result["tx_char"] = tx_char
            final_result["raw_class"] = "其他"
            final_result["display_class"] = get_display_name("其他")

        # 本轮只允许发送一次
        if not sent:
            self.serial_mgr.send_char(tx_char)
            self.last_tx_char = tx_char
            sent = True
            snapshot_path = ""
            if self.args.save_on_send:
                try:
                    frame = self.read_frame()
                    show = self.draw_overlay(frame, final_result, final_state, extra=f"已发送 {tx_char}")
                    snapshot_path = self.save_snapshot(show, final_result, "sent")
                except Exception:
                    snapshot_path = ""
            self.log_event("tx_class_once", f"sent {tx_char} once", final_result, final_state, snapshot_path)

        done_status = self.wait_done_locked()
        print(f"[ROUND {self.round_id}] 本轮结束：{done_status}")

        self.state = "COOLDOWN"
        self.status_text = "本轮结束，清理残留串口并冷却"
        self.serial_mgr.drain_input(self.args.cooldown_sec)
        self.stable_predictor.reset()
        self.last_result = None
        self.last_state = None
        self.state = "WAIT_TRIGGER"
        self.status_text = "等待超声波 T，不进行 AI 识别"

    def run_trigger_mode(self) -> None:
        print("\n[READY] 锁定触发模式：等待下位机 T；未收到 T 时完全不做 AI 识别")
        self.state = "WAIT_TRIGGER"
        self.status_text = "等待超声波 T，不进行 AI 识别"
        self.serial_mgr.drain_input(0.15)

        while self.running:
            ch = self.serial_mgr.read_char()
            if ch is not None:
                self.last_rx_char = ch
                print(f"[RX] 空闲等待期间收到：{repr(ch)} {hex_of_char(ch)}")
                self.log_event("rx_char", f"received {repr(ch)}")
                if ch == MCU_TRIGGER_CHAR:
                    self.run_prediction_round()
                    continue
                elif ch == MCU_DONE_CHAR:
                    # 可能是上一轮遗留，空闲期收到 D 直接忽略
                    self.last_done_status = "空闲期收到遗留 D，已忽略"
                elif ch == MCU_ERROR_CHAR:
                    self.last_done_status = "空闲期收到 E"

            # 等待 T 时只显示画面，不预测
            try:
                frame = self.read_frame()
                show = self.draw_overlay(frame, None, None, extra="空闲期不识别，避免干扰舵机动作")
                self.show_and_handle_key(show, None)
            except Exception:
                pass
            time.sleep(self.args.idle_frame_interval)

    def run_manual_mode(self) -> None:
        print("\n[READY] 手动模式：按 r/k/h/o 发送；不会自动识别发送")
        self.state = "MANUAL"
        self.status_text = "手动模式：r/k/h/o 直接发送"
        while self.running:
            frame = self.read_frame()
            show = self.draw_overlay(frame, None, None)
            if not self.args.no_window:
                self.cv2.imshow(WINDOW_NAME, show)
                key = self.cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    self.running = False
                elif key in (ord("r"), ord("k"), ord("h"), ord("o")):
                    ch = chr(key).upper()
                    self.serial_mgr.send_char(ch)
                    self.last_tx_char = ch
                    self.log_event("manual_tx", f"manual sent {ch}")
                elif key == ord("s"):
                    self.save_snapshot(show, None, "manual")
            time.sleep(self.args.frame_interval)

    def run(self) -> None:
        self.open_camera()
        try:
            if self.args.mode == "trigger":
                self.run_trigger_mode()
            elif self.args.mode == "manual":
                self.run_manual_mode()
            else:
                raise ValueError(f"未知模式：{self.args.mode}")
        except KeyboardInterrupt:
            print("\n[EXIT] 用户 Ctrl+C 退出")
        except Exception:
            print("[ERROR] 主程序异常：")
            traceback.print_exc()
        finally:
            self.close()

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
        if not self.args.no_window:
            try:
                self.cv2.destroyAllWindows()
            except Exception:
                pass
        self.serial_mgr.close()


# =========================================================
# 7. 命令行入口
# =========================================================

def run_test_char(args) -> None:
    ch = args.test_char.upper().strip()
    if ch not in VALID_TX_CHARS:
        raise ValueError("--test-char 只能是 R/K/H/O")
    mgr = LockedCharSerial(args.serial_port, args.baudrate, args.serial_timeout)
    mgr.open()
    try:
        mgr.send_char(ch)
        print(f"[TEST] 已发送 {ch}，如果下位机烧录了舵机程序，应看到舵机动作")
    finally:
        mgr.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="树莓派垃圾分类上位机：超声波触发 + 单字符协议 + 锁定状态机 v4")
    parser.add_argument("--mode", choices=["trigger", "manual"], default="trigger", help="运行模式，默认 trigger")
    parser.add_argument("--test-char", choices=["R", "K", "H", "O", "r", "k", "h", "o"], default=None, help="只发送一个测试字符后退出")

    parser.add_argument("--serial-port", default=DEFAULT_SERIAL_PORT)
    parser.add_argument("--baudrate", type=int, default=DEFAULT_BAUDRATE)
    parser.add_argument("--serial-timeout", type=float, default=DEFAULT_SERIAL_TIMEOUT)

    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--class-mapping-path", default=str(DEFAULT_CLASS_MAPPING_PATH))
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR))
    parser.add_argument("--capture-dir", default=str(DEFAULT_CAPTURE_DIR))

    parser.add_argument("--camera-index", type=int, default=DEFAULT_CAMERA_INDEX)
    parser.add_argument("--capture-width", type=int, default=DEFAULT_CAPTURE_WIDTH)
    parser.add_argument("--capture-height", type=int, default=DEFAULT_CAPTURE_HEIGHT)
    parser.add_argument("--no-window", action="store_true", help="无桌面/SSH 运行，不显示窗口")

    parser.add_argument("--conf-threshold", type=float, default=0.80, help="稳定识别置信度阈值")
    parser.add_argument("--stable-frames", type=int, default=4, help="连续稳定帧数，建议 3~6")
    parser.add_argument("--min-predict-sec", type=float, default=0.80, help="收到 T 后最短识别时间，避免刚触发就误判")
    parser.add_argument("--max-predict-sec", type=float, default=4.00, help="最长识别时间，超时按 fallback 策略处理")
    parser.add_argument("--uncertain-policy", choices=["best", "other", "skip"], default="best", help="超时未稳定时的处理策略")
    parser.add_argument("--min-fallback-conf", type=float, default=0.55, help="best fallback 的最低置信度；低于则发 O")
    parser.add_argument("--done-timeout", type=float, default=10.0, help="发送分类后等待 D/E 的最长时间")
    parser.add_argument("--cooldown-sec", type=float, default=0.50, help="本轮结束后清串口残留/冷却时间")
    parser.add_argument("--frame-interval", type=float, default=0.03)
    parser.add_argument("--idle-frame-interval", type=float, default=0.06, help="等待 T 时画面刷新间隔，不做 AI")
    parser.add_argument("--save-on-send", action="store_true", default=True, help="发送分类时保存截图")
    parser.add_argument("--font-path", default=None, help="手动指定中文字体路径")
    parser.add_argument("--font-size", type=int, default=24)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.test_char:
        run_test_char(args)
        return
    app = LockedTriggerApp(args)
    app.run()


if __name__ == "__main__":
    main()
