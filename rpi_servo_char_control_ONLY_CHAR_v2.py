#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rpi_servo_char_control_ONLY_CHAR_v2.py

用途：
    树莓派上位机脚本，专门配合你当前“不修改”的《舵机控制.c》使用。

通信方式：
    不使用 AA xx 55 帧协议。
    不等待 0xA1 / 0xCC / 0xDD。
    识别稳定后，只向 52RC/8051 串口发送 1 个 ASCII 字符：R / K / H / O。

下位机《舵机控制.c》对应关系：
    R -> 可回收 -> angle_pwm = 8
    K -> 厨余   -> angle_pwm = 29
    H -> 有害   -> angle_pwm = 19
    O -> 其他   -> angle_pwm = 36

推荐先做串口直测：
    python3 rpi_servo_char_control_ONLY_CHAR_v2.py --test-char R
    python3 rpi_servo_char_control_ONLY_CHAR_v2.py --test-char K
    python3 rpi_servo_char_control_ONLY_CHAR_v2.py --test-char H
    python3 rpi_servo_char_control_ONLY_CHAR_v2.py --test-char O

正式自动识别：
    python3 rpi_servo_char_control_ONLY_CHAR_v2.py --mode auto

手动辅助模式：
    python3 rpi_servo_char_control_ONLY_CHAR_v2.py --mode manual
    按键：r/k/h/o 直接发字符；空格发送当前识别结果；s 截图；q 退出。
"""

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

# 必须尽量放在 import cv2 前面，避免树莓派桌面环境 Qt 插件问题
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

try:
    import serial
except ImportError as exc:
    raise RuntimeError("缺少 pyserial，请先执行：python3 -m pip install pyserial") from exc


# =========================================================
# 1. 默认配置
# =========================================================

PROJECT_ROOT = Path("/home/amina/workspaces/Garbage Classification")
DEFAULT_MODEL_PATH = PROJECT_ROOT / "export" / "latest_tflite_fp16.tflite"
DEFAULT_CLASS_MAPPING_PATH = PROJECT_ROOT / "class_mapping.json"
DEFAULT_LOG_DIR = PROJECT_ROOT / "Logs"
DEFAULT_CAPTURE_DIR = PROJECT_ROOT / "Captures_Servo_Char"

DEFAULT_SERIAL_PORT = "/dev/ttyAMA0"
DEFAULT_BAUDRATE = 9600
DEFAULT_SERIAL_TIMEOUT = 0.05

DEFAULT_CAMERA_INDEX = 0
DEFAULT_CAPTURE_WIDTH = 640
DEFAULT_CAPTURE_HEIGHT = 480

# 模型预处理参数：与你之前导出的 FP16 TFLite 模型保持一致
RESIZE_SIZE = 256
CROP_SIZE = 224
RGB_INPUT = True
NORMALIZE = True
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

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

# 这是本脚本最关键的地方：迎合《舵机控制.c》的单字符协议
CLASS_TO_SERVO_CHAR = {
    "可回收": "R",
    "厨余": "K",
    "有害": "H",
    "其他": "O",
}

VALID_TEST_CHARS = {"R", "K", "H", "O"}
WINDOW_NAME = "Servo Char Garbage Sorting"


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
    file_exists = log_file.exists()

    fieldnames = [
        "time",
        "event",
        "mode",
        "frame_id",
        "pred_id",
        "raw_class",
        "display_class",
        "confidence",
        "stable_count",
        "is_stable",
        "tx_char",
        "serial_port",
        "message",
    ]

    with open(log_file, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


# =========================================================
# 3. 串口管理：只发送单个字符 R/K/H/O
# =========================================================

class ServoCharSerial:
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
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()
        print("[SERIAL] 串口打开成功")

    def send_char(self, ch: str) -> None:
        if ch not in VALID_TEST_CHARS:
            raise ValueError(f"只能发送 R/K/H/O，当前为：{ch}")
        if self.ser is None or not self.ser.is_open:
            raise RuntimeError("串口未打开")

        # 注意：这里严格只发送一个 ASCII 字节，不加换行、不加帧头帧尾
        data = ch.encode("ascii")
        self.ser.write(data)
        self.ser.flush()
        print(f"[TX] 已发送字符：{ch}  hex=0x{data[0]:02X}")

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
            import tensorflow as tf
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
            return StableState(False, 0, "LOW_CONF")

        if pred_id == self.last_pred_id:
            self.stable_count += 1
        else:
            self.last_pred_id = pred_id
            self.stable_count = 1

        is_stable = self.stable_count >= self.stable_frames
        return StableState(is_stable, self.stable_count, "STABLE" if is_stable else "PREDICTING")


# =========================================================
# 6. 摄像头运行主循环
# =========================================================

class ServoCharApp:
    def __init__(self, args):
        self.args = args
        self.log_file = Path(args.log_dir) / "servo_char_runtime_log.csv"
        self.capture_dir = Path(args.capture_dir)
        ensure_dir(Path(args.log_dir))
        ensure_dir(self.capture_dir)

        self.idx_to_class = load_idx_to_class(Path(args.class_mapping))
        self.classifier = GarbageClassifierTFLite(Path(args.model_path), self.idx_to_class)
        self.cv2 = self.classifier.cv2

        self.serial_mgr = ServoCharSerial(args.serial_port, args.baudrate, args.serial_timeout)
        self.serial_mgr.open()

        self.stable_predictor = StablePredictor(args.conf_threshold, args.stable_frames)
        self.frame_id = 0
        self.last_send_time = 0.0
        self.last_sent_char: Optional[str] = None
        self.running = True

        self.cap = None

    def open_camera(self) -> None:
        cv2 = self.cv2
        print(f"[CAMERA] 打开摄像头 index={self.args.camera_index}")
        self.cap = cv2.VideoCapture(self.args.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.args.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.args.height)

        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开摄像头：{self.args.camera_index}")

        for _ in range(8):
            self.cap.read()
            time.sleep(0.03)

        if not self.args.no_window:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW_NAME, self.args.width, self.args.height)

        print("[CAMERA] 摄像头打开成功")

    def can_send_now(self, ch: str) -> bool:
        now = time.time()
        if now - self.last_send_time < self.args.send_cooldown:
            return False
        if self.args.no_repeat_same and ch == self.last_sent_char:
            return False
        return True

    def send_prediction(self, result: Dict, reason: str) -> bool:
        ch = result["tx_char"]
        if ch not in VALID_TEST_CHARS:
            print(f"[WARN] 识别类别无法映射到 R/K/H/O：{result}")
            return False

        self.serial_mgr.send_char(ch)
        self.last_send_time = time.time()
        self.last_sent_char = ch

        append_csv_log(self.log_file, {
            "time": now_str(),
            "event": "send_char",
            "mode": self.args.mode,
            "frame_id": self.frame_id,
            "pred_id": result.get("pred_id"),
            "raw_class": result.get("raw_class"),
            "display_class": result.get("display_class"),
            "confidence": f"{result.get('confidence', 0.0):.4f}",
            "stable_count": self.stable_predictor.stable_count,
            "is_stable": True,
            "tx_char": ch,
            "serial_port": self.args.serial_port,
            "message": reason,
        })
        return True

    def draw_overlay(self, frame, result: Optional[Dict], state: Optional[StableState]):
        cv2 = self.cv2
        show = frame.copy()

        if result is None:
            lines = [
                "Waiting for camera frame...",
                f"Mode: {self.args.mode}",
            ]
        else:
            stable_text = "-" if state is None else f"{state.stable_count}/{self.args.stable_frames} {state.status}"
            cooldown_left = max(0.0, self.args.send_cooldown - (time.time() - self.last_send_time))
            lines = [
                "SERVO CHAR MODE: send only R/K/H/O",
                f"Class: {result['display_class']}  raw={result['raw_class']}",
                f"Conf: {result['confidence']:.3f}  Stable: {stable_text}",
                f"TX char: {result['tx_char']}  Last TX: {self.last_sent_char or '-'}",
                f"Cooldown: {cooldown_left:.1f}s",
                "Keys: r/k/h/o send, SPACE send pred, s save, q quit",
            ]

        x, y = 12, 28
        for line in lines:
            cv2.putText(show, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
            cv2.putText(show, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 1)
            y += 28
        return show

    def save_snapshot(self, frame, result: Optional[Dict], tag: str) -> str:
        if result:
            filename = f"frame_{self.frame_id:06d}_{result['raw_class']}_{result['confidence']:.3f}_{tag}_{timestamp_str()}.jpg"
        else:
            filename = f"frame_{self.frame_id:06d}_{tag}_{timestamp_str()}.jpg"
        path = self.capture_dir / filename
        self.cv2.imwrite(str(path), frame)
        print(f"[CAPTURE] 已保存：{path}")
        return str(path)

    def handle_key(self, key: int, current_result: Optional[Dict], current_show) -> None:
        if key < 0:
            return

        key_char = chr(key).lower() if 0 <= key < 256 else ""

        if key_char == "q":
            self.running = False
            return

        if key_char == "s":
            self.save_snapshot(current_show, current_result, "manual")
            return

        manual_map = {"r": "R", "k": "K", "h": "H", "o": "O"}
        if key_char in manual_map:
            ch = manual_map[key_char]
            self.serial_mgr.send_char(ch)
            self.last_send_time = time.time()
            self.last_sent_char = ch
            append_csv_log(self.log_file, {
                "time": now_str(),
                "event": "manual_key_send",
                "mode": self.args.mode,
                "frame_id": self.frame_id,
                "tx_char": ch,
                "serial_port": self.args.serial_port,
                "message": f"keyboard {key_char}",
            })
            return

        if key == 32 and current_result is not None:  # SPACE
            self.send_prediction(current_result, "keyboard_space_send_current_prediction")

    def run(self) -> None:
        self.open_camera()
        cv2 = self.cv2

        print("\n[READY] 字符协议版上位机已启动")
        print("[READY] 自动模式：识别稳定后发送 R/K/H/O")
        print("[READY] 手动模式：按 r/k/h/o 或空格发送，q 退出\n")

        try:
            while self.running:
                ok, frame = self.cap.read()
                if not ok or frame is None:
                    print("[WARN] 摄像头读取失败")
                    time.sleep(0.1)
                    continue

                self.frame_id += 1
                result = self.classifier.predict(frame)
                state = self.stable_predictor.update(result)

                # 控制终端输出频率，避免刷屏
                if self.frame_id % max(1, self.args.print_every) == 0:
                    print(
                        f"[PRED] frame={self.frame_id} "
                        f"class={result['display_class']} conf={result['confidence']:.3f} "
                        f"stable={state.stable_count}/{self.args.stable_frames} "
                        f"tx={result['tx_char']}"
                    )

                if self.args.mode == "auto" and state.is_stable:
                    ch = result["tx_char"]
                    if self.can_send_now(ch):
                        sent = self.send_prediction(result, "auto_stable_prediction")
                        if sent and self.args.save_on_send:
                            self.save_snapshot(frame, result, "sent")

                show = self.draw_overlay(frame, result, state)

                if not self.args.no_window:
                    cv2.imshow(WINDOW_NAME, show)
                    key = cv2.waitKey(1) & 0xFF
                    self.handle_key(key, result, show)
                else:
                    time.sleep(self.args.frame_interval)

                time.sleep(self.args.frame_interval)

        except KeyboardInterrupt:
            print("\n[EXIT] 用户 Ctrl+C 退出")
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
    if ch not in VALID_TEST_CHARS:
        raise ValueError("--test-char 只能是 R / K / H / O")

    ser = ServoCharSerial(args.serial_port, args.baudrate, args.serial_timeout)
    try:
        ser.open()
        time.sleep(0.2)
        ser.send_char(ch)
        print("[DONE] 单字符串口测试完成。")
    finally:
        ser.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="树莓派上位机：专门配合不修改版《舵机控制.c》，只发送 R/K/H/O 单字符。"
    )

    parser.add_argument("--mode", choices=["auto", "manual"], default="auto",
                        help="auto=稳定识别后自动发送；manual=只按键发送")
    parser.add_argument("--test-char", choices=["R", "K", "H", "O", "r", "k", "h", "o"], default=None,
                        help="只测试串口发送一个字符，不打开摄像头、不加载模型")

    parser.add_argument("--serial-port", default=DEFAULT_SERIAL_PORT,
                        help="串口设备，例如 /dev/ttyAMA0 或 /dev/ttyUSB0")
    parser.add_argument("--baudrate", type=int, default=DEFAULT_BAUDRATE,
                        help="波特率，必须和舵机控制.c 一致，默认 9600")
    parser.add_argument("--serial-timeout", type=float, default=DEFAULT_SERIAL_TIMEOUT)

    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH),
                        help="TFLite 模型路径")
    parser.add_argument("--class-mapping", default=str(DEFAULT_CLASS_MAPPING_PATH),
                        help="class_mapping.json 路径")
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR))
    parser.add_argument("--capture-dir", default=str(DEFAULT_CAPTURE_DIR))

    parser.add_argument("--camera-index", type=int, default=DEFAULT_CAMERA_INDEX)
    parser.add_argument("--width", type=int, default=DEFAULT_CAPTURE_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_CAPTURE_HEIGHT)
    parser.add_argument("--no-window", action="store_true",
                        help="SSH 无桌面时使用，不显示 OpenCV 窗口")

    parser.add_argument("--conf-threshold", type=float, default=0.80,
                        help="置信度阈值")
    parser.add_argument("--stable-frames", type=int, default=3,
                        help="连续多少帧同类别且高置信才认为稳定")
    parser.add_argument("--send-cooldown", type=float, default=4.0,
                        help="自动模式下两次发送之间的最小间隔，防止舵机被连续刷指令")
    parser.add_argument("--no-repeat-same", action="store_true",
                        help="自动模式下同一类别只发一次，直到识别类别变化或重启脚本")
    parser.add_argument("--save-on-send", action="store_true",
                        help="每次自动发送时保存截图")
    parser.add_argument("--frame-interval", type=float, default=0.03)
    parser.add_argument("--print-every", type=int, default=10,
                        help="每多少帧打印一次识别结果")

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print("========== RPI SERVO CHAR CONTROL ONLY_CHAR_V2 ==========")
    print("协议：只发送单字符 R / K / H / O")
    print("不使用：AA xx 55、0xA1、0xCC、0xDD")
    print(f"串口：{args.serial_port}, baud={args.baudrate}")

    try:
        if args.test_char is not None:
            run_test_char(args)
            return 0

        app = ServoCharApp(args)
        app.run()
        return 0

    except Exception as exc:
        print("\n[ERROR] 程序异常：", exc)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
