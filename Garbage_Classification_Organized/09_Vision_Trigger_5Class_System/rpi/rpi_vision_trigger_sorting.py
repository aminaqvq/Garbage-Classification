"""★ 最终树莓派上位机：五分类视觉触发 + 串口 RKHO + 满载保护 + MCU 反馈\n\n可直接运行版：默认使用 float32 TFLite，默认连接单片机，自动查找串口

状态机：
  BOOT → IDLE_WAIT_VISUAL → CANDIDATE_DETECTED → SEND_SORT_COMMAND
  → WAIT_MCU_DONE → WAIT_RETURN_TO_PENDING → IDLE_WAIT_VISUAL
  FULL_PAUSED（任意状态收到 F） → IDLE_WAIT_VISUAL（收到 N）
  ERROR_RECOVERY（收到 E 或 D 超时）

协议: RPi→MCU: R/K/H/O, MCU→RPi: D/F/N/E
不发送「待分拣」，不等待 T 触发，不使用超声波。

部署目录：09_Vision_Trigger_5Class_System/rpi/
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse
import csv
import json
import logging
import math
import sys
import time
import traceback
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# =========================================================
# 默认配置 — 指向 09_Vision_Trigger_5Class_System
# =========================================================

def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent

PROJECT_ROOT_DEFAULT = _project_root()
CONFIG_DIR = PROJECT_ROOT_DEFAULT / "09_Vision_Trigger_5Class_System" / "config"
MODELS_DIR = PROJECT_ROOT_DEFAULT / "models" / "vision_trigger_5class_tflite"
DEFAULT_CLASS_CONFIG = CONFIG_DIR / "class_mapping_5class.json"
DEFAULT_RUNTIME_CONFIG = CONFIG_DIR / "runtime_config.example.json"


def _default_model_path(project_root: Optional[Path] = None) -> Path:
    """
    树莓派 + tflite_runtime 稳定版：
    优先加载 latest_tflite_float32.tflite（最常见的导出命名），
    回退到 model_float32_simplified_float32.tflite。
    禁止加载 float16 模型，避免 CONV_2D allocate_tensors 阶段报 input_type 错误。
    """
    models_dir = (project_root or PROJECT_ROOT_DEFAULT) / "models" / "vision_trigger_5class_tflite"
    candidates = [
        models_dir / "latest_tflite_float32.tflite",
        models_dir / "model_float32_simplified_float32.tflite",
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(
        "未找到 float32 TFLite 模型文件。请确认存在：\n"
        f"  {models_dir / 'latest_tflite_float32.tflite'}\n"
        f"  {models_dir / 'model_float32_simplified_float32.tflite'}"
    )

# GPIO14/15 UART0 on this Raspberry Pi 5.
SERIAL_PORT_DEFAULT = "/dev/ttyAMA0"
BAUDRATE_DEFAULT = 9600
SERIAL_TIMEOUT = 0.05
CAMERA_INDEX_DEFAULT = 0
CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 480
WINDOW_NAME = "Garbage Sorting — Vision Trigger 5Class"

RESIZE_SIZE = 256
CROP_SIZE = 224
RGB_INPUT = True
NORMALIZE = True
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
ROI_W_RATIO = 0.56
ROI_H_RATIO = 0.70

CONF_THRESHOLD_DEFAULT = 0.82
STABLE_FRAMES_DEFAULT = 5
RETURN_TO_PENDING_FRAMES_DEFAULT = 8
# MCU firmware action: 0.5s servo + 1.0s open + 1.0s stop + 0.5s servo
# + 0.2s settle + 1.0s close; allow serial/scheduler margin.
DONE_TIMEOUT_SEC_DEFAULT = 15.0
FRAME_INTERVAL_SEC = 0.03

CLASS_TO_SERVO_CHAR = {"可回收": "R", "厨余": "K", "有害": "H", "其他": "O"}
VALID_TX_CHARS = {"R", "H", "K", "O"}
VALID_RX_CHARS = {"D", "F", "N", "E"}
CLASS_DISPLAY_NAME = {
    "可回收": "可回收垃圾", "厨余": "厨余垃圾",
    "有害": "有害垃圾", "其他": "其他垃圾", "待分拣": "待分拣（视觉等待）",
}
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
]
FONT_SIZE_MAIN = 24
FONT_SIZE_SMALL = 19

STATE_BOOT = "BOOT"
STATE_IDLE_WAIT_VISUAL = "IDLE_WAIT_VISUAL"
STATE_CANDIDATE_DETECTED = "CANDIDATE_DETECTED"
STATE_SEND_SORT_COMMAND = "SEND_SORT_COMMAND"
STATE_WAIT_MCU_DONE = "WAIT_MCU_DONE"
STATE_WAIT_RETURN_TO_PENDING = "WAIT_RETURN_TO_PENDING"
STATE_FULL_PAUSED = "FULL_PAUSED"
STATE_ERROR_RECOVERY = "ERROR_RECOVERY"


# =========================================================
# 工具函数
# =========================================================

def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def timestamp_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)

def hex_str(data: bytes) -> str:
    return " ".join(f"{b:02X}" for b in data)


def auto_detect_serial_port(preferred: Optional[str] = None) -> str:
    """
    自动查找常见单片机串口：
      USB 转串口: /dev/ttyUSB*
      Arduino/CDC: /dev/ttyACM*
      树莓派 UART: /dev/ttyAMA0, /dev/serial0
      稳定路径: /dev/serial/by-id/*
    如果 preferred 存在，则优先使用 preferred。
    """
    candidates: List[Path] = []

    if preferred is not None:
        p = Path(preferred)
        if not p.is_char_device():
            raise FileNotFoundError(f"指定串口不存在，不会自动切换到其他设备: {p}")
        return str(p)

    by_id_dir = Path("/dev/serial/by-id")
    if by_id_dir.exists():
        candidates.extend(sorted(by_id_dir.glob("*")))

    for pattern in (
        "/dev/ttyUSB*",
        "/dev/ttyACM*",
        "/dev/serial0",
        "/dev/ttyAMA0",
        "/dev/ttyS0",
    ):
        candidates.extend(sorted(Path("/").glob(pattern.lstrip("/"))))

    seen = set()
    unique = []
    for p in candidates:
        sp = str(p)
        if sp not in seen:
            unique.append(p)
            seen.add(sp)

    existing = []
    for p in unique:
        if p.is_char_device():
            resolved = p.resolve()
            if all(resolved != other.resolve() for other in existing):
                existing.append(p)
    if len(existing) == 1:
        return str(existing[0])
    if len(existing) > 1:
        raise RuntimeError(
            "检测到多个串口设备，拒绝猜测 MCU："
            + ", ".join(str(p) for p in existing)
            + "；请使用 --serial-port 或 serial.port 明确指定"
        )

    checked = "\n  ".join(str(p) for p in unique) if unique else "无候选串口"
    raise FileNotFoundError(
        "没有找到可用串口。请先连接单片机，然后执行：\n"
        "  ls -l /dev/ttyUSB* /dev/ttyACM* /dev/ttyAMA* /dev/serial/by-id/* 2>/dev/null\n"
        "也可以手动指定，例如：\n"
        "  --serial-port /dev/ttyUSB0\n"
        f"已检查：\n  {checked}"
    )


# =========================================================
# 日志
# =========================================================

def setup_logger(log_dir: Path) -> logging.Logger:
    ensure_dirs(log_dir)
    logger = logging.getLogger("VisionTriggerSorting")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    console = logging.StreamHandler(sys.stdout); console.setFormatter(fmt)
    fh = RotatingFileHandler(log_dir / "vision_trigger_runtime.log", maxBytes=2*1024*1024, backupCount=5, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(console); logger.addHandler(fh)
    return logger


def append_csv(csv_path: Path, row: Dict) -> None:
    ensure_dirs(csv_path.parent)
    exists = csv_path.exists()
    FIELDS = ["time","round_id","event","pred_id","raw_class","display_class",
              "confidence","stable_count","tx_char","tx_hex","rx_hex","mcu_event",
              "system_state","message"]
    with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists: w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})


# =========================================================
# 五分类映射加载
# =========================================================

def load_class_mapping_5class(
    config_path: Path, logger: logging.Logger
) -> Tuple[Dict[int, str], Dict[str, str], Set[str]]:
    if not config_path.exists():
        raise FileNotFoundError(f"五分类配置文件不存在：{config_path}")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"读取五分类配置失败：{config_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"五分类配置必须是 JSON 对象：{config_path}")
    c2i = data.get("class_to_idx")
    if not isinstance(c2i, dict) or len(c2i) != 5:
        raise ValueError(f"缺少 class_to_idx：{config_path}")
    expected = {"待分拣", "其他", "厨余", "可回收", "有害"}
    if set(c2i) != expected:
        raise ValueError(f"类别不匹配：需要 {expected}，实际 {set(c2i)}")
    if any(isinstance(v, bool) or not isinstance(v, int) for v in c2i.values()):
        raise ValueError(f"class_to_idx 索引必须是整数：{config_path}")
    if set(c2i.values()) != set(range(5)):
        raise ValueError(f"class_to_idx 索引必须严格为 0~4 且唯一：{c2i}")
    idx_to_class = {v: k for k, v in c2i.items()}
    if set(idx_to_class.values()) != expected:
        raise ValueError(f"类别不匹配：需要 {expected}，实际 {set(idx_to_class.values())}")
    action_mapping = data.get("action_mapping", CLASS_TO_SERVO_CHAR)
    if not isinstance(action_mapping, dict):
        raise ValueError(f"action_mapping 必须是对象：{config_path}")
    no_action = data.get("no_action_classes", ["待分拣"])
    if not isinstance(no_action, list) or not all(isinstance(v, str) for v in no_action):
        raise ValueError(f"no_action_classes 必须是字符串数组：{config_path}")
    no_action_classes = set(no_action)
    if not no_action_classes or not no_action_classes.issubset(expected):
        raise ValueError(f"no_action_classes 包含未知类别或为空：{no_action}")
    if no_action_classes != {"待分拣"}:
        raise ValueError(f"固定五分类协议要求 no_action_classes 只能是 ['待分拣']：{no_action}")
    if set(action_mapping) & no_action_classes:
        raise ValueError(f"no_action_classes 不得出现在 action_mapping：{action_mapping}")
    if set(action_mapping) != expected - no_action_classes:
        raise ValueError(
            f"action_mapping 必须覆盖所有可动作类别：需要 {expected - no_action_classes}，实际 {set(action_mapping)}"
        )
    if any(not isinstance(v, str) or v not in VALID_TX_CHARS for v in action_mapping.values()):
        raise ValueError(f"action_mapping 包含非法串口字符：{action_mapping}")
    if len(set(action_mapping.values())) != len(action_mapping):
        raise ValueError(f"action_mapping 不允许多个类别共用同一串口字符：{action_mapping}")
    if action_mapping != CLASS_TO_SERVO_CHAR:
        raise ValueError(
            f"固定 MCU 协议要求 action_mapping={CLASS_TO_SERVO_CHAR}，实际：{action_mapping}"
        )
    logger.info("五分类映射: %s", idx_to_class)
    logger.info("分拣动作: %s  | 不发送: %s", action_mapping, no_action_classes)
    return idx_to_class, action_mapping, no_action_classes


def load_runtime_config(
    config_path: Optional[Path], logger: logging.Logger, project_root: Path = PROJECT_ROOT_DEFAULT,
    model_path_override: Optional[Path] = None, require_model: bool = True,
) -> Dict:
    defaults = {
        "model_path": None,
        "camera_device_index": CAMERA_INDEX_DEFAULT,
        "capture_width": CAPTURE_WIDTH, "capture_height": CAPTURE_HEIGHT,
        "confidence_threshold": CONF_THRESHOLD_DEFAULT,
        "stable_frames_required": STABLE_FRAMES_DEFAULT,
        "return_to_pending_frames_required": RETURN_TO_PENDING_FRAMES_DEFAULT,
        "done_timeout_seconds": DONE_TIMEOUT_SEC_DEFAULT,
        "send_cooldown_seconds": 0.0,
        "serial_port": SERIAL_PORT_DEFAULT, "serial_baudrate": BAUDRATE_DEFAULT,
        "serial_timeout_seconds": SERIAL_TIMEOUT,
        "roi_enabled": False, "roi": {"x": 0, "y": 0, "w": 640, "h": 480},
    }
    if config_path is None:
        if not require_model:
            return defaults
        model_path = model_path_override or _default_model_path(project_root)
        if not model_path.is_file():
            raise FileNotFoundError(f"TFLite 模型不存在：{model_path}")
        defaults["model_path"] = str(model_path.resolve())
        return defaults
    if not config_path.exists():
        raise FileNotFoundError(f"运行时配置文件不存在：{config_path}")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"读取运行时配置失败：{config_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"运行时配置必须是 JSON 对象：{config_path}")

    m = data.get("model", {})
    c = data.get("camera", {})
    d = data.get("decision", {})
    s = data.get("serial", {})
    for name, section in (("model", m), ("camera", c), ("decision", d), ("serial", s)):
        if not isinstance(section, dict):
            raise ValueError(f"运行时配置 {name} 必须是对象：{config_path}")

    if model_path_override is not None:
        if not model_path_override.is_file():
            raise FileNotFoundError(f"命令行指定模型不存在：{model_path_override}")
        defaults["model_path"] = str(model_path_override.resolve())
    elif require_model:
        if "path" not in m:
            raise ValueError(f"运行时配置缺少 model.path：{config_path}")
        if not isinstance(m["path"], str) or not m["path"].strip():
            raise ValueError(f"model.path 必须是非空字符串：{config_path}")
        model_path = Path(m["path"]).expanduser()
        if not model_path.is_absolute():
            model_path = project_root / model_path
        if not model_path.is_file():
            raise FileNotFoundError(f"配置指定模型不存在：{model_path}")
        defaults["model_path"] = str(model_path.resolve())

    if "device_index" in c:
        if isinstance(c["device_index"], bool) or not isinstance(c["device_index"], int) or c["device_index"] < 0:
            raise ValueError(f"camera.device_index 无效：{c['device_index']}")
        defaults["camera_device_index"] = c["device_index"]
    for key, default_key in (("width", "capture_width"), ("height", "capture_height")):
        if key in c:
            if isinstance(c[key], bool) or not isinstance(c[key], int) or c[key] <= 0:
                raise ValueError(f"camera.{key} 无效：{c[key]}")
            defaults[default_key] = c[key]
    if "roi_enabled" in c:
        if not isinstance(c["roi_enabled"], bool):
            raise ValueError(f"camera.roi_enabled 必须是布尔值：{c['roi_enabled']}")
        defaults["roi_enabled"] = c["roi_enabled"]
    if "roi" in c:
        roi = c["roi"]
        if not isinstance(roi, dict) or any(key not in roi for key in ("x", "y", "w", "h")):
            raise ValueError(f"camera.roi 必须包含 x/y/w/h：{roi}")
        if any(isinstance(roi[key], bool) or not isinstance(roi[key], int) for key in ("x", "y", "w", "h")):
            raise ValueError(f"camera.roi 坐标必须是整数：{roi}")
        if roi["x"] < 0 or roi["y"] < 0 or roi["w"] <= 0 or roi["h"] <= 0:
            raise ValueError(f"camera.roi 范围无效：{roi}")
        if defaults["roi_enabled"] and (
            roi["x"] + roi["w"] > defaults["capture_width"]
            or roi["y"] + roi["h"] > defaults["capture_height"]
        ):
            raise ValueError(f"camera.roi 超出配置画面：{roi} / {defaults['capture_width']}x{defaults['capture_height']}")
        defaults["roi"] = dict(roi)

    decision_types = {
        "confidence_threshold": (float, 0.0, 1.0, False),
        "stable_frames_required": (int, 1, None, True),
        "return_to_pending_frames_required": (int, 1, None, True),
        "done_timeout_seconds": (float, 0.0, None, False),
        "send_cooldown_seconds": (float, 0.0, None, True),
    }
    for key, (value_type, minimum, maximum, inclusive_minimum) in decision_types.items():
        if key not in d:
            continue
        value = d[key]
        valid_type = isinstance(value, value_type)
        if value_type is float:
            valid_type = isinstance(value, (int, float)) and not isinstance(value, bool)
        elif value_type is int:
            valid_type = isinstance(value, int) and not isinstance(value, bool)
        if not valid_type or (isinstance(value, float) and not math.isfinite(value)):
            raise ValueError(f"decision.{key} 类型无效：{value}")
        lower_invalid = value < minimum or (not inclusive_minimum and value == minimum)
        if lower_invalid or (maximum is not None and value > maximum):
            raise ValueError(f"decision.{key} 范围无效：{value}")
        defaults[key] = float(value) if value_type is float else value

    if "port" in s:
        if not isinstance(s["port"], str) or not s["port"].strip():
            raise ValueError(f"serial.port 必须是非空字符串：{s['port']}")
        defaults["serial_port"] = s["port"]
    if "baudrate" in s:
        if isinstance(s["baudrate"], bool) or not isinstance(s["baudrate"], int) or s["baudrate"] <= 0:
            raise ValueError(f"serial.baudrate 无效：{s['baudrate']}")
        if s["baudrate"] != BAUDRATE_DEFAULT:
            raise ValueError(f"固定 MCU 协议要求 baudrate={BAUDRATE_DEFAULT}：{s['baudrate']}")
        defaults["serial_baudrate"] = s["baudrate"]
    if "timeout_seconds" in s:
        if isinstance(s["timeout_seconds"], bool) or not isinstance(s["timeout_seconds"], (int, float)) or s["timeout_seconds"] <= 0:
            raise ValueError(f"serial.timeout_seconds 无效：{s['timeout_seconds']}")
        if not math.isfinite(float(s["timeout_seconds"])):
            raise ValueError(f"serial.timeout_seconds 无效：{s['timeout_seconds']}")
        defaults["serial_timeout_seconds"] = float(s["timeout_seconds"])
    logger.info("运行时配置已从 %s 加载", config_path)
    return defaults


# =========================================================
# TFLite 分类器（使用延迟导入的模块引用）
# =========================================================

class GarbageClassifierTFLite:
    def __init__(self, model_path: Path, idx_to_class: Dict[int, str], logger: logging.Logger,
                 cv2_mod, np_mod, Interpreter_cls):
        self.cv2 = cv2_mod; self.np = np_mod
        if not model_path.is_file():
            raise FileNotFoundError(f"TFLite 模型不存在: {model_path}")
        self.model_path = model_path; self.idx_to_class = idx_to_class; self.logger = logger
        self.logger.info("加载模型: %s", model_path)
        self.interpreter = Interpreter_cls(model_path=str(model_path))
        self.interpreter.allocate_tensors()
        self.input_detail = self.interpreter.get_input_details()[0]
        self.output_detail = self.interpreter.get_output_details()[0]
        self.input_shape = list(self.input_detail["shape"])
        self.input_layout = self._infer_layout(self.input_shape)
        if len(self.input_shape) != 4:
            raise ValueError(f"模型输入必须是四维张量：{self.input_shape}")
        if self.input_layout == "NCHW":
            self.input_channels = int(self.input_shape[1])
            self.input_height = int(self.input_shape[2])
            self.input_width = int(self.input_shape[3])
        else:
            self.input_height = int(self.input_shape[1])
            self.input_width = int(self.input_shape[2])
            self.input_channels = int(self.input_shape[3])
        if self.input_channels != 3 or self.input_height <= 0 or self.input_width <= 0:
            raise ValueError(f"模型输入必须是固定 RGB 尺寸：{self.input_shape}")
        self.logger.info("输入 shape: %s 布局: %s", self.input_shape, self.input_layout)
        self._warmup()

    @staticmethod
    def _infer_layout(shape):
        if len(shape) != 4: return "NHWC"
        if shape[1] in (1, 3) and shape[-1] not in (1, 3): return "NCHW"
        return "NHWC"

    def _warmup(self):
        self.logger.info("模型预热...")
        np = self.np
        if self.input_layout == "NCHW":
            dummy = np.zeros((1, self.input_channels, self.input_height, self.input_width), dtype=np.float32)
        else:
            dummy = np.zeros((1, self.input_height, self.input_width, self.input_channels), dtype=np.float32)
        dummy = self._quantize(dummy, self.input_detail)
        self.interpreter.set_tensor(self.input_detail["index"], dummy)
        self.interpreter.invoke()
        _ = self.interpreter.get_tensor(self.output_detail["index"])
        self.logger.info("预热完成")

    def _quantize(self, x, detail):
        np = self.np
        dtype = detail["dtype"]; scale, zp = detail.get("quantization", (0.0, 0))
        if dtype in (np.float16, np.float32): return x.astype(dtype)
        if scale in (0.0, None):
            raise ValueError(f"量化模型缺少有效 scale：dtype={dtype} quantization={detail.get('quantization')}")
        q = np.round(x / float(scale) + int(zp))
        if dtype == np.int8: return np.clip(q, -128, 127).astype(np.int8)
        if dtype == np.uint8: return np.clip(q, 0, 255).astype(np.uint8)
        return q.astype(dtype)

    @staticmethod
    def _dequant(y, detail):
        import numpy as np
        dtype = detail["dtype"]; scale, zp = detail.get("quantization", (0.0, 0))
        if dtype in (np.float16, np.float32): return y.astype(np.float32)
        if scale in (0.0, None):
            raise ValueError(f"量化输出缺少有效 scale：dtype={dtype} quantization={detail.get('quantization')}")
        return (y.astype(np.float32) - int(zp)) * float(scale)

    def preprocess(self, roi_bgr):
        cv2, np = self.cv2, self.np
        if RGB_INPUT:
            img = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        else:
            img = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        resize_size = max(RESIZE_SIZE, self.input_height, self.input_width)
        img = _resize_keep_ratio(img, resize_size)
        img = _center_crop(img, (self.input_height, self.input_width))
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
        return self._quantize(x.astype(np.float32), self.input_detail)

    def predict(self, roi_bgr) -> Dict:
        np = self.np
        x = self.preprocess(roi_bgr)
        self.interpreter.set_tensor(self.input_detail["index"], x)
        self.interpreter.invoke()
        y = self.interpreter.get_tensor(self.output_detail["index"])
        y = self._dequant(y, self.output_detail)
        logits = y.reshape(-1).astype(np.float32)
        if logits.size != len(self.idx_to_class):
            raise ValueError(f"模型输出类别数不匹配：输出 {logits.size}，配置 {len(self.idx_to_class)}")
        probs = self._softmax(logits)
        pred_id = int(np.argmax(probs))
        confidence = float(probs[pred_id])
        raw_class = self.idx_to_class.get(pred_id, str(pred_id))
        display = CLASS_DISPLAY_NAME.get(raw_class, raw_class)
        return {"pred_id": pred_id, "raw_class": raw_class, "display_class": display,
                "confidence": confidence, "probs": probs.tolist()}

    @staticmethod
    def _softmax(x):
        import numpy as np
        x = x - np.max(x); e = np.exp(x); return e / np.sum(e)


# 独立图像工具（不依赖类）
def _resize_keep_ratio(img, shorter_side):
    import cv2
    h, w = img.shape[:2]
    if h <= 0 or w <= 0: raise ValueError("无效尺寸")
    if w < h: nw, nh = shorter_side, int(round(h * shorter_side / w))
    else: nh, nw = shorter_side, int(round(w * shorter_side / h))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

def _center_crop(img, crop_size):
    import cv2
    h, w = img.shape[:2]
    if isinstance(crop_size, int):
        crop_h = crop_w = crop_size
    else:
        crop_h, crop_w = crop_size
    left = max(0, int(round((w - crop_w) / 2)))
    top = max(0, int(round((h - crop_h) / 2)))
    crop = img[top:top + crop_h, left:left + crop_w]
    if crop.shape[0] != crop_h or crop.shape[1] != crop_w:
        crop = cv2.resize(crop, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
    return crop


# =========================================================
# 稳定识别器
# =========================================================

class StablePredictor:
    def __init__(self, conf_threshold: float, stable_frames: int):
        self.conf_threshold = conf_threshold; self.stable_frames = stable_frames; self.reset()

    def reset(self):
        self.last_pred_id = None; self.stable_count = 0; self.best_result = None; self.best_conf = -1.0

    def update(self, result: Dict) -> Dict:
        pred_id = result["pred_id"]; conf = result["confidence"]
        if conf > self.best_conf: self.best_conf = conf; self.best_result = result
        if conf < self.conf_threshold:
            self.stable_count = 0; self.last_pred_id = None
            return {"is_stable": False, "stable_count": 0, "status": "低置信度"}
        if pred_id == self.last_pred_id: self.stable_count += 1
        else: self.last_pred_id = pred_id; self.stable_count = 1
        is_stable = self.stable_count >= self.stable_frames
        return {"is_stable": is_stable, "stable_count": self.stable_count,
                "status": "识别稳定" if is_stable else "识别中"}


# =========================================================
# 串口管理
# =========================================================

class ServoCharSerial:
    def __init__(self, port: str, baudrate: int, timeout: float, logger: logging.Logger, serial_mod):
        self.port = port; self.baudrate = baudrate; self.logger = logger
        self.timeout = timeout; self.serial_mod = serial_mod; self.ser = None; self.faulted = False

    def open(self):
        s = self.serial_mod
        self.logger.info("打开串口: %s @ %d", self.port, self.baudrate)
        self.ser = s.Serial(port=self.port, baudrate=self.baudrate, bytesize=s.EIGHTBITS,
                            parity=s.PARITY_NONE, stopbits=s.STOPBITS_ONE,
                            timeout=self.timeout, xonxoff=False, rtscts=False, dsrdtr=False)
        # Drop stale bytes before waiting for fresh MCU status.
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()
        time.sleep(1.0)
        self.logger.info("串口就绪")

    def send_char(self, ch: str) -> bytes:
        if ch not in VALID_TX_CHARS: raise ValueError(f"非法控制字符: {ch!r}")
        if self.faulted: raise RuntimeError("串口已故障，禁止继续发送")
        if self.ser is None or not self.ser.is_open: raise RuntimeError("串口未打开")
        data = ch.encode("ascii")
        try:
            written = self.ser.write(data)
            if written != len(data):
                raise IOError(f"串口只写入 {written}/{len(data)} 字节")
            self.ser.flush()
        except Exception as exc:
            self.faulted = True
            self.logger.exception("串口写异常，已锁定发送")
            raise RuntimeError(f"串口写失败: {exc}") from exc
        self.logger.info("TX: %s -> %s", ch, hex_str(data))
        return data

    def read_available(self) -> bytes:
        if self.ser is None or not self.ser.is_open:
            raise RuntimeError("串口未打开")
        if self.faulted: raise RuntimeError("串口已故障，禁止继续读取")
        try:
            n = self.ser.in_waiting
            if n > 0: return self.ser.read(n)
            return b""
        except Exception as exc:
            self.faulted = True
            self.logger.exception("串口读异常，已锁定发送")
            raise RuntimeError(f"串口读失败: {exc}") from exc

    def close(self):
        if self.ser and self.ser.is_open: self.ser.close(); self.logger.info("串口关闭")


# =========================================================
# 摄像头管理
# =========================================================

class CameraManager:
    def __init__(self, camera_index: int, width: int, height: int, logger: logging.Logger, cv2_mod):
        self.idx = camera_index; self.width = width; self.height = height
        self.logger = logger; self.cv2 = cv2_mod; self.cap = None

    def open(self):
        cv2 = self.cv2
        self.logger.info("打开摄像头 %d", self.idx)
        self.cap = cv2.VideoCapture(self.idx)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if not self.cap.isOpened():
            self.cap.release()
            self.cap = None
            raise RuntimeError(f"无法打开摄像头: {self.idx}")
        for _ in range(10):
            self.cap.read()
            time.sleep(0.03)
        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.close()
            raise RuntimeError(f"摄像头无有效画面: {self.idx}")
        actual_height, actual_width = frame.shape[:2]
        if (actual_width, actual_height) != (self.width, self.height):
            self.close()
            raise RuntimeError(
                f"摄像头实际分辨率为 {actual_width}x{actual_height}，"
                f"配置请求 {self.width}x{self.height}"
            )
        self.logger.info("摄像头就绪")

    def read(self):
        if self.cap is None: raise RuntimeError("摄像头未打开")
        ret, frame = self.cap.read()
        if not ret or frame is None: raise RuntimeError("读取失败")
        height, width = frame.shape[:2]
        if (width, height) != (self.width, self.height):
            raise RuntimeError(f"摄像头分辨率变化为 {width}x{height}，期望 {self.width}x{self.height}")
        return frame

    def close(self):
        if self.cap: self.cap.release(); self.logger.info("摄像头关闭")


# =========================================================
# 画面叠加
# =========================================================

def _get_center_roi(frame):
    h, w = frame.shape[:2]
    rw = int(w * ROI_W_RATIO); rh = int(h * ROI_H_RATIO)
    x1 = max((w - rw)//2, 0); y1 = max((h - rh)//2, 0)
    return x1, y1, min(x1+rw, w), min(y1+rh, h)


def _get_configured_roi(frame, enabled: bool, roi: Dict) -> Tuple[int, int, int, int]:
    if not enabled:
        return _get_center_roi(frame)
    h, w = frame.shape[:2]
    x1 = int(roi["x"])
    y1 = int(roi["y"])
    if x1 < 0 or y1 < 0 or x1 + int(roi["w"]) > w or y1 + int(roi["h"]) > h:
        raise ValueError(f"ROI 起点超出当前画面：{roi}，画面尺寸：{w}x{h}")
    x2 = x1 + int(roi["w"])
    y2 = y1 + int(roi["h"])
    return x1, y1, x2, y2


def _get_font(cv2_mod, size):
    from PIL import ImageFont
    for p in FONT_CANDIDATES:
        if os.path.exists(p):
            try: return ImageFont.truetype(p, size)
            except: continue
    return None


def _draw_overlay(cv2_mod, frame, roi, status, cls, conf, stable, ser, state, rnd, fps, color=(0,255,0)):
    cv2 = cv2_mod; x1, y1, x2, y2 = roi
    show = frame.copy(); cv2.rectangle(show, (x1,y1), (x2,y2), color, 2)
    over = show.copy(); cv2.rectangle(over, (10,10), (630,320), (0,0,0), -1)
    show = cv2.addWeighted(over, 0.45, show, 0.55, 0)
    lines = [f"状态：{status}", f"类别：{cls}", f"置信度：{conf}", f"稳定帧：{stable}",
             f"串口：   {ser}", f"系统状态：{state}", f"轮次：{rnd}    FPS：{fps}"]
    font = _get_font(cv2, FONT_SIZE_SMALL)
    if font is None:
        y = 34
        for l in lines:
            sl = l.encode("ascii", errors="replace").decode("ascii")
            cv2.putText(show, sl, (22,y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2); y += 30
        cv2.putText(show, "ROI", (x1+8, y1-8 if y1>25 else y1+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return show
    try:
        from PIL import Image, ImageDraw
        import numpy as np
        rgb = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        pi = Image.fromarray(rgb); dr = ImageDraw.Draw(pi)
        py = 15
        for l in lines: dr.text((22,py), l, font=font, fill=(255,255,255)); py += 30
        dr.text((x1+8, y1-30 if y1>35 else y1+8), "ROI", font=font, fill=(color[2],color[1],color[0]))
        return cv2.cvtColor(np.array(pi), cv2.COLOR_RGB2BGR)
    except: return show


# =========================================================
# 主系统
# =========================================================

class VisionTriggerSortingApp:
    def __init__(self, args, cv2_mod, np_mod, serial_mod, Interpreter_cls, tflite_backend):
        self.args = args
        self.cv2 = cv2_mod; self.np = np_mod
        self.project_root = Path(args.project_root).expanduser().resolve()
        rc_path = self._resolve_path(args.runtime_config) if args.runtime_config else None
        model_override = self._resolve_path(args.model_path) if args.model_path else None
        self.runtime_cfg = load_runtime_config(
            rc_path, _dummy_logger(), self.project_root, model_override
        )
        self.model_path = Path(self.runtime_cfg["model_path"])
        self.class_config_path = self._resolve_path(args.class_config) if args.class_config else (
            self.project_root / "09_Vision_Trigger_5Class_System" / "config" / "class_mapping_5class.json"
        )
        self.log_dir = self.project_root / "Logs"
        self.cap_dir = self.project_root / "Captures_Vision_Trigger"
        self.csv_path = self.log_dir / "vision_trigger_rounds.csv"
        ensure_dirs(self.log_dir, self.cap_dir)
        self.logger = setup_logger(self.log_dir)
        self.logger.info("PROJECT_ROOT: %s", self.project_root)
        self.logger.info("MODEL: %s | CLASS_CONFIG: %s", self.model_path, self.class_config_path)
        self.logger.info("TFLite 后端: %s", tflite_backend)
        self.idx_to_class, self.action_mapping, self.no_action_classes = load_class_mapping_5class(
            self.class_config_path, self.logger
        )

        self.conf_threshold = float(
            args.conf_threshold if args.conf_threshold is not None else self.runtime_cfg["confidence_threshold"]
        )
        self.stable_frames = int(
            args.stable_frames if args.stable_frames is not None else self.runtime_cfg["stable_frames_required"]
        )
        self.return_pending_frames = int(
            args.return_pending_frames
            if args.return_pending_frames is not None
            else self.runtime_cfg["return_to_pending_frames_required"]
        )
        self.done_timeout = float(
            args.done_timeout if args.done_timeout is not None else self.runtime_cfg["done_timeout_seconds"]
        )
        self.send_cooldown_seconds = float(self.runtime_cfg["send_cooldown_seconds"])
        if not 0.0 < self.conf_threshold <= 1.0:
            raise ValueError(f"confidence threshold 必须在 (0, 1]：{self.conf_threshold}")
        if not math.isfinite(self.conf_threshold) or not math.isfinite(self.done_timeout):
            raise ValueError("confidence threshold 和 D 超时必须是有限数值")
        if self.stable_frames < 1 or self.return_pending_frames < 1:
            raise ValueError("稳定帧数必须 >= 1")
        if self.done_timeout <= 0.0:
            raise ValueError("D 超时必须 > 0")
        self.logger.info("决策: conf=%.2f stable=%d ret_pending=%d timeout=%.1f",
                         self.conf_threshold, self.stable_frames, self.return_pending_frames, self.done_timeout)

        self.classifier = None; self.serial_mgr = None; self.camera_mgr = None
        if not args.dry_run:
            self.classifier = GarbageClassifierTFLite(self.model_path, self.idx_to_class, self.logger, cv2_mod, np_mod, Interpreter_cls)
            if not args.no_serial:
                if args.serial_port is not None and not args.serial_port.strip():
                    raise ValueError("--serial-port 不能为空")
                preferred_sp = args.serial_port if args.serial_port is not None else self.runtime_cfg.get("serial_port")
                sp = auto_detect_serial_port(preferred_sp)
                br = args.baudrate if args.baudrate is not None else int(self.runtime_cfg["serial_baudrate"])
                if br != BAUDRATE_DEFAULT:
                    raise ValueError(f"固定 MCU 协议要求 baudrate={BAUDRATE_DEFAULT}：{br}")
                self.logger.info("单片机串口: %s @ %d", sp, br)
                self.serial_mgr = ServoCharSerial(
                    sp, br, float(self.runtime_cfg["serial_timeout_seconds"]), self.logger, serial_mod
                )
            cam_idx = int(args.camera_index if args.camera_index is not None else self.runtime_cfg["camera_device_index"])
            if cam_idx < 0:
                raise ValueError(f"摄像头编号必须 >= 0：{cam_idx}")
            self.camera_mgr = CameraManager(
                cam_idx,
                int(self.runtime_cfg["capture_width"]),
                int(self.runtime_cfg["capture_height"]),
                self.logger,
                cv2_mod,
            )

        self.state = STATE_BOOT; self.full_flag = False
        # Fixed MCU firmware sends no startup F/N. Treat boot as normal and
        # switch to FULL_PAUSED only when a real F arrives.
        self.mcu_status_known = True
        self.fault_latched = False
        self.assume_mcu_normal = bool(getattr(args, "assume_mcu_normal", False))
        self.exit_code = 0
        self.error_count = 0
        self.stable_predictor = StablePredictor(self.conf_threshold, self.stable_frames)
        self.pending_predictor = StablePredictor(self.conf_threshold, self.return_pending_frames)
        self.running = True; self.round_id = 0
        self.last_send_time = None; self.mcu_done_wait_start = None
        self.candidate_result = None
        self.roi_enabled = bool(self.runtime_cfg["roi_enabled"])
        self.roi = dict(self.runtime_cfg["roi"])
        self.last_frame_time = time.monotonic(); self.fps = 0.0
        self.serial_rx_text = "未通信"
        self.latest_result = None; self.latest_stable_info = None; self.latest_show_frame = None

    def _resolve_path(self, value: Optional[str]) -> Path:
        path = Path(value).expanduser()
        return path.resolve()

    def _update_fps(self):
        now = time.monotonic(); dt = now - self.last_frame_time; self.last_frame_time = now
        if dt > 0:
            cur = 1.0/dt; self.fps = 0.9*self.fps + 0.1*cur if self.fps > 0 else cur

    def _save_snap(self, frame, result, tag):
        ensure_dirs(self.cap_dir); ts = timestamp_str()
        if result: fn = f"r{self.round_id:04d}_{result.get('raw_class','?')}_{result.get('confidence',0):.3f}_{tag}_{ts}.jpg"
        else: fn = f"r{self.round_id:04d}_{tag}_{ts}.jpg"
        p = self.cap_dir / fn; self.cv2.imwrite(str(p), frame); self.logger.info("截图: %s", p)
        return str(p)

    def _process_mcu(self):
        if self.serial_mgr is None: return
        if (
            self.state == STATE_WAIT_MCU_DONE
            and self.mcu_done_wait_start is not None
            and time.monotonic() - self.mcu_done_wait_start > self.done_timeout
        ):
            self._enter_error(f"D 超时 {self.done_timeout}s")
            return
        try:
            data = self.serial_mgr.read_available()
        except Exception as exc:
            self._enter_error(f"串口读取失败，已停止动作: {exc}")
            return
        if not data: return
        for b in data:
            ch = chr(b)
            if ch not in VALID_RX_CHARS: continue
            self.serial_rx_text = f"RX '{ch}' / 0x{b:02X}"
            self.logger.info("[MCU] %s", self.serial_rx_text)
            if ch == 'F':
                self.full_flag = True; self.mcu_status_known = True
                self.state = STATE_FULL_PAUSED
                self.stable_predictor.reset(); self.pending_predictor.reset(); self.candidate_result = None
                self.logger.warning("满载暂停"); self._csv("mcu_full", rx_hex=hex_str(bytes([b])), mcu_event="FULL", msg="满载暂停")
            elif ch == 'N':
                if self.state == STATE_FULL_PAUSED and self.full_flag:
                    self.full_flag = False; self.mcu_status_known = True
                    self.state = STATE_ERROR_RECOVERY if self.fault_latched else STATE_IDLE_WAIT_VISUAL
                    self.stable_predictor.reset(); self.pending_predictor.reset(); self.candidate_result = None
                    self.logger.info("满载解除"); self._csv("mcu_normal", rx_hex=hex_str(bytes([b])), mcu_event="NORMAL", msg="满载解除")
                else:
                    self.logger.warning("忽略当前状态下的 N：state=%s full=%s", self.state, self.full_flag)
                    self._csv("mcu_normal_ignored", rx_hex=hex_str(bytes([b])), mcu_event="NORMAL", msg="状态不允许解除满载")
            elif ch == 'D':
                if (
                    self.state == STATE_WAIT_MCU_DONE
                    and self.mcu_done_wait_start is not None
                    and time.monotonic() - self.mcu_done_wait_start <= self.done_timeout
                ):
                    self.state = STATE_WAIT_RETURN_TO_PENDING; self.pending_predictor.reset()
                    self.logger.info("分拣完成 → WAIT_RETURN_TO_PENDING")
                    self._csv("mcu_done", rx_hex=hex_str(bytes([b])), mcu_event="DONE", msg="分拣完成")
                elif self.state == STATE_WAIT_MCU_DONE:
                    self._enter_error(f"收到迟到 D，超过 {self.done_timeout}s")
                    break
            elif ch == 'E':
                self.logger.error("MCU E"); self._enter_error("MCU 返回 E"); break

    def _csv(self, event, **kw):
        row = {"time": now_str(), "round_id": self.round_id, "event": event, "system_state": self.state}
        row.update(kw); append_csv(self.csv_path, row)

    def _enter_error(self, reason):
        self.error_count += 1
        self.state = STATE_ERROR_RECOVERY
        self.fault_latched = True
        self.running = False
        self.exit_code = 1
        self.logger.warning("ERROR_RECOVERY #%d: %s", self.error_count, reason)
        self._csv("error_recovery", message=reason)
        self.logger.error("动作已锁定，必须人工检查并重新启动程序")
        self.stable_predictor.reset(); self.pending_predictor.reset(); self.candidate_result = None

    def _send(self, result) -> bool:
        if self.fault_latched:
            self.logger.error("通信/动作故障已锁定，禁止发送")
            return False
        if not self.mcu_status_known:
            self.logger.warning("MCU 满载状态未知，禁止发送")
            return False
        if self.full_flag: self.logger.warning("满载禁止发送"); return False
        raw = result["raw_class"]
        if raw in self.no_action_classes: return False
        if raw not in self.action_mapping:
            self.logger.error("%s 不在 action_mapping", raw); return False
        ch = self.action_mapping[raw]
        now = time.monotonic()
        if self.last_send_time is not None and now - self.last_send_time < self.send_cooldown_seconds:
            self.logger.info("发送冷却中：剩余 %.2fs", self.send_cooldown_seconds - (now - self.last_send_time))
            return False
        if self.serial_mgr is None:
            self.logger.info("[No-Serial] 模拟发送: %s→%s", raw, ch)
            self.round_id += 1; self.last_send_time = now; self.candidate_result = None
            self.state = STATE_WAIT_RETURN_TO_PENDING; self.pending_predictor.reset(); return True
        try:
            tx = self.serial_mgr.send_char(ch)
            self.round_id += 1; self.last_send_time = now; self.candidate_result = None
            self.mcu_done_wait_start = now; self.state = STATE_WAIT_MCU_DONE
            self.serial_rx_text = f"TX '{ch}' / 0x{tx[0]:02X}"
            self.logger.info("发送: %s", self.serial_rx_text)
            self._csv("send", pred_id=result.get("pred_id"), raw_class=raw,
                      display_class=result.get("display_class"),
                      confidence=f"{result.get('confidence',0):.6f}",
                      tx_char=ch, tx_hex=hex_str(tx), msg="已发送")
            return True
        except Exception as e:
            self.logger.exception("发送失败，已停止后续动作")
            self._enter_error(f"发送失败: {e}")
            return False

    def run(self):
        cv2 = self.cv2
        try:
            if self.serial_mgr:
                try:
                    self.serial_mgr.open()
                except Exception as exc:
                    self._enter_error(f"串口打开失败，已停止动作: {exc}")
            if not self.running:
                return self.exit_code
            if self.camera_mgr: self.camera_mgr.open()
            if not self.args.no_window and self.camera_mgr:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(WINDOW_NAME, self.runtime_cfg["capture_width"], self.runtime_cfg["capture_height"])
            self.state = STATE_IDLE_WAIT_VISUAL
            self.logger.info("状态机启动 → %s", self.state)
            if self.serial_mgr:
                self.logger.warning(
                    "MCU 固件无启动 F/N 上报：启动默认按未满载运行；收到 F 后暂停，收到 N 后恢复"
                )
            while self.running:
                self._process_mcu()
                if not self.running:
                    break
                if self.state == STATE_WAIT_MCU_DONE:
                    time.sleep(0.005)
                    continue
                if self.camera_mgr is None: time.sleep(0.1); continue
                frame = self.camera_mgr.read(); self._update_fps()
                roi = _get_configured_roi(frame, self.roi_enabled, self.roi)
                x1, y1, x2, y2 = roi; roi_crop = frame[y1:y2, x1:x2]
                res = self.classifier.predict(roi_crop); self.latest_result = res
                raw = res["raw_class"]; conf = res["confidence"]; disp = res["display_class"]

                if self.state == STATE_IDLE_WAIT_VISUAL:
                    if raw not in self.no_action_classes:
                        si = self.stable_predictor.update(res); self.latest_stable_info = si
                        if si["is_stable"]:
                            self.candidate_result = dict(res)
                            self.state = STATE_CANDIDATE_DETECTED
                            self.logger.info("候选: %s conf=%.3f", raw, conf)
                    else: self.stable_predictor.reset(); self.latest_stable_info = {"is_stable": False, "stable_count": 0, "status": "待分拣"}
                elif self.state == STATE_CANDIDATE_DETECTED:
                    if self.candidate_result is not None:
                        self._send(self.candidate_result)
                    else:
                        self.logger.error("候选结果丢失，禁止发送")
                        self._enter_error("候选结果丢失")
                elif self.state == STATE_WAIT_MCU_DONE:
                    if time.monotonic() - self.mcu_done_wait_start > self.done_timeout:
                        self._enter_error(f"D 超时 {self.done_timeout}s")
                elif self.state == STATE_WAIT_RETURN_TO_PENDING:
                    if raw in self.no_action_classes:
                        si = self.pending_predictor.update(res); self.latest_stable_info = si
                        if si["is_stable"]:
                            self.logger.info("已回到待分拣"); self.state = STATE_IDLE_WAIT_VISUAL
                            self.stable_predictor.reset(); self.pending_predictor.reset(); self.candidate_result = None
                    else: self.pending_predictor.reset(); self.latest_stable_info = {"is_stable": False, "stable_count": 0, "status": "等待回到待分拣..."}
                elif self.state == STATE_FULL_PAUSED: self.stable_predictor.reset()
                elif self.state == STATE_ERROR_RECOVERY: pass

                st_text = self.latest_stable_info.get("status", self.state) if self.latest_stable_info else self.state
                sc = str(self.latest_stable_info.get("stable_count", 0)) if self.latest_stable_info else "0"
                show = _draw_overlay(cv2, frame, roi, st_text, disp, f"{conf:.3f}", sc,
                                     self.serial_rx_text, self.state, str(self.round_id), f"{self.fps:.1f}")
                self.latest_show_frame = show
                if not self.args.no_window:
                    cv2.imshow(WINDOW_NAME, show); k = cv2.waitKey(1) & 0xFF; self._key(k)
                else:
                    if int(time.monotonic()*2) % 4 == 0:
                        print(f"\r{now_str()} | {st_text} | {disp} conf={conf:.3f} "
                              f"stable={sc} ser={self.serial_rx_text}", end="", flush=True)
                time.sleep(FRAME_INTERVAL_SEC)
        except KeyboardInterrupt: self.logger.info("用户中断")
        except Exception: self.logger.error("主循环异常:\n%s", traceback.format_exc()); raise
        finally: self._close()
        return self.exit_code

    def _key(self, k):
        if k == 255: return
        if k == ord("q"): self.running = False
        elif k == ord("s") and self.latest_show_frame is not None:
            self._save_snap(self.latest_show_frame, self.latest_result, "manual")
        elif (
            k in (ord("r"), ord("k"), ord("h"), ord("o"))
            and self.serial_mgr
            and self.state == STATE_IDLE_WAIT_VISUAL
            and not self.full_flag
            and self.mcu_status_known
            and not self.fault_latched
        ):
            try:
                now = time.monotonic()
                if self.last_send_time is not None and now - self.last_send_time < self.send_cooldown_seconds:
                    self.logger.info("手动发送冷却中")
                    return
                ch = chr(k).upper(); tx = self.serial_mgr.send_char(ch)
                self.round_id += 1
                self.last_send_time = now
                self.mcu_done_wait_start = now
                self.state = STATE_WAIT_MCU_DONE
                self.candidate_result = None
                self.serial_rx_text = f"手动TX '{ch}' / 0x{tx[0]:02X}"
                self.logger.info("手动测试: %s", ch)
            except Exception as e:
                self.logger.error("手动发送失败，已停止后续动作: %s", e)
                self._enter_error(f"手动发送失败: {e}")

    def _close(self):
        if self.camera_mgr: self.camera_mgr.close()
        if self.serial_mgr: self.serial_mgr.close()
        if not self.args.no_window: self.cv2.destroyAllWindows()
        self.logger.info("程序退出")


# =========================================================
# Dry-run
# =========================================================

def _dummy_logger():
    l = logging.getLogger("dummy"); l.handlers.clear(); l.addHandler(logging.NullHandler()); return l


def dry_run_check(args) -> bool:
    # 确保 stdout 能输出中文（Windows GBK → UTF-8）
    try: sys.stdout.reconfigure(encoding='utf-8')
    except: pass
    print("=" * 60); print("五分类视觉触发系统 — Dry-Run 预检"); print("=" * 60)
    proot = Path(args.project_root).expanduser().resolve()
    cc = (
        Path(args.class_config).expanduser().resolve()
        if args.class_config
        else proot / "09_Vision_Trigger_5Class_System" / "config" / "class_mapping_5class.json"
    )
    print(f"[1/4] 项目根: {proot} 存在={proot.exists()}")
    print(f"[2/4] 五分类配置: {cc} 存在={cc.exists()}")
    try:
        idx, act, no_action = load_class_mapping_5class(cc, _dummy_logger())
        print(f"      类别: {list(idx.values())}"); print(f"      动作: {act}  | 不发送: {sorted(no_action)}")
    except Exception as exc:
        print(f"      错误: {exc}"); return False

    rc = None
    if args.runtime_config:
        rc_path = Path(args.runtime_config).expanduser()
        rc = rc_path.resolve()
    model_override = None
    if args.model_path:
        model_path = Path(args.model_path).expanduser()
        model_override = model_path.resolve()
    try:
        cfg = load_runtime_config(rc, _dummy_logger(), proot, model_override)
    except Exception as exc:
        print(f"[3/4] 运行时配置错误: {exc}")
        return False
    mp = Path(cfg["model_path"])
    print(f"[3/4] TFLite 模型: {mp} 存在={mp.is_file()}")
    if not mp.is_file():
        print("      错误: 模型文件不存在")
        return False
    if args.no_serial:
        print("[4/4] 串口: 已按参数跳过")
    else:
        if args.serial_port is not None and not args.serial_port.strip():
            print("[4/4] 串口检查失败")
            print("      错误: --serial-port 不能为空")
            return False
        preferred_sp = args.serial_port if args.serial_port is not None else cfg.get("serial_port")
        try:
            detected_sp = auto_detect_serial_port(preferred_sp)
            print(f"[4/4] 串口: {detected_sp} 存在=True")
        except Exception as exc:
            print("[4/4] 串口检查失败")
            print(f"      错误: {exc}")
            return False
    print("=" * 60); print("Dry-Run 完成。")
    print("树莓派运行前请确认串口设备存在并已关闭串口终端。"); return True


# =========================================================
# 测试字符模式
# =========================================================

def test_char_mode(args, serial_mod):
    l = _dummy_logger(); l.handlers.clear(); l.addHandler(logging.StreamHandler(sys.stdout)); l.setLevel(logging.INFO)
    project_root = Path(args.project_root).expanduser().resolve()
    rc_path = None
    if args.runtime_config:
        rc_value = Path(args.runtime_config).expanduser()
        rc_path = rc_value.resolve()
    cfg = load_runtime_config(rc_path, l, project_root, require_model=False)
    preferred_sp = args.serial_port if args.serial_port is not None else cfg.get("serial_port")
    sp = auto_detect_serial_port(preferred_sp)
    br = args.baudrate if args.baudrate is not None else int(cfg["serial_baudrate"])
    if br != BAUDRATE_DEFAULT:
        raise ValueError(f"固定 MCU 协议要求 baudrate={BAUDRATE_DEFAULT}：{br}")
    ser = ServoCharSerial(sp, br, float(cfg["serial_timeout_seconds"]), l, serial_mod)
    done_timeout = float(
        args.done_timeout if args.done_timeout is not None else cfg["done_timeout_seconds"]
    )
    if done_timeout <= 0.0 or not math.isfinite(done_timeout):
        raise ValueError(f"D 超时必须是有限正数：{done_timeout}")
    try:
        ser.open()
        startup_rx = ser.read_available()
        if any(ch in startup_rx for ch in b"FDE"):
            raise RuntimeError(f"MCU 启动状态异常，拒绝发送测试字符：{hex_str(startup_rx)}")
        if startup_rx:
            l.info("发送前收到: %s", hex_str(startup_rx))
        l.info("发送测试字符: %s", args.test_char)
        tx = ser.send_char(args.test_char)
        l.info("发送完成: %s", hex_str(tx))
        deadline = time.monotonic() + done_timeout
        response = bytearray()
        while time.monotonic() < deadline:
            rx = ser.read_available()
            if rx:
                response.extend(rx)
                l.info("收到: %s", hex_str(rx))
                if b"F" in rx or b"E" in rx:
                    break
                if b"D" in rx:
                    drain_deadline = time.monotonic() + 0.5
                    while time.monotonic() < drain_deadline:
                        extra = ser.read_available()
                        if extra:
                            response.extend(extra)
                            l.info("收到: %s", hex_str(extra))
                        time.sleep(0.05)
                    break
            time.sleep(0.05)
        if b"F" in response:
            raise RuntimeError("MCU 返回 F，满载，测试动作状态不安全")
        if b"E" in response:
            raise RuntimeError("MCU 返回 E，测试动作失败")
        if b"D" not in response:
            raise TimeoutError("等待 MCU D 超时，动作结果未知")
    finally: ser.close()


# =========================================================
# 命令行
# =========================================================

def parse_args():
    p = argparse.ArgumentParser(description="★ 五分类视觉触发垃圾分类分拣系统 — 最终树莓派上位机")
    p.add_argument("--project-root", default=str(PROJECT_ROOT_DEFAULT), help="项目根目录")
    p.add_argument("--model-path", default=None, help="TFLite 模型路径")
    p.add_argument("--class-config", default=None, help="class_mapping_5class.json 路径")
    p.add_argument("--runtime-config", default=None, help="runtime_config.json 路径；默认不读取 example，避免旧配置覆盖模型/串口")
    p.add_argument("--serial-port", default=None, help="串口设备")
    p.add_argument("--baudrate", type=int, default=None, help="波特率(默认使用 runtime config)")
    p.add_argument("--camera-index", type=int, default=None, help="摄像头编号(默认使用 runtime config)")
    p.add_argument("--conf-threshold", type=float, default=None, help="置信度阈值")
    p.add_argument("--stable-frames", type=int, default=None, help="稳定帧数")
    p.add_argument("--return-pending-frames", type=int, default=None, help="回到待分拣帧数")
    p.add_argument("--done-timeout", type=float, default=None, help="等待 D 超时秒数")
    p.add_argument("--no-window", action="store_true", help="不显示 OpenCV 窗口")
    p.add_argument("--no-serial", action="store_true", help="不打开串口")
    p.add_argument("--preview-only", action="store_true", help="仅预览（等同 --no-serial）")
    p.add_argument("--assume-mcu-normal", action="store_true",
                   help="兼容旧命令；当前启动默认按未满载运行")
    p.add_argument("--test-char", choices=sorted(VALID_TX_CHARS), default=None, help="发送测试字符后退出 (R/K/H/O)")
    p.add_argument("--dry-run", action="store_true", help="只检查配置不打开硬件")
    return p.parse_args()


# =========================================================
# 入口
# =========================================================

def main():
    args = parse_args()
    if args.preview_only: args.no_serial = True
    if args.test_char and (args.no_serial or args.preview_only):
        raise ValueError("--test-char 不能与 --no-serial 或 --preview-only 同时使用")
    if args.baudrate is not None and args.baudrate != BAUDRATE_DEFAULT:
        raise ValueError(f"固定 MCU 协议要求 baudrate={BAUDRATE_DEFAULT}：{args.baudrate}")
    if args.dry_run:
        ok = dry_run_check(args); sys.exit(0 if ok else 1)

    if args.test_char:
        import serial as _serial_mod
        test_char_mode(args, _serial_mod)
        return

    # 延迟导入重型依赖（保证 --help/--dry-run 无需硬件环境）
    import cv2
    import numpy as np
    import serial as _serial_mod

    try:
        from tflite_runtime.interpreter import Interpreter as _Interp
        _tflite_backend = "tflite_runtime"
    except ImportError:
        import tensorflow as tf
        _Interp = tf.lite.Interpreter
        _tflite_backend = "tensorflow"

    app = VisionTriggerSortingApp(args, cv2, np, _serial_mod, _Interp, _tflite_backend)
    sys.exit(app.run() or 0)


if __name__ == "__main__":
    main()
