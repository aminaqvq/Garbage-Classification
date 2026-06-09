import os

# =========================================================
# 0. 环境设置：必须放在 import cv2 之前
# =========================================================

# 树莓派桌面环境下，OpenCV 窗口可能需要这个
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

# 避免部分数学库线程冲突
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


import cv2
import csv
import json
import time
import logging
import traceback
import numpy as np

from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Tuple, List

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
            "没有找到 TFLite 解释器。\n"
            "树莓派建议安装：python3 -m pip install tflite-runtime\n"
            "如果用完整 TensorFlow：python3 -m pip install tensorflow"
        ) from e


# =========================================================
# 2. 基础配置
# =========================================================

from pathlib import Path

# 项目根目录，绝对路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 模型路径
MODEL_PATH = PROJECT_ROOT / "export" / "latest_tflite_fp16.tflite"

# 类别映射文件
# 如果你暂时没有 class_mapping.json，也没关系，脚本会使用默认映射
CLASS_MAPPING_PATH = PROJECT_ROOT / "config" / "class_mapping.json"

# 日志和截图目录
LOG_DIR = PROJECT_ROOT / "Logs"
CAPTURE_DIR = PROJECT_ROOT / "Captures_AI_Test"
PREDICT_LOG_CSV = LOG_DIR / "ai_preview_predictions.csv"

CAMERA_INDEX = 0
CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 480

SHOW_WINDOW = True
WINDOW_NAME = "Garbage AI Preview Test"

# 与当前训练 / 导出保持一致
RESIZE_SIZE = 256
CROP_SIZE = 224
RGB_INPUT = True

# 当前新模型按 ImageNet Normalize 走
# 你这次导出验证 PyTorch / ONNX / TFLite FP16 都是这个预处理链路
NORMALIZE = True
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# ROI 引导框比例
ROI_W_RATIO = 0.56
ROI_H_RATIO = 0.70

# 判定参数
CONF_THRESHOLD = 0.80
STABLE_FRAMES = 3

# 控制终端输出频率，避免刷屏
PRINT_INTERVAL_SEC = 0.50

# 画面刷新间隔
FRAME_INTERVAL_SEC = 0.03

# 默认类别映射：必须和你当前模型 class_to_idx 一致
DEFAULT_IDX_TO_CLASS = {
    0: "其他",
    1: "厨余",
    2: "可回收",
    3: "有害",
}

CLASS_DISPLAY_NAME = {
    "其他": "其他垃圾",
    "厨余": "厨余垃圾",
    "可回收": "可回收垃圾",
    "有害": "有害垃圾",
}

# 字体路径候选
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
]

FONT_SIZE_MAIN = 28
FONT_SIZE_SMALL = 22


# =========================================================
# 3. 日志
# =========================================================

def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def setup_logger():
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    log_path = LOG_DIR / f"ai_preview_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logger = logging.getLogger("GarbageAIPreview")
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

    logger.info("AI 识别测试启动")
    logger.info("日志文件: %s", log_path)
    logger.info("TFLite 后端: %s", TFLITE_BACKEND)

    return logger


logger = setup_logger()


def append_prediction_log(row: Dict):
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    file_exists = PREDICT_LOG_CSV.exists()

    fieldnames = [
        "time",
        "frame_id",
        "pred_id",
        "raw_class",
        "display_class",
        "confidence",
        "stable_count",
        "is_stable",
        "fps",
        "probs"
    ]

    with open(PREDICT_LOG_CSV, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


# =========================================================
# 4. 类别映射
# =========================================================

def load_idx_to_class() -> Dict[int, str]:
    """
    优先从 class_mapping.json 读取。
    如果没有，则使用当前项目默认映射。
    """
    if CLASS_MAPPING_PATH.exists():
        try:
            with open(CLASS_MAPPING_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)

            idx_to_class = data.get("idx_to_class", None)

            if isinstance(idx_to_class, dict):
                result = {int(k): str(v) for k, v in idx_to_class.items()}
                logger.info("已从 class_mapping.json 读取 idx_to_class: %s", result)
                return result

            class_to_idx = data.get("class_to_idx", None)

            if isinstance(class_to_idx, dict):
                result = {int(v): str(k) for k, v in class_to_idx.items()}
                logger.info("已从 class_mapping.json 读取 class_to_idx 并反转: %s", result)
                return result

        except Exception as e:
            logger.warning("读取 class_mapping.json 失败，将使用默认映射: %s", e)

    logger.info("使用默认类别映射: %s", DEFAULT_IDX_TO_CLASS)
    return DEFAULT_IDX_TO_CLASS.copy()


def get_display_name(raw_class: str) -> str:
    return CLASS_DISPLAY_NAME.get(raw_class, raw_class)


# =========================================================
# 5. 图像与显示工具
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

    right = left + crop_size
    bottom = top + crop_size

    crop = img_rgb[top:bottom, left:right]

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
    line_gap: int = 34,
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
    fps_text: str,
    color=(0, 255, 0),
) -> np.ndarray:
    x1, y1, x2, y2 = roi_box

    show = frame_bgr.copy()

    cv2.rectangle(show, (x1, y1), (x2, y2), color, 2)

    # 半透明信息面板
    panel_x1, panel_y1 = 12, 12
    panel_x2, panel_y2 = 430, 210

    overlay = show.copy()
    cv2.rectangle(overlay, (panel_x1, panel_y1), (panel_x2, panel_y2), (0, 0, 0), -1)
    show = cv2.addWeighted(overlay, 0.45, show, 0.55, 0)

    lines = [
        f"状态：{status_text}",
        f"类别：{class_text}",
        f"置信度：{conf_text}",
        f"稳定帧：{stable_text}",
        f"FPS：{fps_text}",
    ]

    if FONT_MAIN is None:
        y = 38
        for line in lines:
            safe_line = line.encode("ascii", errors="replace").decode("ascii")
            cv2.putText(
                show,
                safe_line,
                (24, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (0, 255, 0),
                2
            )
            y += 34

        cv2.putText(
            show,
            "ROI",
            (x1 + 8, y1 - 8 if y1 > 25 else y1 + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            color,
            2
        )
        return show

    show = draw_text_pil(
        show,
        lines,
        x=24,
        y=24,
        color_rgb=(255, 255, 255),
        line_gap=36,
        font=FONT_MAIN
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
# 6. TFLite 分类器
# =========================================================

class GarbageClassifierTFLite:
    def __init__(self, model_path: Path, idx_to_class: Dict[int, str]):
        if not model_path.exists():
            raise FileNotFoundError(f"TFLite 模型不存在: {model_path}")

        self.model_path = model_path
        self.idx_to_class = idx_to_class

        logger.info("加载模型: %s", model_path)

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
        logger.info("模型预热...")

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

        x = quantize_tensor_if_needed(x.astype(np.float32), self.input_detail)

        return x

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
# 7. 稳定判断器
# =========================================================

class StablePredictor:
    def __init__(self, conf_threshold: float, stable_frames: int):
        self.conf_threshold = conf_threshold
        self.stable_frames = stable_frames

        self.last_pred_id = None
        self.stable_count = 0
        self.best_result = None

    def update(self, result: Dict) -> Dict:
        pred_id = result["pred_id"]
        confidence = result["confidence"]

        if confidence < self.conf_threshold:
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

        self.best_result = result

        is_stable = self.stable_count >= self.stable_frames

        if is_stable:
            status = "识别稳定"
            color = (0, 255, 0)
        else:
            status = "识别中"
            color = (255, 255, 0)

        return {
            "is_stable": is_stable,
            "stable_count": self.stable_count,
            "status": status,
            "color": color,
        }


# =========================================================
# 8. 摄像头管理
# =========================================================

class CameraManager:
    def __init__(self, cam_index: int, width: int, height: int):
        logger.info("打开摄像头 index=%s", cam_index)

        self.cap = cv2.VideoCapture(cam_index)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开摄像头: {cam_index}")

        for _ in range(10):
            self.cap.read()
            time.sleep(0.03)

        logger.info("摄像头打开成功，目标分辨率=%dx%d", width, height)

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
# 9. 主程序
# =========================================================

def save_snapshot(frame_bgr, result: Dict, frame_id: int):
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    class_name = result.get("raw_class", "unknown")
    conf = result.get("confidence", 0.0)

    filename = f"frame_{frame_id:06d}_{class_name}_{conf:.3f}_{ts}.jpg"
    path = CAPTURE_DIR / filename

    cv2.imwrite(str(path), frame_bgr)

    logger.info("已保存截图: %s", path)


def main():
    logger.info("========== 第二阶段：AI 模型识别预览测试 ==========")
    logger.info("PROJECT_ROOT: %s", PROJECT_ROOT)
    logger.info("MODEL_PATH: %s", MODEL_PATH)
    logger.info("CLASS_MAPPING_PATH: %s", CLASS_MAPPING_PATH)
    logger.info("NORMALIZE: %s", NORMALIZE)
    logger.info("MEAN: %s", MEAN)
    logger.info("STD: %s", STD)

    idx_to_class = load_idx_to_class()

    classifier = GarbageClassifierTFLite(MODEL_PATH, idx_to_class)
    stable_predictor = StablePredictor(CONF_THRESHOLD, STABLE_FRAMES)
    camera = CameraManager(CAMERA_INDEX, CAPTURE_WIDTH, CAPTURE_HEIGHT)

    if SHOW_WINDOW:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, CAPTURE_WIDTH, CAPTURE_HEIGHT)

    logger.info("按键说明：q 退出 | s 保存当前画面")
    logger.info("开始识别预览。")

    frame_id = 0
    last_print_time = 0.0
    last_frame_time = time.time()
    fps = 0.0
    latest_result = None

    try:
        while True:
            loop_start = time.time()

            frame = camera.read()
            frame_id += 1

            roi_box = get_center_roi(frame)
            x1, y1, x2, y2 = roi_box
            roi = frame[y1:y2, x1:x2]

            result = classifier.predict(roi)
            latest_result = result

            stable_info = stable_predictor.update(result)

            now = time.time()
            dt = now - last_frame_time
            last_frame_time = now

            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else (1.0 / dt)

            raw_class = result["raw_class"]
            display_class = result["display_class"]
            confidence = result["confidence"]

            status_text = stable_info["status"]
            class_text = f"{display_class} ({raw_class})"
            conf_text = f"{confidence:.3f}"
            stable_text = f"{stable_info['stable_count']} / {STABLE_FRAMES}"
            fps_text = f"{fps:.1f}"

            if now - last_print_time >= PRINT_INTERVAL_SEC:
                print(
                    f"[{now_str()}] "
                    f"frame={frame_id} | "
                    f"class={display_class}({raw_class}) | "
                    f"conf={confidence:.3f} | "
                    f"stable={stable_info['stable_count']}/{STABLE_FRAMES} | "
                    f"status={status_text} | "
                    f"fps={fps:.1f}"
                )

                append_prediction_log({
                    "time": now_str(),
                    "frame_id": frame_id,
                    "pred_id": result["pred_id"],
                    "raw_class": raw_class,
                    "display_class": display_class,
                    "confidence": confidence,
                    "stable_count": stable_info["stable_count"],
                    "is_stable": int(stable_info["is_stable"]),
                    "fps": fps,
                    "probs": json.dumps(result["probs"], ensure_ascii=False),
                })

                last_print_time = now

            if SHOW_WINDOW:
                show = draw_overlay(
                    frame_bgr=frame,
                    roi_box=roi_box,
                    status_text=status_text,
                    class_text=class_text,
                    conf_text=conf_text,
                    stable_text=stable_text,
                    fps_text=fps_text,
                    color=stable_info["color"],
                )

                cv2.imshow(WINDOW_NAME, show)

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    logger.info("用户按 q 退出")
                    break

                if key == ord("s"):
                    save_snapshot(show, result, frame_id)

            elapsed = time.time() - loop_start
            sleep_time = FRAME_INTERVAL_SEC - elapsed

            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        logger.info("用户 Ctrl+C 退出")

    except Exception as e:
        logger.error("程序异常: %s", e)
        logger.error(traceback.format_exc())

    finally:
        camera.close()

        if SHOW_WINDOW:
            cv2.destroyAllWindows()

        logger.info("第二阶段 AI 识别测试结束")
        logger.info("预测日志: %s", PREDICT_LOG_CSV)


if __name__ == "__main__":
    main()