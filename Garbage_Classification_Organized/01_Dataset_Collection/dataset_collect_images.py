import argparse
import csv
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


# =========================================================
# 项目路径
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT
    / "09_Vision_Trigger_5Class_System"
    / "config"
    / "class_mapping_5class.json"
)

DEFAULT_ROI_CONFIG_PATH = (
    PROJECT_ROOT
    / "09_Vision_Trigger_5Class_System"
    / "config"
    / "roi_config.json"
)


# =========================================================
# 基础配置
# =========================================================

DATASET_ROOT = str(PROJECT_ROOT / "dataset")

CAMERA_INDEX = 1
CAPTURE_INTERVAL = 1.5

DEFAULT_SHARPNESS_THRESHOLD = 100

MIN_BRIGHTNESS = 40
MAX_BRIGHTNESS = 220

IMAGE_EXT = ".jpg"
JPEG_QUALITY = 95

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

CHINESE_FONT_PATH = None

IMAGE_SUFFIXES = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

MIN_ROI_SIZE = 32


# =========================================================
# ROI 数据结构
# =========================================================

@dataclass
class SquareROI:
    x: int
    y: int
    size: int

    def as_dict(self):
        return {
            "x": int(self.x),
            "y": int(self.y),
            "size": int(self.size),
        }

    def clamp(self, frame_width: int, frame_height: int) -> "SquareROI":
        size = max(MIN_ROI_SIZE, int(self.size))
        size = min(size, frame_width, frame_height)

        x = max(0, min(int(self.x), frame_width - size))
        y = max(0, min(int(self.y), frame_height - size))

        return SquareROI(x=x, y=y, size=size)


class ROISelector:
    """
    鼠标拖拽选择正方形 ROI。

    操作：
    - 左键按下：开始选择
    - 拖动：显示预览 ROI
    - 左键松开：确定 ROI
    """

    def __init__(self, roi: Optional[SquareROI] = None):
        self.roi = roi
        self.preview_roi = None
        self.dragging = False
        self.start_x = 0
        self.start_y = 0
        self.frame_width = CAMERA_WIDTH
        self.frame_height = CAMERA_HEIGHT

    def set_frame_shape(self, frame_shape):
        h, w = frame_shape[:2]
        self.frame_width = int(w)
        self.frame_height = int(h)

        if self.roi is not None:
            self.roi = self.roi.clamp(self.frame_width, self.frame_height)

    def get_active_roi(self) -> Optional[SquareROI]:
        if self.dragging and self.preview_roi is not None:
            return self.preview_roi

        return self.roi

    def clear(self):
        self.roi = None
        self.preview_roi = None
        self.dragging = False

    def _make_square_roi(self, x1: int, y1: int, x2: int, y2: int) -> Optional[SquareROI]:
        dx = x2 - x1
        dy = y2 - y1

        side = min(abs(dx), abs(dy))

        if side < MIN_ROI_SIZE:
            return None

        sx = 1 if dx >= 0 else -1
        sy = 1 if dy >= 0 else -1

        x_end = x1 + sx * side
        y_end = y1 + sy * side

        left = min(x1, x_end)
        top = min(y1, y_end)

        roi = SquareROI(x=left, y=top, size=side)
        return roi.clamp(self.frame_width, self.frame_height)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.start_x = int(x)
            self.start_y = int(y)
            self.preview_roi = None

        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.preview_roi = self._make_square_roi(
                self.start_x,
                self.start_y,
                int(x),
                int(y),
            )

        elif event == cv2.EVENT_LBUTTONUP:
            if self.dragging:
                final_roi = self._make_square_roi(
                    self.start_x,
                    self.start_y,
                    int(x),
                    int(y),
                )

                if final_roi is not None:
                    self.roi = final_roi
                    print(
                        f"\n已设置 ROI：x={final_roi.x}, "
                        f"y={final_roi.y}, size={final_roi.size}"
                    )
                    print("按 O 可保存 ROI 配置，下次自动加载。")
                else:
                    print(f"\nROI 太小，至少需要 {MIN_ROI_SIZE}x{MIN_ROI_SIZE}。")

            self.dragging = False
            self.preview_roi = None


# =========================================================
# 五分类配置读取
# =========================================================

def load_class_mapping(config_path=None):
    """
    从 class_mapping_5class.json 读取五分类配置。
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    config_path = Path(config_path).expanduser().resolve()

    if not config_path.exists():
        raise FileNotFoundError(
            f"五分类配置文件不存在：{config_path}\n"
            f"请确认 09_Vision_Trigger_5Class_System/config/class_mapping_5class.json 已创建。"
        )

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"配置文件 JSON 格式非法：{config_path}\n{e}") from e

    class_to_idx = data.get("class_to_idx")
    if not isinstance(class_to_idx, dict):
        raise ValueError(f"配置文件中缺少有效的 class_to_idx：{config_path}")

    sorted_pairs = sorted(class_to_idx.items(), key=lambda kv: kv[1])
    class_names = [name for name, _ in sorted_pairs]

    expected = {"待分拣", "其他", "厨余", "可回收", "有害"}
    actual = set(class_names)

    if len(class_names) != 5:
        raise ValueError(
            f"配置文件类别数应为 5，实际为 {len(class_names)}：{class_names}"
        )

    if actual != expected:
        missing = expected - actual
        extra = actual - expected

        msg_parts = []
        if missing:
            msg_parts.append(f"缺少类别：{missing}")
        if extra:
            msg_parts.append(f"多余类别：{extra}")

        raise ValueError(f"配置文件类别不匹配：{'；'.join(msg_parts)}")

    return class_names, class_to_idx


# =========================================================
# ROI 配置读写
# =========================================================

def load_roi_config(path: Path) -> Optional[SquareROI]:
    path = Path(path).expanduser().resolve()

    if not path.exists():
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"警告：ROI 配置读取失败，将忽略：{path}")
        print(f"原因：{e}")
        return None

    roi_data = data.get("roi", data)

    try:
        x = int(roi_data["x"])
        y = int(roi_data["y"])
        size = int(roi_data["size"])
    except Exception:
        print(f"警告：ROI 配置格式不正确，将忽略：{path}")
        return None

    if size < MIN_ROI_SIZE:
        print(f"警告：ROI size 过小，将忽略：{size}")
        return None

    return SquareROI(x=x, y=y, size=size)


def save_roi_config(path: Path, roi: SquareROI, frame_width: int, frame_height: int):
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "version": "roi_config_v1",
        "note": "正方形 ROI，用于采集数据集时裁剪原始摄像头画面。保存的图片不包含预览文字、黑色面板或 ROI 红框。",
        "frame": {
            "width": int(frame_width),
            "height": int(frame_height),
        },
        "roi": roi.as_dict(),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\nROI 配置已保存：{path}")
    print(f"ROI：x={roi.x}, y={roi.y}, size={roi.size}")


def parse_roi_arg(values) -> Optional[SquareROI]:
    if values is None:
        return None

    if len(values) != 3:
        raise ValueError("--roi 需要 3 个参数：x y size")

    x, y, size = map(int, values)

    if x < 0 or y < 0 or size < MIN_ROI_SIZE:
        raise ValueError(
            f"ROI 参数非法：x={x}, y={y}, size={size}，"
            f"size 至少为 {MIN_ROI_SIZE}"
        )

    return SquareROI(x=x, y=y, size=size)


# =========================================================
# 工具函数
# =========================================================

def create_dataset_folders(root_dir: Path, class_names):
    root_dir.mkdir(parents=True, exist_ok=True)

    for class_name in class_names:
        class_dir = root_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)


def safe_filename(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|]', "_", name)


def count_images_in_class(class_dir: Path) -> int:
    if not class_dir.exists():
        return 0

    count = 0

    for file in class_dir.iterdir():
        if file.is_file() and file.suffix.lower() in IMAGE_SUFFIXES:
            count += 1

    return count


def count_all_classes(root_dir: Path, class_names) -> dict:
    result = {}

    for class_name in class_names:
        class_dir = root_dir / class_name
        result[class_name] = count_images_in_class(class_dir)

    return result


def get_next_index_by_max_number(class_dir: Path, class_name: str, ext: str) -> int:
    safe_name = safe_filename(class_name)

    pattern = re.compile(
        rf"^{re.escape(safe_name)}_(\d+){re.escape(ext)}$",
        re.IGNORECASE,
    )

    max_index = 0

    if not class_dir.exists():
        return 1

    for file in class_dir.iterdir():
        if not file.is_file():
            continue

        if file.suffix.lower() != ext.lower():
            continue

        match = pattern.match(file.name)
        if match:
            number = int(match.group(1))
            max_index = max(max_index, number)

    return max_index + 1


def get_unique_file_path(
    class_dir: Path,
    class_name: str,
    start_index: int,
    ext: str,
):
    safe_name = safe_filename(class_name)
    index = start_index

    while True:
        file_name = f"{safe_name}_{index:04d}{ext}"
        file_path = class_dir / file_name

        if not file_path.exists():
            return file_path, index

        index += 1


def crop_by_roi(frame, roi: SquareROI):
    h, w = frame.shape[:2]
    roi = roi.clamp(w, h)

    crop = frame[roi.y: roi.y + roi.size, roi.x: roi.x + roi.size]

    if crop is None or crop.size == 0:
        return None

    return crop.copy()


def maybe_resize_square(image, output_size: int):
    if output_size is None or output_size <= 0:
        return image

    return cv2.resize(
        image,
        (int(output_size), int(output_size)),
        interpolation=cv2.INTER_AREA,
    )


def calc_sharpness(frame) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    return float(sharpness)


def calc_brightness(frame) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    return float(brightness)


def save_image_unicode(path: Path, frame) -> bool:
    encode_params = []

    if path.suffix.lower() in [".jpg", ".jpeg"]:
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]

    success, encoded_img = cv2.imencode(
        path.suffix,
        frame,
        encode_params,
    )

    if success:
        encoded_img.tofile(str(path))
        return True

    return False


def write_log(log_path: Path, row: dict):
    file_exists = log_path.exists()

    fieldnames = [
        "time",
        "class_name",
        "file_path",
        "sharpness",
        "brightness",
        "capture_type",
        "save_mode",
        "roi_x",
        "roi_y",
        "roi_size",
        "saved_width",
        "saved_height",
    ]

    with open(log_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        safe_row = {key: row.get(key, "") for key in fieldnames}
        writer.writerow(safe_row)


def select_class(class_names):
    print("\n请选择当前要采集的垃圾类别：")

    for index, class_name in enumerate(class_names, start=1):
        print(f"{index}. {class_name}")

    while True:
        user_input = input("\n请输入编号或类别名：").strip()

        if user_input.isdigit():
            index = int(user_input)
            if 1 <= index <= len(class_names):
                return class_names[index - 1]

        if user_input in class_names:
            return user_input

        print("输入无效，请重新输入。")


def get_chinese_font(font_size=24):
    font_candidates = []

    if CHINESE_FONT_PATH:
        font_candidates.append(CHINESE_FONT_PATH)

    font_candidates.extend([
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    ])

    for font_path in font_candidates:
        if font_path and os.path.exists(font_path):
            return ImageFont.truetype(font_path, font_size)

    return ImageFont.load_default()


def get_status_text(started: bool, paused: bool) -> str:
    if not started:
        return "预览中，未开始采集"

    if paused:
        return "已暂停"

    return "正在采集"


def get_brightness_status(brightness: float) -> str:
    if brightness < MIN_BRIGHTNESS:
        return "过暗"
    elif brightness > MAX_BRIGHTNESS:
        return "过亮"
    else:
        return "正常"


def get_quality_status(
    sharpness: float,
    sharpness_threshold: float,
    brightness: float,
) -> str:
    sharp_ok = sharpness >= sharpness_threshold
    bright_ok = MIN_BRIGHTNESS <= brightness <= MAX_BRIGHTNESS

    if sharp_ok and bright_ok:
        return "画面合格，可保存"

    if not sharp_ok and not bright_ok:
        return "清晰度和亮度不合格"

    if not sharp_ok:
        return "清晰度不足"

    return "亮度不合格"


def draw_info_panel(
    frame,
    class_name: str,
    class_counts: dict,
    saved_count_this_time: int,
    target_count,
    sharpness: float,
    sharpness_threshold: float,
    brightness: float,
    capture_interval: float,
    started: bool,
    paused: bool,
    next_file_name: str,
    class_names,
    save_mode_text: str,
):
    """
    兼容旧版信息面板。

    注意：
    - 默认不显示这个面板。
    - 只有传入 --show-panel 或按 V 切换时才显示。
    - 保存图片永远不会保存这个面板。
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb).convert("RGBA")

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    font_title = get_chinese_font(30)
    font_text = get_chinese_font(23)
    font_small = get_chinese_font(20)

    panel_x = 15
    panel_y = 15
    panel_w = 560

    num_classes = len(class_names)
    panel_h = 405 + num_classes * 32

    draw.rounded_rectangle(
        [panel_x, panel_y, panel_x + panel_w, panel_y + panel_h],
        radius=18,
        fill=(0, 0, 0, 165),
    )

    x = panel_x + 22
    y = panel_y + 18

    status_text = get_status_text(started, paused)
    brightness_status = get_brightness_status(brightness)
    quality_status = get_quality_status(
        sharpness,
        sharpness_threshold,
        brightness,
    )

    if target_count is None:
        target_text = "不限制"
    else:
        target_text = str(target_count)

    total_count = sum(class_counts.values())

    lines = [
        ("垃圾分类图像采集", font_title),
        (f"当前类别：{class_name}", font_text),
        (f"保存模式：{save_mode_text}", font_text),
        (f"运行状态：{status_text}", font_text),
        (f"本次已保存：{saved_count_this_time} / {target_text}", font_text),
        (f"拍照间隔：{capture_interval:.2f} 秒", font_text),
        (f"清晰度：{sharpness:.1f} / 阈值 {sharpness_threshold:.1f}", font_text),
        (f"亮度：{brightness:.1f} / {brightness_status}", font_text),
        (f"画面判断：{quality_status}", font_text),
        (f"下一张文件：{next_file_name}", font_small),
        (f"数据集总数：{total_count} 张", font_text),
        ("各类图片数量：", font_text),
    ]

    for text, font in lines:
        draw.text((x, y), text, font=font, fill=(255, 255, 255, 255))
        y += 35

    for cname in class_names:
        count = class_counts.get(cname, 0)
        mark = " ← 当前" if cname == class_name else ""

        draw.text(
            (x + 20, y),
            f"{cname}：{count} 张{mark}",
            font=font_text,
            fill=(255, 255, 255, 255),
        )

        y += 32

    image = Image.alpha_composite(image, overlay).convert("RGB")
    frame_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    return frame_bgr


def draw_roi_overlay(frame, roi: Optional[SquareROI], roi_required: bool):
    """
    默认干净预览：不画黑色文字背景，只画 ROI 框。
    """
    out = frame.copy()

    if roi is not None:
        x, y, s = roi.x, roi.y, roi.size

        cv2.rectangle(
            out,
            (x, y),
            (x + s, y + s),
            (0, 0, 255),
            3,
        )

        cv2.putText(
            out,
            "ROI",
            (x, max(25, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            out,
            f"{x},{y},{s}",
            (x, y + s + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    elif roi_required:
        cv2.putText(
            out,
            "Drag mouse to select square ROI",
            (30, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    return out


def print_pending_tip_for_class(class_name: str):
    if class_name == "待分拣":
        print()
        print("=" * 64)
        print("提示：待分拣类用于表示空平台/等待状态。")
        print("请采集：空平台、不同光照、轻微阴影、手刚离开、分拣后残留状态。")
        print("不要只采一张完全空白背景，否则模型泛化会很差。")
        print("=" * 64)


def prepare_frame_for_saving(frame, roi: Optional[SquareROI], args):
    """
    返回真正要保存的图片。

    关键点：
    - 默认保存 ROI crop。
    - 保存内容来自原始 frame。
    - 不保存预览文字。
    - 不保存黑色背景。
    - 不保存红色 ROI 框。
    """
    if args.save_full_frame:
        output = frame.copy()
        save_mode = "full_frame"
        used_roi = None
    else:
        if roi is None:
            return None, "roi_missing", None

        output = crop_by_roi(frame, roi)
        if output is None:
            return None, "roi_invalid", roi

        output = maybe_resize_square(output, args.roi_output_size)
        save_mode = "roi_crop"
        used_roi = roi

    return output, save_mode, used_roi


# =========================================================
# 命令行参数
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="垃圾分类图像采集程序（五分类 + 正方形 ROI 裁剪版）"
    )

    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help=f"五分类配置文件路径。默认：{DEFAULT_CONFIG_PATH}",
    )

    parser.add_argument(
        "--dataset-dir",
        default=DATASET_ROOT,
        help=f"原始采集数据输出目录。默认：{DATASET_ROOT}",
    )

    parser.add_argument(
        "--class-name",
        default=None,
        help="指定要采集的类别名，例如 待分拣、可回收。不指定则交互选择。",
    )

    parser.add_argument(
        "--list-classes",
        action="store_true",
        help="只打印可采集类别并退出。",
    )

    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="本次最多采集图片数量。不指定则可交互输入。",
    )

    parser.add_argument(
        "--camera-index",
        type=int,
        default=CAMERA_INDEX,
        help=f"摄像头编号。默认：{CAMERA_INDEX}",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=CAMERA_WIDTH,
        help=f"摄像头宽度。默认：{CAMERA_WIDTH}",
    )

    parser.add_argument(
        "--height",
        type=int,
        default=CAMERA_HEIGHT,
        help=f"摄像头高度。默认：{CAMERA_HEIGHT}",
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=CAPTURE_INTERVAL,
        help=f"自动采集间隔秒数。默认：{CAPTURE_INTERVAL}",
    )

    parser.add_argument(
        "--sharpness-threshold",
        type=float,
        default=DEFAULT_SHARPNESS_THRESHOLD,
        help=f"清晰度阈值。默认：{DEFAULT_SHARPNESS_THRESHOLD}",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只显示配置和输出目录，不打开摄像头、不写图片。",
    )

    parser.add_argument(
        "--roi",
        nargs=3,
        metavar=("X", "Y", "SIZE"),
        help="直接指定正方形 ROI，例如 --roi 610 80 580",
    )

    parser.add_argument(
        "--roi-config",
        default=str(DEFAULT_ROI_CONFIG_PATH),
        help=f"ROI 配置文件路径。默认：{DEFAULT_ROI_CONFIG_PATH}",
    )

    parser.add_argument(
        "--roi-output-size",
        type=int,
        default=0,
        help="保存 ROI 图时是否缩放到固定大小。例如 224。默认 0 表示不缩放。",
    )

    parser.add_argument(
        "--save-full-frame",
        action="store_true",
        help="保存整帧图片。默认关闭。关闭时保存 ROI 裁剪图。",
    )

    parser.add_argument(
        "--show-panel",
        action="store_true",
        help="显示旧版黑色信息面板。默认不显示，保持干净预览。",
    )

    parser.add_argument(
        "--no-interactive-prompts",
        action="store_true",
        help="不询问采集张数、清晰度阈值、拍照间隔，直接使用命令行参数。",
    )

    return parser.parse_args()


# =========================================================
# 主程序
# =========================================================

def main():
    args = parse_args()

    config_path = Path(args.config).expanduser().resolve()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    roi_config_path = Path(args.roi_config).expanduser().resolve()

    try:
        class_names, class_to_idx = load_class_mapping(config_path)
    except Exception as e:
        print("错误：无法加载五分类配置文件。")
        print(e)
        sys.exit(1)

    if args.list_classes:
        print("当前五分类类别：")
        for i, name in enumerate(class_names):
            print(f"  {i}: {name}")
        sys.exit(0)

    # 延迟导入重型依赖 — 只有真正采集时才加载 GUI 库
    global cv2, np, Image, ImageDraw, ImageFont
    try:
        import cv2
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as e:
        print(f"缺少依赖：{e}")
        print("请安装：pip install opencv-python numpy pillow")
        sys.exit(1)

    if args.class_name is not None and args.class_name not in class_names:
        print(f"错误：未知类别 '{args.class_name}'。")
        print("合法类别：")
        for name in class_names:
            print(f"  - {name}")
        sys.exit(1)

    try:
        cli_roi = parse_roi_arg(args.roi)
    except Exception as e:
        print(f"错误：ROI 参数非法：{e}")
        sys.exit(1)

    loaded_roi = cli_roi
    if loaded_roi is None:
        loaded_roi = load_roi_config(roi_config_path)

    print("========== 垃圾分类图像采集程序（五分类 + ROI 裁剪版）==========")
    print(f"配置文件：{config_path}")
    print(f"当前五分类类别：{class_names}")
    print(f"类别数量：{len(class_names)}")
    print(f"输出目录：{dataset_dir}")
    print(f"摄像头编号：{args.camera_index}")
    print(f"摄像头分辨率：{args.width}x{args.height}")
    print(f"保存模式：{'整帧 full frame' if args.save_full_frame else 'ROI 裁剪图 roi crop'}")
    print(f"ROI 配置文件：{roi_config_path}")

    if loaded_roi is not None:
        print(
            f"已加载 ROI：x={loaded_roi.x}, "
            f"y={loaded_roi.y}, size={loaded_roi.size}"
        )
    else:
        if not args.save_full_frame:
            print("当前未设置 ROI。启动摄像头后，请用鼠标左键拖拽一个正方形 ROI。")

    if args.roi_output_size and args.roi_output_size > 0:
        print(f"ROI 保存时将缩放为：{args.roi_output_size}x{args.roi_output_size}")
    else:
        print("ROI 保存时不额外缩放，保留裁剪区域原始分辨率。")

    if args.class_name:
        class_name = args.class_name
        print(f"指定采集类别：{class_name}")
    else:
        class_name = None
        print("未指定 --class-name，非 dry-run 模式下将交互选择。")

    if class_name == "待分拣":
        print_pending_tip_for_class(class_name)

    if args.dry_run:
        print()
        print("[DRY-RUN] 不会打开摄像头。")
        print("[DRY-RUN] 不会创建目录。")
        print("[DRY-RUN] 不会写入图片。")
        print("[DRY-RUN] 不会保存 ROI 配置。")
        print(f"[DRY-RUN] 采集类别：{class_name or '(交互选择)'}")
        print(f"[DRY-RUN] 目标数量：{args.count or '不限制'}")
        print(f"[DRY-RUN] 采集间隔：{args.interval}s")
        print(f"[DRY-RUN] 清晰度阈值：{args.sharpness_threshold}")
        print(f"[DRY-RUN] 保存模式：{'full_frame' if args.save_full_frame else 'roi_crop'}")
        if loaded_roi is not None:
            print(f"[DRY-RUN] ROI：{loaded_roi.as_dict()}")
        else:
            print("[DRY-RUN] ROI：未设置")
        print("[DRY-RUN] 验证通过。")
        sys.exit(0)

    create_dataset_folders(dataset_dir, class_names)

    print(f"\n数据集目录已创建：{dataset_dir}")
    print("各类目录：")
    for name in class_names:
        print(f"- {dataset_dir / name}")

    if class_name is None:
        class_name = select_class(class_names)

    print_pending_tip_for_class(class_name)

    class_dir = dataset_dir / class_name

    target_count = args.count
    sharpness_threshold = args.sharpness_threshold
    capture_interval = args.interval

    if not args.no_interactive_prompts:
        if target_count is None:
            target_input = input("\n请输入本次计划采集张数，直接回车表示不限制：").strip()
            target_count = int(target_input) if target_input.isdigit() else None

        threshold_input = input(
            f"\n请输入清晰度阈值，直接回车使用默认值 {sharpness_threshold}："
        ).strip()

        if threshold_input:
            sharpness_threshold = float(threshold_input)

        interval_input = input(
            f"\n请输入拍照间隔秒数，直接回车使用默认值 {capture_interval}："
        ).strip()

        if interval_input:
            capture_interval = float(interval_input)

    next_index = get_next_index_by_max_number(
        class_dir,
        class_name,
        IMAGE_EXT,
    )

    saved_count_this_time = 0
    log_path = dataset_dir / "capture_log.csv"

    print("\n正在打开摄像头...")
    cap = cv2.VideoCapture(args.camera_index)

    if not cap.isOpened():
        print("摄像头打开失败。")
        print("请检查：")
        print("1. 摄像头是否被其他软件占用")
        print("2. --camera-index 是否正确，可以尝试 0、1、2")
        print("3. 当前系统是否允许 Python / 终端访问摄像头")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    window_name = "Garbage Image Capture - ROI Crop"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    selector = ROISelector(roi=loaded_roi)
    cv2.setMouseCallback(window_name, selector.mouse_callback)

    print("\n摄像头已打开。")
    print("当前是预览模式，不会自动保存图片。")
    print()
    print("鼠标操作：")
    print("  左键拖拽：选择正方形 ROI")
    print()
    print("键盘操作：")
    print("  空格：开始采集")
    print("  P：暂停 / 继续")
    print("  S：手动保存一张")
    print("  O：保存当前 ROI 到配置文件")
    print("  C：清除当前 ROI")
    print("  V：显示 / 隐藏旧版文字信息面板")
    print("  +：提高清晰度阈值")
    print("  -：降低清晰度阈值")
    print("  Q：退出")
    print()
    print("重要：保存的图片来自原始 frame 的 ROI 裁剪，不包含文字、黑色背景或 ROI 红框。")
    print("注意：按键前请先点击摄像头预览窗口，让窗口获得焦点。")

    started = False
    paused = False
    show_panel = bool(args.show_panel)

    last_capture_time = time.time()

    class_counts = count_all_classes(dataset_dir, class_names)
    last_count_update_time = 0

    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            print("读取摄像头画面失败。")
            break

        selector.set_frame_shape(frame.shape)

        current_time = time.time()

        active_roi = selector.get_active_roi()

        save_frame_for_quality, save_mode_for_quality, used_roi_for_quality = prepare_frame_for_saving(
            frame,
            active_roi,
            args,
        )

        if save_frame_for_quality is not None:
            sharpness = calc_sharpness(save_frame_for_quality)
            brightness = calc_brightness(save_frame_for_quality)
        else:
            sharpness = 0.0
            brightness = 0.0

        if current_time - last_count_update_time >= 1:
            class_counts = count_all_classes(dataset_dir, class_names)
            last_count_update_time = current_time

        next_file_path, preview_next_index = get_unique_file_path(
            class_dir,
            class_name,
            next_index,
            IMAGE_EXT,
        )
        next_file_name = next_file_path.name

        save_mode_text = "整帧" if args.save_full_frame else "ROI裁剪"

        display_frame = frame.copy()

        if show_panel:
            display_frame = draw_info_panel(
                frame=display_frame,
                class_name=class_name,
                class_counts=class_counts,
                saved_count_this_time=saved_count_this_time,
                target_count=target_count,
                sharpness=sharpness,
                sharpness_threshold=sharpness_threshold,
                brightness=brightness,
                capture_interval=capture_interval,
                started=started,
                paused=paused,
                next_file_name=next_file_name,
                class_names=class_names,
                save_mode_text=save_mode_text,
            )

        display_frame = draw_roi_overlay(
            display_frame,
            active_roi,
            roi_required=not args.save_full_frame,
        )

        cv2.imshow(window_name, display_frame)

        has_valid_save_region = args.save_full_frame or active_roi is not None

        can_auto_capture = (
            has_valid_save_region
            and started
            and not paused
            and current_time - last_capture_time >= capture_interval
            and sharpness >= sharpness_threshold
            and MIN_BRIGHTNESS <= brightness <= MAX_BRIGHTNESS
        )

        if can_auto_capture:
            image_to_save, save_mode, used_roi = prepare_frame_for_saving(
                frame,
                active_roi,
                args,
            )

            if image_to_save is None:
                print("当前没有有效 ROI，无法保存。请先用鼠标拖拽选择正方形 ROI。")
                last_capture_time = current_time
            else:
                file_path, real_index = get_unique_file_path(
                    class_dir,
                    class_name,
                    next_index,
                    IMAGE_EXT,
                )

                if save_image_unicode(file_path, image_to_save):
                    saved_count_this_time += 1
                    next_index = real_index + 1
                    last_capture_time = current_time

                    class_counts = count_all_classes(dataset_dir, class_names)

                    h_saved, w_saved = image_to_save.shape[:2]

                    print(
                        f"已保存：{file_path} | "
                        f"模式：{save_mode} | "
                        f"尺寸：{w_saved}x{h_saved} | "
                        f"清晰度：{sharpness:.1f} | "
                        f"亮度：{brightness:.1f}"
                    )

                    write_log(log_path, {
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "class_name": class_name,
                        "file_path": str(file_path),
                        "sharpness": f"{sharpness:.2f}",
                        "brightness": f"{brightness:.2f}",
                        "capture_type": "auto",
                        "save_mode": save_mode,
                        "roi_x": used_roi.x if used_roi else "",
                        "roi_y": used_roi.y if used_roi else "",
                        "roi_size": used_roi.size if used_roi else "",
                        "saved_width": w_saved,
                        "saved_height": h_saved,
                    })
                else:
                    print("图片保存失败。")

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("\n用户退出采集。")
            break

        elif key == ord(" "):
            if not started:
                if not args.save_full_frame and selector.roi is None:
                    print("\n请先用鼠标左键拖拽选择正方形 ROI，再开始采集。")
                else:
                    started = True
                    paused = False
                    last_capture_time = time.time()
                    print("\n已开始采集。")
            else:
                print("\n当前已经开始采集，按 P 可以暂停或继续。")

        elif key == ord("p"):
            if not started:
                print("\n当前还没有开始采集，请先按空格开始。")
            else:
                paused = not paused

                if paused:
                    print("\n采集已暂停。")
                else:
                    print("\n采集已继续。")
                    last_capture_time = time.time()

        elif key == ord("s"):
            active_roi = selector.get_active_roi()

            image_to_save, save_mode, used_roi = prepare_frame_for_saving(
                frame,
                active_roi,
                args,
            )

            if image_to_save is None:
                print("\n当前没有有效 ROI，无法手动保存。请先用鼠标拖拽选择正方形 ROI。")
            else:
                manual_sharpness = calc_sharpness(image_to_save)
                manual_brightness = calc_brightness(image_to_save)

                file_path, real_index = get_unique_file_path(
                    class_dir,
                    class_name,
                    next_index,
                    IMAGE_EXT,
                )

                if save_image_unicode(file_path, image_to_save):
                    saved_count_this_time += 1
                    next_index = real_index + 1
                    last_capture_time = time.time()

                    class_counts = count_all_classes(dataset_dir, class_names)

                    h_saved, w_saved = image_to_save.shape[:2]

                    print(
                        f"手动保存：{file_path} | "
                        f"模式：{save_mode} | "
                        f"尺寸：{w_saved}x{h_saved} | "
                        f"清晰度：{manual_sharpness:.1f} | "
                        f"亮度：{manual_brightness:.1f}"
                    )

                    write_log(log_path, {
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "class_name": class_name,
                        "file_path": str(file_path),
                        "sharpness": f"{manual_sharpness:.2f}",
                        "brightness": f"{manual_brightness:.2f}",
                        "capture_type": "manual",
                        "save_mode": save_mode,
                        "roi_x": used_roi.x if used_roi else "",
                        "roi_y": used_roi.y if used_roi else "",
                        "roi_size": used_roi.size if used_roi else "",
                        "saved_width": w_saved,
                        "saved_height": h_saved,
                    })
                else:
                    print("手动保存失败。")

        elif key == ord("o"):
            if selector.roi is None:
                print("\n当前没有 ROI，无法保存。请先用鼠标拖拽选择正方形 ROI。")
            else:
                h, w = frame.shape[:2]
                save_roi_config(
                    roi_config_path,
                    selector.roi.clamp(w, h),
                    frame_width=w,
                    frame_height=h,
                )

        elif key == ord("c"):
            selector.clear()
            started = False
            paused = False
            print("\n已清除 ROI。请重新拖拽选择正方形 ROI。采集已停止。")

        elif key == ord("v"):
            show_panel = not show_panel
            print(f"\n文字信息面板：{'显示' if show_panel else '隐藏'}")
            print("无论是否显示面板，保存的图片都不会包含文字面板或 ROI 红框。")

        elif key == ord("+") or key == ord("="):
            sharpness_threshold += 10
            print(f"\n清晰度阈值提高为：{sharpness_threshold}")

        elif key == ord("-") or key == ord("_"):
            sharpness_threshold = max(0, sharpness_threshold - 10)
            print(f"\n清晰度阈值降低为：{sharpness_threshold}")

        if target_count is not None and saved_count_this_time >= target_count:
            print(f"\n已达到目标采集数量：{target_count} 张。")
            break

    cap.release()
    cv2.destroyAllWindows()

    final_counts = count_all_classes(dataset_dir, class_names)

    print("\n========== 采集结束 ==========")
    print(f"采集类别：{class_name}")
    print(f"本次保存数量：{saved_count_this_time}")
    print(f"图片保存目录：{class_dir.resolve()}")
    print(f"采集日志：{log_path.resolve()}")

    if selector.roi is not None:
        print(
            f"当前 ROI：x={selector.roi.x}, "
            f"y={selector.roi.y}, size={selector.roi.size}"
        )
        print(f"ROI 配置文件：{roi_config_path}")
    else:
        print("当前未设置 ROI。")

    print("\n当前数据集数量：")
    for cname in class_names:
        print(f"{cname}：{final_counts.get(cname, 0)} 张")

    print(f"总计：{sum(final_counts.values())} 张")


if __name__ == "__main__":
    main()
