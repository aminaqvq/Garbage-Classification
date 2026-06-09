import cv2
import time
import csv
import re
import os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
from datetime import datetime

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# =========================================================
# 基础配置区：你主要改这里
# =========================================================

# 固定四大类，不再让用户自定义
CLASS_NAMES = ["可回收", "有害", "厨余", "其他"]

# 数据集根目录
DATASET_ROOT = str(PROJECT_ROOT / "dataset")

# 摄像头编号
# 笔记本内置摄像头一般是 0
# 外接摄像头可能是 1、2
CAMERA_INDEX = 1

# 自动采集时间间隔，单位：秒
CAPTURE_INTERVAL = 1.5

# 默认清晰度阈值
# 如果一直不保存，说明阈值太高，可以调低，比如 60
# 如果保存了很多模糊图，说明阈值太低，可以调高，比如 150
DEFAULT_SHARPNESS_THRESHOLD = 100

# 亮度范围
# 太暗或者太亮都会影响训练质量
MIN_BRIGHTNESS = 40
MAX_BRIGHTNESS = 220

# 图片保存格式
IMAGE_EXT = ".jpg"

# JPG 保存质量
JPEG_QUALITY = 95

# 摄像头分辨率
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# 中文字体路径
# 如果预览窗口中文显示为方块，可以手动改成你电脑里的字体路径
# Windows 常见：
# C:/Windows/Fonts/msyh.ttc
# C:/Windows/Fonts/simhei.ttf
CHINESE_FONT_PATH = None


# =========================================================
# 工具函数
# =========================================================

def create_dataset_folders(root_dir: Path):
    """
    创建数据集目录：
    dataset/
        可回收/
        有害/
        厨余/
        其他/
    """
    root_dir.mkdir(parents=True, exist_ok=True)

    for class_name in CLASS_NAMES:
        class_dir = root_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)


def safe_filename(name: str) -> str:
    """
    清理文件名非法字符。
    中文会保留。
    """
    return re.sub(r'[\\/:*?"<>|]', "_", name)


def count_images_in_class(class_dir: Path) -> int:
    """
    统计某个类别文件夹中的图片数量。
    """
    image_suffixes = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

    count = 0
    for file in class_dir.iterdir():
        if file.is_file() and file.suffix.lower() in image_suffixes:
            count += 1

    return count


def count_all_classes(root_dir: Path) -> dict:
    """
    统计四大类每类当前图片数量。
    """
    result = {}

    for class_name in CLASS_NAMES:
        class_dir = root_dir / class_name
        result[class_name] = count_images_in_class(class_dir)

    return result


def get_next_index_by_max_number(class_dir: Path, class_name: str, ext: str) -> int:
    """
    根据已有文件名中的最大编号决定下一张编号。

    例如已有：
    可回收_0001.jpg
    可回收_0002.jpg
    可回收_0031.jpg
    可回收_0035.jpg

    那么下一张从 36 开始，而不是按图片数量 + 1。
    """
    safe_name = safe_filename(class_name)

    pattern = re.compile(
        rf"^{re.escape(safe_name)}_(\d+){re.escape(ext)}$",
        re.IGNORECASE
    )

    max_index = 0

    for file in class_dir.glob(f"*{ext}"):
        match = pattern.match(file.name)
        if match:
            number = int(match.group(1))
            max_index = max(max_index, number)

    return max_index + 1


def get_unique_file_path(
    class_dir: Path,
    class_name: str,
    start_index: int,
    ext: str
):
    """
    获取不会重复的文件路径。

    这是第二层保险：
    即使 next_index 因为某些原因不准，
    保存前也会检查文件是否存在。
    如果存在，就继续 +1，直到找到不存在的文件名。
    """
    safe_name = safe_filename(class_name)
    index = start_index

    while True:
        file_name = f"{safe_name}_{index:04d}{ext}"
        file_path = class_dir / file_name

        if not file_path.exists():
            return file_path, index

        index += 1


def calc_sharpness(frame) -> float:
    """
    计算图像清晰度。

    使用拉普拉斯方差：
    数值越大，通常说明图像越清晰。
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    return float(sharpness)


def calc_brightness(frame) -> float:
    """
    计算图像平均亮度。
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    return float(brightness)


def save_image_unicode(path: Path, frame) -> bool:
    """
    兼容中文路径保存图片。

    cv2.imwrite 在部分 Windows 中文路径下可能失败，
    所以这里用 imencode + tofile。
    """
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]

    success, encoded_img = cv2.imencode(
        path.suffix,
        frame,
        encode_params
    )

    if success:
        encoded_img.tofile(str(path))
        return True

    return False


def write_log(log_path: Path, row: dict):
    """
    写入采集日志。
    """
    file_exists = log_path.exists()

    with open(log_path, "a", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "time",
            "class_name",
            "file_path",
            "sharpness",
            "brightness",
            "capture_type"
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def select_class():
    """
    终端选择当前采集类别。
    """
    print("\n请选择当前要采集的垃圾类别：")

    for index, class_name in enumerate(CLASS_NAMES, start=1):
        print(f"{index}. {class_name}")

    while True:
        user_input = input("\n请输入编号或类别名：").strip()

        if user_input.isdigit():
            index = int(user_input)
            if 1 <= index <= len(CLASS_NAMES):
                return CLASS_NAMES[index - 1]

        if user_input in CLASS_NAMES:
            return user_input

        print("输入无效，请重新输入。")


def get_chinese_font(font_size=24):
    """
    获取中文字体。
    尽量自动适配 Windows / macOS / Linux。
    """
    font_candidates = []

    if CHINESE_FONT_PATH:
        font_candidates.append(CHINESE_FONT_PATH)

    font_candidates.extend([
        # Windows
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",

        # macOS
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/Library/Fonts/Arial Unicode.ttf",

        # Linux
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    ])

    for font_path in font_candidates:
        if font_path and os.path.exists(font_path):
            return ImageFont.truetype(font_path, font_size)

    return ImageFont.load_default()


def get_status_text(started: bool, paused: bool) -> str:
    """
    获取当前运行状态文字。
    """
    if not started:
        return "预览中，未开始采集"

    if paused:
        return "已暂停"

    return "正在采集"


def get_brightness_status(brightness: float) -> str:
    """
    获取亮度状态文字。
    """
    if brightness < MIN_BRIGHTNESS:
        return "过暗"
    elif brightness > MAX_BRIGHTNESS:
        return "过亮"
    else:
        return "正常"


def get_quality_status(
    sharpness: float,
    sharpness_threshold: float,
    brightness: float
) -> str:
    """
    判断当前画面是否满足自动保存条件。
    """
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
    next_file_name: str
):
    """
    在预览画面中绘制信息面板。

    使用 PIL 绘制中文，避免 OpenCV 默认中文乱码。
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
    panel_w = 520
    panel_h = 500

    # 半透明背景
    draw.rounded_rectangle(
        [panel_x, panel_y, panel_x + panel_w, panel_y + panel_h],
        radius=18,
        fill=(0, 0, 0, 165)
    )

    x = panel_x + 22
    y = panel_y + 18

    status_text = get_status_text(started, paused)
    brightness_status = get_brightness_status(brightness)
    quality_status = get_quality_status(
        sharpness,
        sharpness_threshold,
        brightness
    )

    if target_count is None:
        target_text = "不限制"
    else:
        target_text = str(target_count)

    total_count = sum(class_counts.values())

    lines = [
        ("垃圾分类图像采集", font_title),
        (f"当前类别：{class_name}", font_text),
        (f"运行状态：{status_text}", font_text),
        (f"本次已保存：{saved_count_this_time} / {target_text}", font_text),
        (f"拍照间隔：{capture_interval:.2f} 秒", font_text),
        (f"清晰度：{sharpness:.1f} / 阈值 {sharpness_threshold:.1f}", font_text),
        (f"亮度：{brightness:.1f} / {brightness_status}", font_text),
        (f"画面判断：{quality_status}", font_text),
        (f"下一张文件：{next_file_name}", font_small),
        (f"数据集总数：{total_count} 张", font_text),
        ("四类图片数量：", font_text),
    ]

    for text, font in lines:
        draw.text((x, y), text, font=font, fill=(255, 255, 255, 255))
        y += 35

    for cname in CLASS_NAMES:
        count = class_counts.get(cname, 0)
        mark = " ← 当前" if cname == class_name else ""
        draw.text(
            (x + 20, y),
            f"{cname}：{count} 张{mark}",
            font=font_text,
            fill=(255, 255, 255, 255)
        )
        y += 32

    # 底部操作提示
    hint_panel_h = 70
    hint_y1 = image.size[1] - hint_panel_h - 15
    hint_y2 = image.size[1] - 15

    draw.rounded_rectangle(
        [15, hint_y1, image.size[0] - 15, hint_y2],
        radius=16,
        fill=(0, 0, 0, 165)
    )

    hint_text = "操作：空格 开始采集 | P 暂停/继续 | S 手动保存 | + 提高清晰度阈值 | - 降低清晰度阈值 | Q 退出"
    draw.text(
        (35, hint_y1 + 22),
        hint_text,
        font=font_small,
        fill=(255, 255, 255, 255)
    )

    image = Image.alpha_composite(image, overlay).convert("RGB")
    frame_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    return frame_bgr


# =========================================================
# 主程序
# =========================================================

def main():
    print("========== 垃圾分类图像采集程序 ==========")

    root_dir = Path(DATASET_ROOT)
    create_dataset_folders(root_dir)

    print(f"\n数据集目录已创建：{root_dir.resolve()}")
    print("固定四大类目录：")

    for class_name in CLASS_NAMES:
        print(f"- {root_dir / class_name}")

    class_name = select_class()
    class_dir = root_dir / class_name

    target_input = input("\n请输入本次计划采集张数，直接回车表示不限制：").strip()
    target_count = int(target_input) if target_input.isdigit() else None

    threshold_input = input(
        f"\n请输入清晰度阈值，直接回车使用默认值 {DEFAULT_SHARPNESS_THRESHOLD}："
    ).strip()

    if threshold_input:
        sharpness_threshold = float(threshold_input)
    else:
        sharpness_threshold = DEFAULT_SHARPNESS_THRESHOLD

    interval_input = input(
        f"\n请输入拍照间隔秒数，直接回车使用默认值 {CAPTURE_INTERVAL}："
    ).strip()

    if interval_input:
        capture_interval = float(interval_input)
    else:
        capture_interval = CAPTURE_INTERVAL

    # 关键：不是按图片数量 + 1，而是扫描当前最大编号
    next_index = get_next_index_by_max_number(
        class_dir,
        class_name,
        IMAGE_EXT
    )

    saved_count_this_time = 0
    log_path = root_dir / "capture_log.csv"

    print("\n正在打开摄像头...")
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("摄像头打开失败。")
        print("请检查：")
        print("1. 摄像头是否被其他软件占用")
        print("2. CAMERA_INDEX 是否正确，可以尝试改成 1 或 2")
        print("3. PyCharm 是否有摄像头权限")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    print("\n摄像头已打开。")
    print("当前是预览模式，不会自动保存图片。")
    print("请先调整垃圾位置，然后在摄像头窗口中按空格开始采集。")
    print("\n按键说明：")
    print("空格：开始采集")
    print("p：暂停 / 继续")
    print("s：手动保存一张")
    print("q：退出")
    print("+：提高清晰度阈值")
    print("-：降低清晰度阈值")
    print("\n注意：按键时请先点击一下摄像头预览窗口，让窗口获得焦点。")

    started = False
    paused = False

    # 按空格开始后，会从当前时间开始等待 capture_interval 秒再保存第一张
    last_capture_time = time.time()

    class_counts = count_all_classes(root_dir)
    last_count_update_time = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            print("读取摄像头画面失败。")
            break

        current_time = time.time()

        sharpness = calc_sharpness(frame)
        brightness = calc_brightness(frame)

        # 每秒刷新一次各类别图片数量
        if current_time - last_count_update_time >= 1:
            class_counts = count_all_classes(root_dir)
            last_count_update_time = current_time

        # 生成下一张文件名预览
        next_file_path, preview_next_index = get_unique_file_path(
            class_dir,
            class_name,
            next_index,
            IMAGE_EXT
        )
        next_file_name = next_file_path.name

        display_frame = draw_info_panel(
            frame=frame,
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
            next_file_name=next_file_name
        )

        cv2.imshow("Garbage Image Capture", display_frame)

        can_auto_capture = (
            started
            and not paused
            and current_time - last_capture_time >= capture_interval
            and sharpness >= sharpness_threshold
            and MIN_BRIGHTNESS <= brightness <= MAX_BRIGHTNESS
        )

        if can_auto_capture:
            file_path, real_index = get_unique_file_path(
                class_dir,
                class_name,
                next_index,
                IMAGE_EXT
            )

            if save_image_unicode(file_path, frame):
                saved_count_this_time += 1
                next_index = real_index + 1
                last_capture_time = current_time

                class_counts = count_all_classes(root_dir)

                print(
                    f"已保存：{file_path} | "
                    f"清晰度：{sharpness:.1f} | "
                    f"亮度：{brightness:.1f}"
                )

                write_log(log_path, {
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "class_name": class_name,
                    "file_path": str(file_path),
                    "sharpness": f"{sharpness:.2f}",
                    "brightness": f"{brightness:.2f}",
                    "capture_type": "auto"
                })
            else:
                print("图片保存失败。")

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("\n用户退出采集。")
            break

        elif key == ord(" "):
            if not started:
                started = True
                paused = False
                last_capture_time = time.time()
                print("\n已开始采集。")
            else:
                print("\n当前已经开始采集，按 p 可以暂停或继续。")

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
            file_path, real_index = get_unique_file_path(
                class_dir,
                class_name,
                next_index,
                IMAGE_EXT
            )

            if save_image_unicode(file_path, frame):
                saved_count_this_time += 1
                next_index = real_index + 1
                last_capture_time = time.time()

                class_counts = count_all_classes(root_dir)

                print(
                    f"手动保存：{file_path} | "
                    f"清晰度：{sharpness:.1f} | "
                    f"亮度：{brightness:.1f}"
                )

                write_log(log_path, {
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "class_name": class_name,
                    "file_path": str(file_path),
                    "sharpness": f"{sharpness:.2f}",
                    "brightness": f"{brightness:.2f}",
                    "capture_type": "manual"
                })
            else:
                print("手动保存失败。")

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

    final_counts = count_all_classes(root_dir)

    print("\n========== 采集结束 ==========")
    print(f"采集类别：{class_name}")
    print(f"本次保存数量：{saved_count_this_time}")
    print(f"图片保存目录：{class_dir.resolve()}")
    print(f"采集日志：{log_path.resolve()}")

    print("\n当前数据集数量：")
    for cname in CLASS_NAMES:
        print(f"{cname}：{final_counts.get(cname, 0)} 张")

    print(f"总计：{sum(final_counts.values())} 张")


if __name__ == "__main__":
    main()