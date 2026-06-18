#!/usr/bin/env python3
"""五分类实时摄像头测试：加载 MobileNetV3 模型，实时预测并显示理论分拣动作。不发送串口，不控制 MCU。"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


# =========================================================
# 默认配置
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = PROJECT_ROOT / "models" / "vision_trigger_5class_mobilenetv3" / "latest_mobilenetv3_best.pt"
DEFAULT_CLASS_CONFIG = PROJECT_ROOT / "09_Vision_Trigger_5Class_System" / "config" / "class_mapping_5class.json"
DEFAULT_ROI_CONFIG = PROJECT_ROOT / "09_Vision_Trigger_5Class_System" / "config" / "roi_config.json"

DEFAULT_CONF_THRESHOLD = 0.82
DEFAULT_STABLE_FRAMES = 5

CLASS_TO_ACTION = {"其他": "O", "厨余": "K", "可回收": "R", "有害": "H"}
ACTION_NAME = {"O": "其他", "K": "厨余", "R": "可回收", "H": "有害"}


# =========================================================
# 五分类读取
# =========================================================

def load_class_mapping(checkpoint=None, config_path=None):
    """从 checkpoint 或权威配置文件读取五分类。返回 (class_names, idx_to_class, action_mapping)"""
    # 1) 尝试 checkpoint 中的 class_to_idx
    if checkpoint and "class_to_idx" in checkpoint:
        c2i = checkpoint["class_to_idx"]
        if isinstance(c2i, dict) and len(c2i) == 5:
            sorted_pairs = sorted(c2i.items(), key=lambda kv: kv[1])
            class_names = [name for name, _ in sorted_pairs]
            expected = {"待分拣", "其他", "厨余", "可回收", "有害"}
            if set(class_names) == expected:
                idx_to_class = {v: k for k, v in c2i.items()}
                return class_names, idx_to_class, CLASS_TO_ACTION

    # 2) 尝试权威配置文件
    if config_path is None:
        config_path = DEFAULT_CLASS_CONFIG
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"五分类配置文件不存在：{config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    c2i = data.get("class_to_idx", {})
    sorted_pairs = sorted(c2i.items(), key=lambda kv: kv[1])
    class_names = [name for name, _ in sorted_pairs]

    if len(class_names) != 5:
        raise ValueError(f"类别数应为 5，实际为 {len(class_names)}")

    expected = {"待分拣", "其他", "厨余", "可回收", "有害"}
    if set(class_names) != expected:
        raise ValueError(f"类别不匹配：需要 {expected}，实际 {set(class_names)}")

    idx_to_class = {i: name for i, name in enumerate(class_names)}
    return class_names, idx_to_class, CLASS_TO_ACTION


# =========================================================
# ROI 工具
# =========================================================

def load_roi(roi_config_path):
    """从 JSON 加载 ROI 配置，返回 (x, y, w, h) 或 None"""
    path = Path(roi_config_path)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    roi = data.get("roi", data)
    x, y, w, h = roi.get("x", 0), roi.get("y", 0), roi.get("w", 0), roi.get("h", 0)
    if w > 0 and h > 0:
        return (x, y, w, h)
    return None


def save_roi(roi_config_path, x, y, w, h):
    path = Path(roi_config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"roi": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"ROI 配置已保存：{path.resolve()}")


# =========================================================
# 中文字体
# =========================================================

def get_chinese_font(font_size=26):
    font_candidates = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    ]
    for fp in font_candidates:
        if fp and os.path.exists(fp):
            return ImageFont.truetype(fp, font_size)
    return ImageFont.load_default()


# =========================================================
# 稳定预测器
# =========================================================

class StablePredictor:
    def __init__(self, stable_frames=5, confidence_threshold=0.82):
        self.stable_frames = stable_frames
        self.confidence_threshold = confidence_threshold
        self.history = []  # [(class_id, confidence), ...]

    def update(self, class_id, confidence):
        self.history.append((class_id, confidence))
        if len(self.history) > self.stable_frames:
            self.history.pop(0)

        if len(self.history) < self.stable_frames:
            return {"is_stable": False, "stable_class": None, "stable_confidence": 0.0, "stable_count": len(self.history)}

        classes = [c for c, _ in self.history]
        if len(set(classes)) != 1:
            return {"is_stable": False, "stable_class": None, "stable_confidence": 0.0, "stable_count": len(self.history)}

        avg_conf = sum(cf for _, cf in self.history) / len(self.history)
        if avg_conf >= self.confidence_threshold:
            return {"is_stable": True, "stable_class": classes[0], "stable_confidence": avg_conf, "stable_count": len(self.history)}
        return {"is_stable": False, "stable_class": None, "stable_confidence": avg_conf, "stable_count": len(self.history)}

    def reset(self):
        self.history.clear()

    def set_stable_frames(self, n):
        self.stable_frames = max(1, n)
        self.reset()

    def set_threshold(self, t):
        self.confidence_threshold = max(0.01, min(1.0, t))
        self.reset()


# =========================================================
# 预处理
# =========================================================

def get_transform(image_size=224):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(image_size + 32),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# =========================================================
# 画面绘制 (PIL 中文)
# =========================================================

def draw_overlay(frame_bgr, roi_box, class_names, top_probs, stable_info, fps, device_str, model_name_short):
    """使用 PIL 绘制中文信息面板，直接覆盖在 frame 上。"""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font_main = get_chinese_font(28)
    font_small = get_chinese_font(22)

    # ROI 红框
    if roi_box:
        rx, ry, rw, rh = roi_box
        draw.rectangle([rx, ry, rx + rw, ry + rh], outline=(255, 50, 50, 255), width=2)
        draw.text((rx + 5, ry + 5), "ROI", font=font_small, fill=(255, 50, 50, 255))

    # 半透明背景
    panel_w = 380
    panel_h = 290
    draw.rounded_rectangle([10, 10, 10 + panel_w, 10 + panel_h], radius=14, fill=(0, 0, 0, 180))
    y = 18

    # Top-1
    top1_name = class_names[top_probs[0][0]] if top_probs else "-"
    top1_conf = top_probs[0][1] if top_probs else 0.0
    is_stable = stable_info["is_stable"]
    stable_cls = stable_info["stable_class"]
    stable_conf = stable_info["stable_confidence"]

    # 颜色
    if is_stable and stable_cls == 0:
        color = (100, 255, 100, 255)  # 待分拣 绿色
    elif is_stable and stable_cls is not None:
        color = (255, 200, 80, 255)  # 可发送 橙黄
    elif top1_conf >= 0.82:
        color = (255, 255, 100, 255)
    else:
        color = (180, 180, 180, 255)

    lines = [
        (f"类别: {top1_name}  {top1_conf:.3f}", font_main, color),
        (f"设备: {device_str}  FPS: {fps:.1f}", font_small, (200, 200, 200, 255)),
        ("", font_small, (255, 255, 255, 255)),
    ]

    # Top-2 / Top-3
    if len(top_probs) >= 2:
        lines.append((f"  #2: {class_names[top_probs[1][0]]}  {top_probs[1][1]:.3f}", font_small, (190, 190, 190, 255)))
    if len(top_probs) >= 3:
        lines.append((f"  #3: {class_names[top_probs[2][0]]}  {top_probs[2][1]:.3f}", font_small, (160, 160, 160, 255)))
    lines.append(("", font_small, (255, 255, 255, 255)))

    # 稳定状态
    if is_stable and stable_cls == 0:
        lines.append((f"稳定: 待分拣 → 不触发分拣", font_small, (100, 255, 100, 255)))
    elif is_stable and stable_cls is not None:
        action = CLASS_TO_ACTION.get(class_names[stable_cls], "?")
        lines.append((f"稳定: {class_names[stable_cls]} → 理论动作: {action}", font_main, (255, 200, 80, 255)))
    else:
        sc = stable_info["stable_count"]
        lines.append((f"稳定帧: {sc}/{DEFAULT_STABLE_FRAMES}  未稳定", font_small, (200, 200, 200, 255)))

    lines.append((f"阈值: {DEFAULT_CONF_THRESHOLD:.2f}  [+/-]  帧数: {DEFAULT_STABLE_FRAMES}  [[]/]]", font_small, (180, 180, 180, 255)))
    lines.append((f"模型: {model_name_short}  [Q退出 S快照]", font_small, (160, 160, 160, 255)))

    for text, font, col in lines:
        draw.text((22, y), text, font=font, fill=col)
        y += 34

    image = Image.alpha_composite(image, overlay).convert("RGB")
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


# =========================================================
# 主程序
# =========================================================

def main():
    # 延迟导入重型依赖 — argparse / --help / --dry-run 无需 GUI 和深度学习栈
    global cv2, np, torch, nn, models, transforms, Image, ImageDraw, ImageFont
    try:
        import cv2
        import numpy as np
        import torch
        import torch.nn as nn
        from torchvision import models, transforms
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as e:
        print(f"缺少依赖：{e}")
        print("请安装：pip install opencv-python numpy torch torchvision pillow")
        sys.exit(1)

    global DEFAULT_CONF_THRESHOLD, DEFAULT_STABLE_FRAMES
    parser = argparse.ArgumentParser(description="五分类实时摄像头模型测试")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help=f"模型路径 (默认: {DEFAULT_MODEL})")
    parser.add_argument("--class-config", default=str(DEFAULT_CLASS_CONFIG), help="五分类配置文件路径")
    parser.add_argument("--roi-config", default=str(DEFAULT_ROI_CONFIG), help="ROI 配置文件路径")
    parser.add_argument("--no-roi", action="store_true", help="不使用 ROI，全帧输入模型")
    parser.add_argument("--confidence-threshold", type=float, default=DEFAULT_CONF_THRESHOLD, help="稳定置信度阈值")
    parser.add_argument("--stable-frames", type=int, default=DEFAULT_STABLE_FRAMES, help="稳定所需连续帧数")
    parser.add_argument("--camera-index", type=int, default=1, help="摄像头编号")
    parser.add_argument("--width", type=int, default=1280, help="摄像头宽度")
    parser.add_argument("--height", type=int, default=720, help="摄像头高度")
    parser.add_argument("--device", default="auto", help="auto/cuda/cpu")
    parser.add_argument("--image-size", type=int, default=224, help="模型输入尺寸")
    parser.add_argument("--save-snapshots", action="store_true", help="按键 S 时保存快照")
    parser.add_argument("--snapshot-dir", default="live_test_snapshots", help="快照保存目录")
    parser.add_argument("--dry-run", action="store_true", help="只检查配置和模型，不打开摄像头")
    args = parser.parse_args()

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda" else "cpu"
    device_str = "cuda" if device == "cuda" else "cpu"

    model_path = Path(args.model).expanduser().resolve()
    print(f"[Config] model: {model_path}")
    print(f"[Config] device: {device_str}")
    print(f"[Config] camera: {args.camera_index} ({args.width}x{args.height})")
    print(f"[Config] image_size: {args.image_size}")
    print(f"[Config] confidence_threshold: {args.confidence_threshold}")
    print(f"[Config] stable_frames: {args.stable_frames}")

    # 加载 checkpoint
    if not model_path.exists():
        print(f"错误：模型文件不存在：{model_path}")
        sys.exit(1)

    ckpt = torch.load(model_path, map_location="cpu")
    print(f"[Checkpoint] model_name: {ckpt.get('model_name', 'unknown')}")
    print(f"[Checkpoint] num_classes: {ckpt.get('num_classes', 'N/A')}")
    print(f"[Checkpoint] best_val_acc: {ckpt.get('best_val_acc', 'N/A')}")

    # 加载类别
    try:
        class_names, idx_to_class, action_mapping = load_class_mapping(ckpt, args.class_config)
    except Exception as e:
        print(f"错误：无法加载五分类配置：{e}")
        sys.exit(1)

    num_classes = len(class_names)
    print(f"[Class] order: {class_names}")
    print(f"[Class] num_classes: {num_classes}")
    print(f"[Class] action_mapping: {action_mapping}")
    if num_classes != 5:
        print(f"错误：类别数必须为 5，实际为 {num_classes}")
        sys.exit(1)

    # 模型结构
    model_name = ckpt.get("model_name", "mobilenet_v3_small")
    if "large" in str(model_name):
        model = models.mobilenet_v3_large(weights=None)
    else:
        model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    model.to(device)
    model_name_short = "mobilenet_v3_large" if "large" in str(model_name) else "mobilenet_v3_small"
    print(f"[Model] {model_name_short}, output: {model.classifier[3].out_features} (expected: {num_classes})")
    assert model.classifier[3].out_features == num_classes, "输出层与类别数不匹配！"
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] parameters: {total_params:,}")

    # ROI
    roi_box = None
    if not args.no_roi:
        roi_box = load_roi(args.roi_config)
        if roi_box:
            print(f"[ROI] loaded: x={roi_box[0]} y={roi_box[1]} w={roi_box[2]} h={roi_box[3]}")
        else:
            print("[ROI] 配置文件不存在或无效，预览中可拖拽选择（按 O 保存）")

    if args.dry_run:
        print("\n[Dry-Run] 检查通过：模型、类别、ROI 均正常。不打开摄像头。")
        sys.exit(0)

    # 预处理
    tf = get_transform(args.image_size)
    stable_predictor = StablePredictor(args.stable_frames, args.confidence_threshold)

    # 摄像头
    cap = cv2.VideoCapture(args.camera_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cv2.namedWindow("Vision Trigger 5Class Live Test")
    cv2.setMouseCallback("Vision Trigger 5Class Live Test", _on_mouse)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头：{args.camera_index}")

    snapshot_dir = Path(args.snapshot_dir)
    if args.save_snapshots:
        snapshot_dir.mkdir(parents=True, exist_ok=True)

    # ROI 拖拽状态
    dragging = False
    roi_start = (0, 0)
    editing_roi = False
    show_details = True

    print(f"\n按键: Q退出 S快照 O保存ROI C清除ROI R重新选ROI V显隐详情 +/-调阈值 []调帧数")
    print(f"预览窗口已打开，请先调整 ROI（鼠标拖拽选区 → 按 O 保存）")

    prev_time = time.time()
    fps_show = 0.0

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # ROI crop 或全帧
            if editing_roi:
                roi_box = None  # 正在拖拽选区时清除旧 ROI
            if roi_box:
                rx, ry, rw, rh = roi_box
                roi_frame = frame[ry:ry + rh, rx:rx + rw]
            else:
                roi_frame = frame

            # 推理
            rgb = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB)
            x = tf(rgb).unsqueeze(0).to(device)
            logits = model(x)
            prob = torch.softmax(logits, dim=1)[0]
            topk_vals, topk_ids = torch.topk(prob, min(3, num_classes))

            pred_id = int(topk_ids[0].item())
            pred_conf = float(topk_vals[0].item())
            top_probs = [(int(topk_ids[i].item()), float(topk_vals[i].item())) for i in range(len(topk_ids))]
            stable_info = stable_predictor.update(pred_id, pred_conf)

            # FPS
            now = time.time()
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                fps_show = 0.9 * fps_show + 0.1 * (1.0 / dt) if fps_show > 0 else 1.0 / dt

            # 绘制
            if show_details:
                display = draw_overlay(frame, roi_box, class_names, top_probs, stable_info, fps_show, device_str, model_name_short)
            else:
                display = frame.copy()
                if roi_box:
                    rx, ry, rw, rh = roi_box
                    cv2.rectangle(display, (rx, ry), (rx + rw, ry + rh), (0, 0, 255), 2)
                cv2.putText(display, f"{class_names[pred_id]} {pred_conf:.3f}", (10, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            # 编辑 ROI — 绘制拖拽提示
            if editing_roi and dragging:
                cx, cy = roi_start
                mx, my = _mouse_pos
                cv2.rectangle(display, (cx, cy), (mx, my), (0, 255, 0), 1)
                cv2.putText(display, "松开鼠标确认ROI", (10, display.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Vision Trigger 5Class Live Test", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                snap_dir = snapshot_dir
                snap_dir.mkdir(parents=True, exist_ok=True)
                orig_path = snap_dir / f"frame_{ts}.jpg"
                roi_path = snap_dir / f"roi_{ts}.jpg"
                cv2.imwrite(str(orig_path), frame)
                cv2.imwrite(str(roi_path), roi_frame)
                print(f"快照已保存：{orig_path}, {roi_path}")
            elif key == ord("o"):
                if roi_box:
                    save_roi(args.roi_config, *roi_box)
                else:
                    print("当前无 ROI，请先拖拽选区")
            elif key == ord("c"):
                roi_box = None
                editing_roi = False
                print("ROI 已清除")
            elif key == ord("r"):
                editing_roi = True
                roi_box = None
                print("请在预览窗口拖拽鼠标选择 ROI 区域")
            elif key == ord("v"):
                show_details = not show_details
            elif key == ord("+") or key == ord("="):
                DEFAULT_CONF_THRESHOLD = min(1.0, DEFAULT_CONF_THRESHOLD + 0.05)
                stable_predictor.set_threshold(DEFAULT_CONF_THRESHOLD)
                print(f"置信度阈值: {DEFAULT_CONF_THRESHOLD:.2f}")
            elif key == ord("-"):
                DEFAULT_CONF_THRESHOLD = max(0.1, DEFAULT_CONF_THRESHOLD - 0.05)
                stable_predictor.set_threshold(DEFAULT_CONF_THRESHOLD)
                print(f"置信度阈值: {DEFAULT_CONF_THRESHOLD:.2f}")
            elif key == ord("["):
                DEFAULT_STABLE_FRAMES = max(1, DEFAULT_STABLE_FRAMES - 1)
                stable_predictor.set_stable_frames(DEFAULT_STABLE_FRAMES)
                print(f"稳定帧数: {DEFAULT_STABLE_FRAMES}")
            elif key == ord("]"):
                DEFAULT_STABLE_FRAMES = min(30, DEFAULT_STABLE_FRAMES + 1)
                stable_predictor.set_stable_frames(DEFAULT_STABLE_FRAMES)
                print(f"稳定帧数: {DEFAULT_STABLE_FRAMES}")

    cap.release()
    cv2.destroyAllWindows()
    print("\n退出")


# =========================================================
# 鼠标回调（ROI 选择）
# =========================================================

_mouse_pos = (0, 0)
_roi_start = (0, 0)
_dragging = False


def _on_mouse(event, x, y, flags, param):
    global _mouse_pos, _roi_start, _dragging
    _mouse_pos = (x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        _roi_start = (x, y)
        _dragging = True
    elif event == cv2.EVENT_LBUTTONUP:
        _dragging = False
        x1, y1 = _roi_start
        x2, y2 = x, y
        rx, ry = min(x1, x2), min(y1, y2)
        rw, rh = abs(x2 - x1), abs(y2 - y1)
        if rw > 20 and rh > 20:
            print(f"ROI 选区: x={rx} y={ry} w={rw} h={rh} (按 O 保存，按 C 清除)")


if __name__ == "__main__":
    main()
