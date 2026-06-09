#!/usr/bin/env python3
"""实时摄像头分类演示：加载训练好的 MobileNetV3 模型，实时预测垃圾分类。"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms


def main():
    parser = argparse.ArgumentParser(description="实时摄像头垃圾分类演示")
    parser.add_argument("--ckpt", default="outputs/latest_mobilenetv3_best.pt",
                        help="训练好的 .pt 模型路径 (默认: outputs/latest_mobilenetv3_best.pt)")
    parser.add_argument("--cam", type=int, default=1,
                        help="摄像头编号 (默认: 1)")
    parser.add_argument("--conf", type=float, default=0.80,
                        help="置信度阈值，低于此值显示 Uncertain (默认: 0.80)")
    parser.add_argument("--device", default="auto",
                        help="设备: auto / cuda / cpu (默认: auto)")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    cam_id = args.cam
    conf_thres = args.conf
    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda" else "cpu"

    print(f"checkpoint: {ckpt_path.resolve()}")
    print(f"device: {device}")
    print(f"camera: {cam_id}")
    print(f"confidence threshold: {conf_thres}")

    # 1) 加载 checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu")
    class_to_idx = ckpt.get("class_to_idx", None)
    if class_to_idx is None:
        raise RuntimeError("checkpoint 里找不到 class_to_idx。请确认你保存时包含了它。")

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(idx_to_class)
    print(f"类别映射: {idx_to_class}")

    # 2) 构建 MobileNet 并加载权重
    model_name = ckpt.get("model_name", "mobilenet_v3_small")
    if "large" in str(model_name):
        model = models.mobilenet_v3_large(weights=None)
    else:
        model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    model.to(device)

    # 3) 预处理
    tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    # 4) 摄像头
    cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头：{cam_id}")

    print("按 q 退出")

    prev = time.time()
    fps_show = 0.0
    last_print = ""

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            x = tf(rgb).unsqueeze(0).to(device)

            logits = model(x)
            prob = torch.softmax(logits, dim=1)[0]
            conf, pred = torch.max(prob, dim=0)

            conf_val = float(conf.item())
            pred_id = int(pred.item())
            pred_name = idx_to_class.get(pred_id, str(pred_id))

            if conf_val >= conf_thres:
                text = f"{pred_name}  {conf_val:.2f}"
            else:
                text = f"Uncertain ({pred_name} {conf_val:.2f})"

            now = time.time()
            dt = now - prev
            prev = now
            if dt > 0:
                fps_show = 0.9 * fps_show + 0.1 * (1.0 / dt)

            annotated = frame.copy()
            cv2.putText(annotated, f"CLS: {text}", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(annotated, f"FPS: {fps_show:.1f}", (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            if text != last_print:
                print(f"\rDetected: {text}        ", end="", flush=True)
                last_print = text

            cv2.imshow("MobileNet Classification Inference", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("\n退出")


if __name__ == "__main__":
    main()
