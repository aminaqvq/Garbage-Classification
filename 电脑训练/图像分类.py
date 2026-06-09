import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import time
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

def main():
    # ========= 参数 =========
    ckpt_path = r"C:\Software\Garbage Classification\mobilenet_cls_best.pt"
    cam_id = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    conf_thres = 0.80  # 只显示 >= 0.8（否则显示 Uncertain）
    # =======================

    # 1) 加载 checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu")
    class_to_idx = ckpt.get("class_to_idx", None)
    if class_to_idx is None:
        raise RuntimeError("checkpoint 里找不到 class_to_idx。请确认你保存时包含了它。")

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(idx_to_class)

    # 2) 构建 MobileNet 并加载权重（要和你训练时一致：mobilenet_v3_small）
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    model.to(device)

    # 3) 预处理：OpenCV(BGR) -> RGB -> 224 输入
    tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # 你训练时没做 Normalize，这里也别加（保持一致）
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

            # BGR -> RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            x = tf(rgb).unsqueeze(0).to(device)  # [1,3,224,224]

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

            # FPS
            now = time.time()
            dt = now - prev
            prev = now
            if dt > 0:
                fps_show = 0.9 * fps_show + 0.1 * (1.0 / dt)

            # 画字
            annotated = frame.copy()
            cv2.putText(annotated, f"CLS: {text}", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(annotated, f"FPS: {fps_show:.1f}", (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            # 控制台仅在结果变化时打印
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