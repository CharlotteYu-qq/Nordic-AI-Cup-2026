
import glob, os, cv2
from ultralytics import YOLO

MODEL_PATH = "/home/azureuser/Nordic-AI-Cup-2026/drone-flyby/best.pt"
SEQ_DIR = "/home/azureuser/Nordic-AI-Cup-2026/drone-flyby/recorded_sequences/76f12f1ef47940c8914ecf97ba09cd55"

model = YOLO(MODEL_PATH)
# 重点检查机库聚集的帧段
target_frames = [f"{i:06d}.png" for i in range(75, 90)] + [f"{i:06d}.png" for i in range(135, 150)]

found_classes = {}
for fname in target_frames:
    p = os.path.join(SEQ_DIR, fname)
    if not os.path.exists(p):
        continue
    img = cv2.imread(p)
    res = model.predict(img, conf=0.03, imgsz=960, verbose=False)[0]
    for b in res.boxes:
        cls_name = model.names[int(b.cls[0])]
        conf = float(b.conf[0])
        if cls_name not in ["hangar", "condor"]:
            found_classes[cls_name] = max(found_classes.get(cls_name, 0.0), conf)
            print(f"[{fname}] 发现隐藏小目标: {cls_name} (conf: {conf:.3f})")

print("\n--- 阵地周围潜在小目标最高置信度 ---")
for k, v in found_classes.items():
    print(f"类别: {k:<16} 最高响应: {v:.3f}")