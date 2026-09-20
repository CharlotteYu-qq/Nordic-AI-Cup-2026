# test_v8s_sahi.py
import cv2, glob
from ultralytics import YOLO

# 请确保指向你的 best-3.pt
MODEL_PATH = "/home/azureuser/Nordic-AI-Cup-2026/drone-flyby/best-3.pt"
SEQ_DIR = "/home/azureuser/Nordic-AI-Cup-2026/drone-flyby/recorded_sequences/76f12f1ef47940c8914ecf97ba09cd55"

model = YOLO(MODEL_PATH)
pngs = sorted(glob.glob(f"{SEQ_DIR}/*.png"))[:10]

total_boxes = 0
for p in pngs:
    img = cv2.imread(p)
    res = model.predict(img, conf=0.05, imgsz=960, verbose=False)[0]
    total_boxes += len(res.boxes)

print(f"前 10 帧检出目标总数: {total_boxes}")