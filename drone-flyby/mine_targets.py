# mine_targets.py
import glob, os, cv2
from ultralytics import YOLO

MODEL_PATH = "/home/azureuser/Nordic-AI-Cup-2026/drone-flyby/best.pt"
SEQ_DIR = "/home/azureuser/Nordic-AI-Cup-2026/drone-flyby/recorded_sequences/76f12f1ef47940c8914ecf97ba09cd55"
OUT_DIR = "mined_crops"
os.makedirs(OUT_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)
pngs = sorted(glob.glob(f"{SEQ_DIR}/*.png"))

candidates = []
for p in pngs:
    img = cv2.imread(p)
    bname = os.path.basename(p)
    res = model.predict(img, conf=0.05, imgsz=960, verbose=False)[0]
    if len(res.boxes) > 0:
        for b in res.boxes:
            conf = float(b.conf[0])
            cls_name = model.names[int(b.cls[0])]
            xyxy = b.xyxy[0].cpu().numpy()
            candidates.append((conf, cls_name, bname, xyxy, img))

candidates.sort(key=lambda x: x[0], reverse=True)

print(f"--- 线上检测最高置信度的 Top 15 目标 ---")
for idx, (conf, cls_name, bname, xyxy, img) in enumerate(candidates[:15]):
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    # 抠出图并保存
    pad = 20
    h, w = img.shape[:2]
    crop = img[max(0, y1-pad):min(h, y2+pad), max(0, x1-pad):min(w, x2+pad)]
    crop_name = f"{OUT_DIR}/rank{idx+1:02d}_{cls_name}_{conf:.2f}_{bname}"
    if crop.size > 0:
        cv2.imwrite(crop_name, crop)
    print(f"Top {idx+1:02d}: [{cls_name}] 置信度 {conf:.3f} | 位于帧: {bname} | 保存至 {crop_name}")