# inspect_val.py
import glob
import os
import cv2
from ultralytics import YOLO

MODEL_PATH = "/home/azureuser/Nordic-AI-Cup-2026/drone-flyby/best.pt"
SEQ_DIR = "/home/azureuser/Nordic-AI-Cup-2026/drone-flyby/recorded_sequences/76f12f1ef47940c8914ecf97ba09cd55"
OUT_DIR = "inspected_online_frames"
os.makedirs(OUT_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)
pngs = sorted(glob.glob(f"{SEQ_DIR}/*.png"))
print(f"开始全量扫描线上录制帧，共 {len(pngs)} 帧...")

detected_summary = {}

for p in pngs:
    img = cv2.imread(p)
    # 使用 960 分辨率，conf 设为 0.08
    res = model.predict(img, conf=0.08, imgsz=960, verbose=False)[0]
    boxes = res.boxes
    base_name = os.path.basename(p)
    if len(boxes) > 0:
        for b in boxes:
            cls_name = model.names[int(b.cls[0])]
            conf = float(b.conf[0])
            detected_summary[cls_name] = detected_summary.get(cls_name, 0) + 1
        
        # 只保存前 10 张有检测结果的图供人工检查
        if len(glob.glob(f"{OUT_DIR}/*.png")) < 10:
            plotted = res.plot()
            cv2.imwrite(f"{OUT_DIR}/pred_{base_name}", plotted)

print("\n=== 全量 247 帧检测统计 ===")
for k, v in detected_summary.items():
    print(f"  {k}: 共出现 {v} 次")
print(f"总计检出框数: {sum(detected_summary.values())}")