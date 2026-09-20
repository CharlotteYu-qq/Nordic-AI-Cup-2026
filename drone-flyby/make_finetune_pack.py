# make_finetune_pack.py
import glob
import os
import shutil
import cv2
from ultralytics import YOLO

MODEL_PATH = "/home/azureuser/Nordic-AI-Cup-2026/drone-flyby/best.pt"
SEQ_DIR = "/home/azureuser/Nordic-AI-Cup-2026/drone-flyby/recorded_sequences/76f12f1ef47940c8914ecf97ba09cd55"
OUT_DIR = "datasets/online_adapt"

os.makedirs(f"{OUT_DIR}/images/train", exist_ok=True)
os.makedirs(f"{OUT_DIR}/labels/train", exist_ok=True)
os.makedirs(f"{OUT_DIR}/images/val", exist_ok=True)
os.makedirs(f"{OUT_DIR}/labels/val", exist_ok=True)

model = YOLO(MODEL_PATH)
pngs = sorted(glob.glob(f"{SEQ_DIR}/*.png"))

valid_boxes = 0
for idx, p in enumerate(pngs):
    bname = os.path.splitext(os.path.basename(p))[0]
    split = "val" if idx % 5 == 0 else "train"
    img = cv2.imread(p)

    # 降低置信度阈值，尽可能多捞取在新场景下置信度受压制的目标
    res = model.predict(img, conf=0.08, imgsz=960, verbose=False)[0]
    lines = []
    if len(res.boxes) > 0:
        for b in res.boxes:
            cls_id = int(b.cls[0])
            xywh = b.xywhn[0].cpu().numpy()
            lines.append(f"{cls_id} {xywh[0]:.6f} {xywh[1]:.6f} {xywh[2]:.6f} {xywh[3]:.6f}")
            valid_boxes += 1

    cv2.imwrite(f"{OUT_DIR}/images/{split}/{bname}.png", img)
    with open(f"{OUT_DIR}/labels/{split}/{bname}.txt", "w") as f:
        f.write("\n".join(lines))

print(f"数据处理完毕：247 帧已划分完毕，共收集伪标签框 {valid_boxes} 个（含纯背景负样本）。")

# 写入 data.yaml
yaml_content = f"""path: {os.path.abspath(OUT_DIR)}
train: images/train
val: images/val

names:
"""
for idx, name in model.names.items():
    yaml_content += f"  {idx}: {name}\n"

with open(f"{OUT_DIR}/data.yaml", "w") as f:
    f.write(yaml_content)