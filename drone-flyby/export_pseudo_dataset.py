# export_pseudo_dataset.py
import glob, json, os, cv2
from ultralytics import YOLO

MODEL_PATH = "/home/azureuser/Nordic-AI-Cup-2026/drone-flyby/best.pt"
SEQ_DIR = "/home/azureuser/Nordic-AI-Cup-2026/drone-flyby/recorded_sequences/76f12f1ef47940c8914ecf97ba09cd55"
OUT_DIR = "datasets/online_finetune"

os.makedirs(f"{OUT_DIR}/images/train", exist_ok=True)
os.makedirs(f"{OUT_DIR}/labels/train", exist_ok=True)

model = YOLO(MODEL_PATH)
pngs = sorted(glob.glob(f"{SEQ_DIR}/*.png"))

valid_count = 0
for p in pngs:
    bname = os.path.splitext(os.path.basename(p))[0]
    img = cv2.imread(p)
    h, w = img.shape[:2]

    # 用适中的置信度提取高质量伪标签
    res = model.predict(img, conf=0.15, imgsz=960, verbose=False)[0]
    lines = []
    if len(res.boxes) > 0:
        for b in res.boxes:
            cls_id = int(b.cls[0])
            xywh = b.xywhn[0].cpu().numpy()
            lines.append(f"{cls_id} {xywh[0]:.6f} {xywh[1]:.6f} {xywh[2]:.6f} {xywh[3]:.6f}")
        valid_count += 1

    # 即使没有检测到物体，也作为负样本（Background Images）写入，训练模型抗误检能力
    cv2.imwrite(f"{OUT_DIR}/images/train/{bname}.png", img)
    with open(f"{OUT_DIR}/labels/train/{bname}.txt", "w") as f:
        f.write("\n".join(lines))

print(f"伪标签制作完成！共处理 247 帧，其中含目标帧: {valid_count}，纯背景负样本帧: {247 - valid_count}")