
import glob, json, os, shutil, cv2
import numpy as np
from ultralytics import YOLO

OUT_DIR = "datasets/joint_dataset"
os.makedirs(f"{OUT_DIR}/images/train", exist_ok=True)
os.makedirs(f"{OUT_DIR}/labels/train", exist_ok=True)
os.makedirs(f"{OUT_DIR}/images/val", exist_ok=True)
os.makedirs(f"{OUT_DIR}/labels/val", exist_ok=True)

# 1. 复制原有的 helsinki_v2 数据（保证基础类别识别）
if os.path.exists("datasets/helsinki_v2"):
    print("正在合并 Helsinki 基础数据集...")
    for split in ["train", "val"]:
        for f in glob.glob(f"datasets/helsinki_v2/images/{split}/*.*"):
            shutil.copy(f, f"{OUT_DIR}/images/{split}/")
        for f in glob.glob(f"datasets/helsinki_v2/labels/{split}/*.*"):
            shutil.copy(f, f"{OUT_DIR}/labels/{split}/")

# 2. 对 247 帧线上数据提取低误检目标与纯背景负样本
SEQ_DIR = "/home/azureuser/Nordic-AI-Cup-2026/drone-flyby/recorded_sequences/76f12f1ef47940c8914ecf97ba09cd55"
model = YOLO("/home/azureuser/Nordic-AI-Cup-2026/drone-flyby/best.pt")
pngs = sorted(glob.glob(f"{SEQ_DIR}/*.png"))

print(f"正在处理 247 帧线上真实数据...")
for idx, p in enumerate(pngs):
    bname = os.path.splitext(os.path.basename(p))[0]
    split = "val" if idx % 6 == 0 else "train"
    img = cv2.imread(p)
    
    # 用 0.12 阈值滤除低置信度噪点
    res = model.predict(img, conf=0.12, imgsz=960, verbose=False)[0]
    lines = []
    if len(res.boxes) > 0:
        for b in res.boxes:
            cls_id = int(b.cls[0])
            xywh = b.xywhn[0].cpu().numpy()
            lines.append(f"{cls_id} {xywh[0]:.6f} {xywh[1]:.6f} {xywh[2]:.6f} {xywh[3]:.6f}")
            
    cv2.imwrite(f"{OUT_DIR}/images/{split}/online_{bname}.png", img)
    with open(f"{OUT_DIR}/labels/{split}/online_{bname}.txt", "w") as f:
        f.write("\n".join(lines))

# 3. 生成统一 data.yaml
yaml_content = f"""path: {os.path.abspath(OUT_DIR)}
train: images/train
val: images/val

names:
"""
for idx, name in model.names.items():
    yaml_content += f"  {idx}: {name}\n"

with open(f"{OUT_DIR}/data.yaml", "w") as f:
    f.write(yaml_content)

print("联合训练集打包完成！")