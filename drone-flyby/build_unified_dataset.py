# build_unified_dataset.py
import json, glob, os, cv2, shutil

OUTPUT_DIR = "datasets/unified_dataset"
os.makedirs(f"{OUTPUT_DIR}/images/train", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/labels/train", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/images/val", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/labels/val", exist_ok=True)

OBJECT_CLASSES = [
    'hangar', 'helicopter', 'jet_plane', 'large_launcher',
    'large_tower', 'medium_launcher', 'medium_plane', 'mine_roller',
    'small_launcher', 'small_plane', 'small_tower', 'ta-ta',
    'tank', 'condor', 'jammer', 'spacecraft'
]
cls_to_idx = {name: i for i, name in enumerate(OBJECT_CLASSES)}

SRC_DIR = "/home/azureuser/Nordic-AI-Cup-2026/drone-flyby/src/helsinki"
img_paths = sorted(glob.glob(f"{SRC_DIR}/images/*.png"))

# 1. 注入 25 帧全局下采样图（学会 Level 0 视角，尺寸为 960x540）
print("1. 生成全局下采样样本 (对齐线上 Level 0)...")
for p in img_paths:
    bname = os.path.splitext(os.path.basename(p))[0]
    frame_num = int(bname.split('_')[-1])
    json_path = f"{SRC_DIR}/annotations/{bname}.json"
    if not os.path.exists(json_path):
        continue
    
    with open(json_path, 'r') as f:
        annos = json.load(f).get('annotations', [])
    
    img = cv2.imread(p)
    H, W = img.shape[:2]
    split = "val" if frame_num % 5 == 0 else "train"
    
    # 缩放到 960x540
    resized_img = cv2.resize(img, (960, 540), interpolation=cv2.INTER_AREA)
    cv2.imwrite(f"{OUTPUT_DIR}/images/{split}/global_{bname}.png", resized_img)
    
    lines = []
    for a in annos:
        obj_id = a['object_id']
        if obj_id not in cls_to_idx:
            continue
        bx1, by1, bx2, by2 = a['bbox']
        # 归一化 xywh
        cx = (bx1 + bx2) / 2.0 / W
        cy = (by1 + by2) / 2.0 / H
        w = (bx2 - bx1) / W
        h = (by2 - by1) / H
        lines.append(f"{cls_to_idx[obj_id]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    
    with open(f"{OUTPUT_DIR}/labels/{split}/global_{bname}.txt", "w") as f:
        f.write("\n".join(lines))

# 2. 注入 395 张局部切片（学会高分辨率特写）
print("2. 导入刚才生成的局部切片样本...")
for split in ["train", "val"]:
    for f in glob.glob(f"datasets/sahi_joint/images/{split}/*.*"):
        shutil.copy(f, f"{OUTPUT_DIR}/images/{split}/")
    for f in glob.glob(f"datasets/sahi_joint/labels/{split}/*.*"):
        shutil.copy(f, f"{OUTPUT_DIR}/labels/{split}/")

# 3. 写入统一 data.yaml
yaml_content = f"""path: {os.path.abspath(OUTPUT_DIR)}
train: images/train
val: images/val

names:
"""
for idx, name in enumerate(OBJECT_CLASSES):
    yaml_content += f"  {idx}: {name}\n"

with open(f"{OUTPUT_DIR}/data.yaml", "w") as f:
    f.write(yaml_content)

print(f"数据构建完成！总训练图数量: {len(glob.glob(f'{OUTPUT_DIR}/images/train/*.*'))}")