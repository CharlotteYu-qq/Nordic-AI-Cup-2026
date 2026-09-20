# build_sahi_dataset.py
import json, glob, os, cv2, shutil
import numpy as np

OUTPUT_DIR = "datasets/sahi_joint"
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

# 1. 切割 Helsinki 4K 原图为 960x540 局部切片（提取 16 类高质量小目标）
SRC_DIR = "/home/azureuser/Nordic-AI-Cup-2026/drone-flyby/src/helsinki"
img_paths = sorted(glob.glob(f"{SRC_DIR}/images/*.png"))
print(f"开始切片处理 Helsinki 原始数据集 (共 {len(img_paths)} 帧 4K 图)...")

crop_w, crop_h = 960, 540
step_x, step_y = 720, 400  # 带重叠切片

patch_count = 0
box_count = 0

for idx, p in enumerate(img_paths):
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

    # 滑动窗口
    for y in range(0, H - crop_h + 1, step_y):
        for x in range(0, W - crop_w + 1, step_x):
            x2_crop, y2_crop = x + crop_w, y + crop_h
            
            # 搜集掉入当前切片窗口内的所有真值框
            patch_labels = []
            for a in annos:
                bx1, by1, bx2, by2 = a['bbox']
                obj_id = a['object_id']
                if obj_id not in cls_to_idx:
                    continue
                
                # 计算与切片的相交区域
                ix1 = max(x, bx1)
                iy1 = max(y, by1)
                ix2 = min(x2_crop, bx2)
                iy2 = min(y2_crop, by2)

                if ix2 > ix1 and iy2 > iy1:
                    # 只有框大部分落在切片里才保留
                    inter_area = (ix2 - ix1) * (iy2 - iy1)
                    orig_area = (bx2 - bx1) * (by2 - by1)
                    if orig_area > 0 and (inter_area / orig_area) >= 0.4:
                        # 转换为相对切片的 YOLO 归一化 xywh
                        cx = ((ix1 + ix2) / 2.0 - x) / crop_w
                        cy = ((iy1 + iy2) / 2.0 - y) / crop_h
                        w = (ix2 - ix1) / crop_w
                        h = (iy2 - iy1) / crop_h
                        patch_labels.append(f"{cls_to_idx[obj_id]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                        box_count += 1
            
            # 如果切片里有目标，或者保留 10% 随机无目标切片作为负样本
            if patch_labels or (patch_count % 10 == 0):
                patch_name = f"hel_{bname}_{x}_{y}"
                crop_img = img[y:y2_crop, x:x2_crop]
                cv2.imwrite(f"{OUTPUT_DIR}/images/{split}/{patch_name}.png", crop_img)
                with open(f"{OUTPUT_DIR}/labels/{split}/{patch_name}.txt", "w") as lf:
                    lf.write("\n".join(patch_labels))
                patch_count += 1

print(f"Helsinki 切片完毕！生成切片图: {patch_count} 张，提取高精度战术框: {box_count} 个。")

# 2. 写入 data.yaml
yaml_content = f"""path: {os.path.abspath(OUTPUT_DIR)}
train: images/train
val: images/val

names:
"""
for idx, name in enumerate(OBJECT_CLASSES):
    yaml_content += f"  {idx}: {name}\n"

with open(f"{OUTPUT_DIR}/data.yaml", "w") as f:
    f.write(yaml_content)

print("data.yaml 已经写入，正在打包数据集...")