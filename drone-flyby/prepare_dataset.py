# prepare_dataset.py
import glob
import json
import os
import cv2
import numpy as np
from dtos import OBJECT_CLASSES

CLASS_TO_IDX = {name: idx for idx, name in enumerate(OBJECT_CLASSES)}
BOOST_CLASSES = {"hangar", "medium_launcher", "medium_plane", "mine_roller", "small_launcher"}

SRC_IMG_DIR = "src/helsinki/images"
SRC_ANN_DIR = "src/helsinki/annotations"
OUT_DIR = "datasets/helsinki_v2"


def make_dirs():
    for split in ["train", "val"]:
        os.makedirs(f"{OUT_DIR}/images/{split}", exist_ok=True)
        os.makedirs(f"{OUT_DIR}/labels/{split}", exist_ok=True)


def parse_annotations(raw_data):
    """安全解析不同格式的 annotation 数据，统一返回 List[dict]"""
    if isinstance(raw_data, list):
        return [item for item in raw_data if isinstance(item, dict)]
    elif isinstance(raw_data, dict):
        for key in ["annotations", "objects", "targets"]:
            if key in raw_data and isinstance(raw_data[key], list):
                return [item for item in raw_data[key] if isinstance(item, dict)]
        # 如果字典本身就是一个目标
        if "object_id" in raw_data and "bbox" in raw_data:
            return [raw_data]
    return []


def xyxy_to_yolo(box, w, h):
    x1, y1, x2, y2 = box
    cx = ((x1 + x2) / 2.0) / w
    cy = ((y1 + y2) / 2.0) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return (
        max(0.0, min(1.0, cx)),
        max(0.0, min(1.0, cy)),
        max(0.0, min(1.0, bw)),
        max(0.0, min(1.0, bh)),
    )


def main():
    make_dirs()
    ann_files = sorted(glob.glob(f"{SRC_ANN_DIR}/*.json"))
    sample_id = 0

    for idx, ann_path in enumerate(ann_files):
        split = "val" if idx in [4, 9, 14, 19] else "train"
        base_name = os.path.splitext(os.path.basename(ann_path))[0]
        img_path = os.path.join(SRC_IMG_DIR, f"{base_name}.png")
        if not os.path.exists(img_path):
            continue

        img_4k = cv2.imread(img_path)
        if img_4k is None:
            continue
        h_4k, w_4k = img_4k.shape[:2]

        with open(ann_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        annotations = parse_annotations(raw_data)
        if not annotations:
            continue

        # --------------------------------------------------
        # 模式 1: 全图降采样至 960x540 (模拟 Level 0 全景)
        # --------------------------------------------------
        l0_img = cv2.resize(img_4k, (960, 540))
        l0_lines = []
        for ann in annotations:
            cls_name = ann.get("object_id")
            bbox = ann.get("bbox")
            if not cls_name or not bbox or cls_name not in CLASS_TO_IDX:
                continue
            cx, cy, bw, bh = xyxy_to_yolo(bbox, w_4k, h_4k)
            l0_lines.append(f"{CLASS_TO_IDX[cls_name]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        cv2.imwrite(f"{OUT_DIR}/images/{split}/l0_{base_name}.png", l0_img)
        with open(f"{OUT_DIR}/labels/{split}/l0_{base_name}.txt", "w") as f:
            f.write("\n".join(l0_lines))
        sample_id += 1

        # --------------------------------------------------
        # 模式 2: 以难例目标为中心切片（模拟 Level 2 局部特写放大）
        # --------------------------------------------------
        for obj_idx, target_ann in enumerate(annotations):
            t_cls = target_ann.get("object_id")
            t_box = target_ann.get("bbox")
            if not t_cls or not t_box or t_cls not in CLASS_TO_IDX:
                continue

            repeat = 4 if (t_cls in BOOST_CLASSES and split == "train") else 1

            for r in range(repeat):
                x1, y1, x2, y2 = t_box
                jitter_x = np.random.randint(-80, 80) if r > 0 else 0
                jitter_y = np.random.randint(-80, 80) if r > 0 else 0

                cx = int((x1 + x2) / 2) + jitter_x
                cy = int((y1 + y2) / 2) + jitter_y

                crop_x1 = max(0, min(cx - 480, w_4k - 960))
                crop_y1 = max(0, min(cy - 270, h_4k - 540))
                crop_x2 = crop_x1 + 960
                crop_y2 = crop_y1 + 540

                crop_img = img_4k[crop_y1:crop_y2, crop_x1:crop_x2]

                crop_lines = []
                for other in annotations:
                    o_cls = other.get("object_id")
                    o_box = other.get("bbox")
                    if not o_cls or not o_box or o_cls not in CLASS_TO_IDX:
                        continue
                    ox1, oy1, ox2, oy2 = o_box

                    ix1 = max(crop_x1, ox1) - crop_x1
                    iy1 = max(crop_y1, oy1) - crop_y1
                    ix2 = min(crop_x2, ox2) - crop_x1
                    iy2 = min(crop_y2, oy2) - crop_y1

                    if ix2 > ix1 and iy2 > iy1:
                        orig_area = (ox2 - ox1) * (oy2 - oy1)
                        inter_area = (ix2 - ix1) * (iy2 - iy1)
                        if inter_area >= 0.3 * orig_area:
                            cx_y, cy_y, bw_y, bh_y = xyxy_to_yolo([ix1, iy1, ix2, iy2], 960, 540)
                            crop_lines.append(f"{CLASS_TO_IDX[o_cls]} {cx_y:.6f} {cy_y:.6f} {bw_y:.6f} {bh_y:.6f}")

                if crop_lines:
                    fname = f"crop_{base_name}_obj{obj_idx}_r{r}"
                    cv2.imwrite(f"{OUT_DIR}/images/{split}/{fname}.png", crop_img)
                    with open(f"{OUT_DIR}/labels/{split}/{fname}.txt", "w") as f:
                        f.write("\n".join(crop_lines))
                    sample_id += 1

    # 生成 data.yaml
    yaml_lines = [
        f"path: {os.path.abspath(OUT_DIR)}",
        "train: images/train",
        "val: images/val",
        "names:",
    ] + [f"  {i}: {name}" for i, name in enumerate(OBJECT_CLASSES)]

    with open(f"{OUT_DIR}/data.yaml", "w", encoding="utf-8") as f:
        f.write("\n".join(yaml_lines))

    print(f"增强数据集制作完成，有效样本图像总数: {sample_id}")


if __name__ == "__main__":
    main()