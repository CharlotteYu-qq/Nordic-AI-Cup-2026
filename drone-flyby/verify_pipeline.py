# verify_pipeline.py
import cv2
import json
import numpy as np
from ultralytics import YOLO
import dtos
from dtos import DroneFlybyPredictRequestDto, DroneFlybyViewDto
from utils import decode_view, draw_boxes, view_bbox_to_global, clip_bbox_to_frame

MODEL_PATH = "/home/azureuser/Nordic-AI-Cup-2026/drone-flyby/best.pt"
model = YOLO(MODEL_PATH)

# 读取 Helsinki 第 0 帧测试
from utils import load_frame, load_annotations, encode_image

orig_img, gt_annos = load_frame(0), load_annotations(0)
h, w = orig_img.shape[:2]

# 模拟 Level 0 传入的 960x540 视图
downsampled = cv2.resize(orig_img, (960, 540))
encoded_str = encode_image(downsampled)

fake_request = DroneFlybyPredictRequestDto(
    request_id="test",
    sequence_id="helsinki",
    frame=0,
    original_width=3840,
    original_height=2160,
    view=DroneFlybyViewDto(
        resolution_level=0,
        center_x=1920,
        center_y=1080,
        source_region_xyxy=[0, 0, 3840, 2160],
        image=encoded_str,
    )
)

# 1. 测试 BGR 输入 vs RGB 输入的检出差异
print("--- 正在对比 BGR vs RGB 检测置信度 ---")
res_bgr = model.predict(downsampled, conf=0.03, imgsz=960, verbose=False)[0]
rgb_img = cv2.cvtColor(downsampled, cv2.COLOR_BGR2RGB)
res_rgb = model.predict(rgb_img, conf=0.03, imgsz=960, verbose=False)[0]

print(f"BGR 输入检出框数: {len(res_bgr.boxes)}, 最高置信度: {max([float(b.conf[0]) for b in res_bgr.boxes], default=0):.3f}")
print(f"RGB 输入检出框数: {len(res_rgb.boxes)}, 最高置信度: {max([float(b.conf[0]) for b in res_rgb.boxes], default=0):.3f}")

# 2. 检查全局坐标还原是否能与真实 GT 框重合
print("\n--- 检查真实 GT 框 (前 3 个) ---")
for g in gt_annos[:3]:
    print(f"GT: {g['object_id']}, box: {g['bbox']}")

print("\n--- 检查当前 Pipeline 计算出的还原坐标 ---")
if len(res_bgr.boxes) > 0:
    for b in res_bgr.boxes[:3]:
        x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
        view_b = (float(x1/960), float(y1/540), float(x2/960), float(y2/540))
        g_box = view_bbox_to_global(view_b, fake_request.view.source_region_xyxy, 3840, 2160)
        # 还原为原图像素
        recon_px = [g_box[0]*3840, g_box[1]*2160, g_box[2]*3840, g_box[3]*2160]
        cls_name = model.names[int(b.cls[0])]
        print(f"Pred: {cls_name} (conf {float(b.conf[0]):.2f}), 还原像素: {[round(v, 1) for v in recon_px]}")