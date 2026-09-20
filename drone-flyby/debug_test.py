import os
import cv2
from ultralytics import YOLO

MODEL_PATH = "/home/azureuser/Nordic-AI-Cup-2026/runs/detect/runs/train/drone_v8n/weights/best.pt"

print(f"检查模型文件是否存在: {os.path.exists(MODEL_PATH)}")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"文件未找到: {MODEL_PATH}")

# 显式实例化模型
model = YOLO(MODEL_PATH)
print("模型加载成功！")
print("模型支持的类别列表:", model.names)

# 读取测试图像 (请确保在 drone-flyby 目录下运行该脚本)
img_path = "src/helsinki/images/frame_000000.png"
if not os.path.exists(img_path):
    raise FileNotFoundError(f"未找到测试图像: {img_path}，请确认运行目录在 drone-flyby/ 下")

img_4k = cv2.imread(img_path)
h, w = img_4k.shape[:2]

# 1. 测试切片图 (模拟 Level 2 局部视角)
crop_l2 = img_4k[h//2 - 270 : h//2 + 270, w//2 - 480 : w//2 + 480]
res_l2 = model.predict(source=crop_l2, conf=0.01, imgsz=960, verbose=False)[0]
print(f"\nLevel 2 局部切片检测到的目标数: {len(res_l2.boxes)}")
for box in res_l2.boxes:
    cls_id = int(box.cls.cpu().numpy()[0])
    conf = float(box.conf.cpu().numpy()[0])
    name = model.names.get(cls_id, "unknown")
    print(f" -> 检测到: {name} (ID: {cls_id}), 置信度: {conf:.3f}")

# 2. 测试整图缩放 (模拟 Level 0 全景视角)
crop_l0 = cv2.resize(img_4k, (960, 540))
res_l0 = model.predict(source=crop_l0, conf=0.01, imgsz=960, verbose=False)[0]
print(f"\nLevel 0 全景缩放检测到的目标数: {len(res_l0.boxes)}")
for box in res_l0.boxes:
    cls_id = int(box.cls.cpu().numpy()[0])
    conf = float(box.conf.cpu().numpy()[0])
    name = model.names.get(cls_id, "unknown")
    print(f" -> 检测到: {name} (ID: {cls_id}), 置信度: {conf:.3f}")