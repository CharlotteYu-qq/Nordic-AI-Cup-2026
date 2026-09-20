# train.py
from ultralytics import YOLO

def train():
    # 升级为 yolov8s.pt，提升小目标表征能力
    model = YOLO("yolov8s.pt")

    model.train(
        data="datasets/helsinki_v2/data.yaml",
        epochs=80,
        imgsz=960,
        batch=4,
        workers=2,
        save=True,
        project="/home/azureuser/Nordic-AI-Cup-2026/runs/detect",
        name="drone_v8s_boost",
        exist_ok=True,
        # 关闭马赛克增强的后 10 个 epoch，避免边界框对不齐
        close_mosaic=10,
    )

if __name__ == "__main__":
    train()