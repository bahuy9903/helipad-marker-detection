import os
os.environ["TQDM_DISABLE"] = "1"

from ultralytics import YOLO

model = YOLO("D:/AI_CDT/helipad-marker-detection/models/yolov8n.pt")

results = model.train(
    data="D:/AI_CDT/helipad-marker-detection/data.yaml",
    epochs=50,
    imgsz=320,
    batch=8,
    device='cuda',
    workers=0,
    verbose=False
)