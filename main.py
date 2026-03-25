from ultralytics import YOLO

# model = YOLO('yolov8n.pt')
model = YOLO("D:/AI_CDT/helipad-marker-detection/models/yolov8n.yaml")

results = model.train(
    data="D:/AI_CDT/helipad-marker-detection/data.yaml",
    epochs=50,
    imgsz=320,
    batch=8,       # hoặc 16 nếu GPU đủ VRAM
    device='cuda',
    workers=0      # 🔑 tắt multiprocessing
)