from ultralytics import YOLO
import numpy as np

model = YOLO("yolov8n.pt")  

def detect_objects(frame):
    results = model.predict(source=frame, imgsz=640, conf=0.4, verbose=False)

    objects = []
    if results and len(results) > 0:
        boxes = results[0].boxes
        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                objects.append((label, conf, x1, y1, x2, y2))
    return objects
