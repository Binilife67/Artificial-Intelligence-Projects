import ultralytics
from ultralytics import YOLO
import cv2
import numpy as np
import time
import os

video_path = r'C:\Users\USER\Downloads\Video\Daily Observations 293.mp4'

model_name = 'yolov8n.pt' # yolov8s or yolov8m
conf_threshold = 0.25      
iou_threshold = 0.45    
target_classes = [2] # 0 person or 1 bicycle
display = True              
save_output = True          
output_path = 'yolov8_car_output.mp4'
device = 'cpu' # 'cpu' or '0' (GPU index)/ use 'cuda:0'

model = YOLO(model_name) # YOLO(model_name, device='cpu')

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"Could not open the video")

fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if save_output:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
else:
    writer = None

COLORS = {
    2: (0, 255, 0),  
}
LABEL_BG_COLOR = (0, 0, 0)

frame_idx = 0
t0 = time.time()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        results = model(frame, imgsz=640, conf=conf_threshold, iou=iou_threshold, device=device)[0]

        if hasattr(results, 'boxes') and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()   
            scores = results.boxes.conf.cpu().numpy()   
            classes = results.boxes.cls.cpu().numpy().astype(int)  
        else:
            boxes = np.empty((0,4))
            scores = np.empty((0,))
            classes = np.empty((0,), dtype=int)

        for (box, score, cls) in zip(boxes, scores, classes):
            if cls not in target_classes:
                continue
            x1, y1, x2, y2 = map(int, box)
            color = COLORS.get(cls, (255, 0, 0))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"car {score:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), LABEL_BG_COLOR, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        if frame_idx % 10 == 0:
            elapsed = time.time() - t0
            fps_est = frame_idx / elapsed if elapsed > 0 else 0
            cv2.putText(frame, f"FPS: {fps_est:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        if display:
            cv2.imshow('YOLOv8 Car Detector', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if writer is not None:
            writer.write(frame)

finally:
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print(f"Processing finished. {"Not saved" if not save_output else "Saved to:", output_path}")