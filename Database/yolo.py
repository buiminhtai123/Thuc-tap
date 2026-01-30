import cv2
import time
import numpy as np
import requests
from ultralytics import YOLO
from datetime import datetime

SERVER_URL = "http://localhost:8000/pose"

model = YOLO("yolov8n-pose.pt")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.5, verbose=False)

    for r in results:
        if r.keypoints is None:
            continue

        kps = r.keypoints.xy.cpu().numpy()
        boxes = r.boxes.xyxy.cpu().numpy()

        for i, person in enumerate(kps):
            x1, y1, x2, y2 = map(int, boxes[i])

            ys = person[:, 1]
            pose_height = ys.max() - ys.min()

            state = "STANDING"
            if pose_height < 200:
                state = "LYING"

            data = {
                "label": state,
                "confidence": 0.9,
                "bbox": [x1, y1, x2, y2],
                "timestamp": datetime.now().isoformat(),
                "description": "Detected by YOLOv8 Pose"
            }

            requests.post(SERVER_URL, json=data)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)
            cv2.putText(frame, state, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    cv2.imshow("YOLO Pose Sender", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
