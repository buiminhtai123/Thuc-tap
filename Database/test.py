import cv2
import time
import numpy as np
import torch
from datetime import datetime
from pymongo import MongoClient
from ultralytics import YOLO

# ===============================
# MONGODB CONFIG
# ===============================
MONGO_URI = "mongodb+srv://buiminhtai1234:191104@cluster0.ydqe2ve.mongodb.net/?retryWrites=true&w=majority"

client = MongoClient(MONGO_URI)
db = client["iot_project"]
collection = db["human_behavior"]

print("✅ Connected to MongoDB")

# ===============================
# DEVICE
# ===============================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on:", device)

# ===============================
# LOAD YOLOv8 POSE
# ===============================
model = YOLO("Camera/yolov8n-pose.pt")
model.to(device)

# ===============================
# CAMERA
# ===============================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

prev_time = 0
last_db_time = 0
DB_INTERVAL = 1.0  # giây

# ===============================
# PERSON STATE MEMORY
# ===============================
person_states = {}
SIT_WARNING_TIME = 5.0  # giây

# ===============================
# MAIN LOOP
# ===============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    now_time = time.time()

    # ===== YOLO INFERENCE =====
    results = model(frame, conf=0.5, device=0, verbose=False)

    people = []

    for r in results:
        if r.keypoints is None:
            continue

        kps = r.keypoints.xy.cpu().numpy()
        boxes = r.boxes.xyxy.cpu().numpy()

        for pid, person in enumerate(kps):
            x1, y1, x2, y2 = map(int, boxes[pid])

            LS, RS = person[5], person[6]
            LH, RH = person[11], person[12]
            LK, RK = person[13], person[14]

            xs = person[:, 0]
            ys = person[:, 1]

            pose_h = ys.max() - ys.min()
            pose_w = xs.max() - xs.min()

            if pose_h < 10 or pose_w < 10:
                continue

            aspect = pose_h / (pose_w + 1e-6)

            mid_s = (LS + RS) / 2
            mid_h = (LH + RH) / 2
            angle = abs(np.degrees(np.arctan2(
                mid_s[1] - mid_h[1],
                mid_s[0] - mid_h[0]
            )))

            knee_y = (LK[1] + RK[1]) / 2
            hip_y = (LH[1] + RH[1]) / 2
            leg_ratio = abs(knee_y - hip_y) / (pose_h + 1e-6)

            # ===== CLASSIFY BEHAVIOR =====
            if aspect < 0.8 or angle < 30:
                behavior = "LYING"
                color = (0, 0, 255)
            elif leg_ratio < 0.25:
                behavior = "SITTING"
                color = (0, 255, 255)
            else:
                behavior = "STANDING"
                color = (0, 255, 0)

            # ===== STATE TRACKING =====
            if pid not in person_states:
                person_states[pid] = {
                    "last_behavior": behavior,
                    "start_time": now_time
                }

            state = person_states[pid]

            if behavior != state["last_behavior"]:
                state["last_behavior"] = behavior
                state["start_time"] = now_time

            duration = now_time - state["start_time"]

            # ===== ALERT LOGIC =====
            alert_type = "normal"
            alert_level = "normal"

            if behavior == "STANDING":
                alert_type = "normal"
                alert_level = "normal"
                duration = 0

            elif behavior == "SITTING":
                alert_type = "abnormal"
                if duration >= SIT_WARNING_TIME:
                    alert_level = "warning"
                else:
                    alert_level = "normal"

            elif behavior == "LYING":
                alert_type = "abnormal"
                alert_level = "warning"

            # ===== DRAW BOX =====
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # ===== DRAW TEXT=====
            cv2.putText(
                frame,
                f"{behavior} | {alert_level} | {int(duration)}s",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

            # ===== PUSH DATA =====
            people.append({
                "person_id": pid,
                "behavior": behavior,
                "type": alert_type,
                "level": alert_level,
                "duration": round(duration, 2)
            })

    # ===== INSERT DB =====
    if people and (now_time - last_db_time > DB_INTERVAL):
        doc = {
            "camera_id": "CAM_01",
            "timestamp": datetime.now(),
            "people": people
        }
        collection.insert_one(doc)
        last_db_time = now_time
        print(f"DB updated: {len(people)} people")

    # ===== FPS =====
    curr = time.time()
    fps = 1 / (curr - prev_time) if prev_time else 0
    prev_time = curr

    cv2.putText(frame, f"FPS: {int(fps)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow("YOLOv8 Pose + Alert + MongoDB", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# ===============================
# CLEANUP
# ===============================
cap.release()
cv2.destroyAllWindows()
client.close()
