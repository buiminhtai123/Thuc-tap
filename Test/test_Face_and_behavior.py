import cv2
import face_recognition
import os
import numpy as np
import time
from datetime import datetime
from pymongo import MongoClient
from ultralytics import YOLO

# ===============================
# CONFIG
# ===============================

FACE_ROOT = "Data_faces"
THRESHOLD = 0.45
DB_INTERVAL = 1
CAMERA_ID = "CAM_01"
TIMEOUT = 1

# ===============================
# MONGODB
# ===============================

MONGO_URI = "mongodb+srv://buiminhtai1234:191104@cluster0.ydqe2ve.mongodb.net/?retryWrites=true&w=majority"

client = MongoClient(MONGO_URI)

db = client["iot_project"]

collection = db["Test_database"]

# ===============================
# LOAD YOLO
# ===============================

model = YOLO("yolov8n-pose.pt")

# ===============================
# LOAD FACE DATASET
# ===============================

print("[INFO] Loading faces...")

known_encodings = []
known_names = []

for person_name in os.listdir(FACE_ROOT):

    person_dir = os.path.join(FACE_ROOT, person_name)

    if not os.path.isdir(person_dir):
        continue

    enc_list = []

    for img_name in os.listdir(person_dir):

        if img_name.endswith((".jpg",".png",".jpeg")):

            img_path = os.path.join(person_dir,img_name)

            img = cv2.imread(img_path)

            if img is None:
                continue

            rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

            encs = face_recognition.face_encodings(rgb)

            if len(encs) > 0:
                enc_list.append(encs[0])

    if len(enc_list) > 0:

        mean_encoding = np.mean(enc_list,axis=0)

        known_encodings.append(mean_encoding)
        known_names.append(person_name.upper())

        print(f"[INFO] Loaded {person_name} : {len(enc_list)} images")

print("[INFO] Persons:",known_names)

# ===============================
# TRACKING VARIABLES
# ===============================

person_start_time = {}
person_behavior = {}
person_last_seen = {}

unknown_counter = 0

last_db_time = 0

# ===============================
# CAMERA
# ===============================

cap = cv2.VideoCapture(0)

prev_time = 0

# ===============================
# MAIN LOOP
# ===============================

while True:

    ret,frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame,1)

    results = model(frame)

    people_data_for_db = []

    detected_people = []

    # ===============================
    # DETECTION
    # ===============================

    for r in results:

        boxes = r.boxes

        for box in boxes:

            cls = int(box.cls[0])

            if cls != 0:
                continue

            x1,y1,x2,y2 = map(int,box.xyxy[0])

            person_img = frame[y1:y2,x1:x2]

            name = None

            # ===============================
            # FACE RECOGNITION
            # ===============================

            if person_img.size != 0:

                rgb = cv2.cvtColor(person_img,cv2.COLOR_BGR2RGB)

                faces = face_recognition.face_locations(rgb)

                encs = face_recognition.face_encodings(rgb,faces)

                if len(encs) > 0:

                    face_encoding = encs[0]

                    distances = face_recognition.face_distance(
                        known_encodings,
                        face_encoding
                    )

                    best_idx = np.argmin(distances)

                    if distances[best_idx] < THRESHOLD:

                        name = known_names[best_idx]

            # ===============================
            # HANDLE UNKNOWN
            # ===============================

            if name is None:

                name = f"UNKNOWN_{unknown_counter}"

                unknown_counter += 1

            detected_people.append(name)

            # ===============================
            # SIMPLE BEHAVIOR
            # ===============================

            h = y2 - y1
            w = x2 - x1

            if h > w * 1.2:

                behavior = "standing"
                level = "normal"

            else:

                behavior = "sitting"
                level = "warning"

            current_time = time.time()

            # ===============================
            # TRACKING LOGIC
            # ===============================

            if name not in person_start_time:

                person_start_time[name] = current_time
                person_behavior[name] = behavior
                person_last_seen[name] = current_time

            else:

                if person_behavior[name] != behavior:

                    person_start_time[name] = current_time
                    person_behavior[name] = behavior

                person_last_seen[name] = current_time

            duration = int(current_time - person_start_time[name])

            # ===============================
            # DRAW
            # ===============================

            display_name = name.split("_")[0]

            label = f"{display_name} | {behavior} | {duration}s"

            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

            cv2.putText(
                frame,
                label,
                (x1,y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0,255,0),
                2
            )

            # ===============================
            # DB DATA
            # ===============================

            people_data_for_db.append({

                "person_id": display_name,
                "behavior": behavior,
                "duration": duration,
                "level": level

            })

    # ===============================
    # REMOVE PEOPLE WHO LEFT FRAME
    # ===============================

    remove_list = []

    for name in person_last_seen:

        if time.time() - person_last_seen[name] > TIMEOUT:

            remove_list.append(name)

    for name in remove_list:

        del person_last_seen[name]
        del person_start_time[name]
        del person_behavior[name]

    # ===============================
    # SEND TO DATABASE
    # ===============================

    now = time.time()

    if now - last_db_time > DB_INTERVAL and len(people_data_for_db) > 0:

        try:

            document = {

                "camera_id": CAMERA_ID,

                "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),

                "people": people_data_for_db

            }

            collection.insert_one(document)

            print("Inserted:", document)

        except Exception as e:

            print("MongoDB error:", e)

        last_db_time = now

    # ===============================
    # FPS
    # ===============================

    current_time = time.time()

    fps = 1/(current_time-prev_time) if prev_time else 0

    prev_time = current_time

    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,255),
        2
    )

    # ===============================
    # DISPLAY
    # ===============================

    cv2.imshow("Behavior + Face Recognition",frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()

cv2.destroyAllWindows()