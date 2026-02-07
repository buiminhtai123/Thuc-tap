import cv2
import face_recognition
import os
import numpy as np
import time

# ===============================
# CONFIG
# ===============================
FACE_ROOT = "/home/tai/Ung_dung/Code/Python/faces_db"
THRESHOLD = 0.45   # càng thấp càng ít nhận nhầm

previousTime = 0

known_encodings = []
known_names = []

# ===============================
# STEP 1 + 2: LOAD & ENCODE (MEAN PER PERSON)
# ===============================
print("[INFO] Loading & encoding faces...")

for person_name in os.listdir(FACE_ROOT):
    person_dir = os.path.join(FACE_ROOT, person_name)
    if not os.path.isdir(person_dir):
        continue

    person_encs = []

    for img_name in os.listdir(person_dir):
        if img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(person_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encs = face_recognition.face_encodings(rgb)

            if encs:
                person_encs.append(encs[0])

    if len(person_encs) >= 3:   # tối thiểu 3 ảnh / người
        mean_enc = np.mean(person_encs, axis=0)
        known_encodings.append(mean_enc)
        known_names.append(person_name)
        print(f"  [+] {person_name}: {len(person_encs)} images")

print("[INFO] Persons loaded:", known_names)

# ===============================
# CAMERA
# ===============================
cap = cv2.VideoCapture(0)

# ===============================
# REALTIME FACE RECOGNITION
# ===============================
while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)

    # resize để tăng FPS
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    # rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # detect face
    face_locations = face_recognition.face_locations(
        small_frame, number_of_times_to_upsample=1, model="hog"
    )
    face_encodings = face_recognition.face_encodings(small_frame, face_locations)

    for face_encoding, face_loc in zip(face_encodings, face_locations):
        name = "Unknown"

        if len(known_encodings) > 0:
            distances = face_recognition.face_distance(
                known_encodings, face_encoding
            )
            best_idx = np.argmin(distances)

            if distances[best_idx] < THRESHOLD:
                name = known_names[best_idx].upper()

        # scale back box
        top, right, bottom, left = face_loc
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2

        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 255), 2)
        cv2.putText(
            frame,
            name,
            (left, top - 10),
            cv2.FONT_HERSHEY_COMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

    # ===============================
    # FPS
    # ===============================
    currentTime = time.time()
    fps = 1 / (currentTime - previousTime) if previousTime else 0
    previousTime = currentTime

    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (10, 40),
        cv2.FONT_HERSHEY_COMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    cv2.imshow("Face Recognition (Stable)", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
