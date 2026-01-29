import cv2
import time
from ultralytics import YOLO

# ===============================
# LOAD YOLOv8 POSE
# ===============================
model = YOLO("yolov8n-pose.pt")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_height = frame.shape[0]

    results = model(frame, conf=0.5, verbose=False)
    frame = results[0].plot()

    for r in results:
        if r.keypoints is None:
            continue

        kps = r.keypoints.xy.cpu().numpy()  # (N, 17, 2)

        for pid, person in enumerate(kps):

            # ===== LẤY KEYPOINT =====
            Nose = person[0]
            LS, RS = person[5], person[6]
            LH, RH = person[11], person[12]
            LA, RA = person[15], person[16]

            # ===== TÍNH CHIỀU CAO CƠ THỂ =====
            body_height = max(LA[1], RA[1]) - min(Nose[1], LS[1], RS[1])
            ratio = body_height / frame_height

            # ===== PHÂN LOẠI HÀNH VI =====
            if ratio < 0.25:
                state = "LYING"
                color = (0, 0, 255)
            elif ratio < 0.5:
                state = "SITTING"
                color = (0, 255, 255)
            else:
                state = "STANDING"
                color = (0, 255, 0)

            # ===== HIỂN THỊ =====
            cv2.putText(
                frame,
                f"ID {pid}: {state}",
                (20, 80 + pid * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

    # ===============================
    # FPS
    # ===============================
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {int(fps)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow("YOLOv8 Pose - Behavior Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
