import cv2
import mediapipe as mp
import torch
import time

# =====================================
# MediaPipe Pose (FAST + STABLE)
# =====================================
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,        # tracking mode
    model_complexity=1,             # 0 = fastest, 1 = balanced
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# =====================================
# YOLOv5 (Person Detection)
# =====================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.hub.load(
    'ultralytics/yolov5',
    'yolov5n',
    pretrained=True
)

model.conf = 0.6        # confidence threshold
model.classes = [0]    # detect only person
model.to(device)
model.eval()
torch.set_grad_enabled(False)

detections = []

# =====================================
# Camera (LOW LATENCY)
# =====================================
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

frame_id = 0
prev_time = 0

# =====================================
# MAIN LOOP
# =====================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_id += 1

    # ---------------------------------
    # MediaPipe Pose (EVERY FRAME)
    # ---------------------------------
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        mp_draw.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_draw.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
            # mp_draw.DrawingSpec(color=(0,0,255), thickness=2)
        )

    # ---------------------------------
    # YOLOv5 (EVERY 3 FRAMES)
    # ---------------------------------
    if frame_id % 3 == 0:
        yolo_results = model(frame, size=416)
        detections = yolo_results.xyxy[0]

    # Draw YOLO bounding boxes
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            frame,
            f"Person {conf:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

    # ---------------------------------
    # FPS
    # ---------------------------------
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time

    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 0),
        2
    )

    # ---------------------------------
    # Display
    # ---------------------------------
    cv2.imshow("YOLOv5 + MediaPipe Pose", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# =====================================
# CLEANUP
# =====================================
cap.release()
cv2.destroyAllWindows()
