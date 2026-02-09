import cv2
import os
import time

# ===============================
# ĐƯỜNG DẪN GỐC (CỐ ĐỊNH)
# ===============================
BASE_DIR = "/home/tai/Ung_dung/Code/Python/faces_db"

# ===============================
# NHẬP TÊN NGƯỜI DÙNG
# ===============================
user_name = input("Nhập tên người dùng: ").strip()

user_dir = os.path.join(BASE_DIR, user_name)
os.makedirs(user_dir, exist_ok=True)

print(f"Lưu ảnh tại: {user_dir}")
    
# ===============================
# ĐẾM ẢNH ĐÃ CÓ
# ===============================
existing_imgs = [
    f for f in os.listdir(user_dir)
    if f.lower().endswith((".jpg", ".png"))
]
img_count = len(existing_imgs)

# ===============================
# LOAD FACE CASCADE
# ===============================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ===============================
# MỞ CAMERA
# ===============================
cap = cv2.VideoCapture(0)

print("Nhấn 's' để chụp 5 ảnh (mỗi ảnh cách 3s), 'q' để thoát")

# ===============================
# BIẾN BỔ SUNG
# ===============================
CAPTURE_TOTAL = 5
CAPTURE_INTERVAL = 3  # giây

is_capturing = False
capture_count = 0
last_capture_time = 0

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        print("Không mở được camera")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(100, 100)
    )

    # ===============================
    # VẼ KHUNG (ĐỔI MÀU KHI ĐANG CHỤP)
    # ===============================
    for (x, y, w, h) in faces:
        if is_capturing:
            color = (0, 0, 255)   # 🔴 Đang chụp
        else:
            color = (0, 255, 0)   # 🟢 Bình thường

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    cv2.imshow("Face Capture", frame)

    key = cv2.waitKey(1) & 0xFF
    current_time = time.time()

    # ===============================
    # BẮT ĐẦU CHỤP 5 ẢNH
    # ===============================
    if key == ord('s') and not is_capturing and len(faces) > 0:
        is_capturing = True
        capture_count = 0
        last_capture_time = 0
        print("▶️ Bắt đầu chụp 5 ảnh...")

    # ===============================
    # TIẾN TRÌNH CHỤP TỰ ĐỘNG
    # ===============================
    if is_capturing and len(faces) > 0:
        if last_capture_time == 0 or (current_time - last_capture_time >= CAPTURE_INTERVAL):
            (x, y, w, h) = faces[0]  # chỉ lấy khuôn mặt đầu tiên
            face_img = frame[y:y + h, x:x + w]

            img_count += 1
            capture_count += 1

            filename = f"{user_name}_{img_count:03d}.jpg"
            filepath = os.path.join(user_dir, filename)

            cv2.imwrite(filepath, face_img)
            print(f"📸 Đã lưu ({capture_count}/{CAPTURE_TOTAL}): {filepath}")

            last_capture_time = current_time

            if capture_count >= CAPTURE_TOTAL:
                is_capturing = False
                print("✅ Hoàn tất chụp 5 ảnh")

    # ===============================
    # THOÁT
    # ===============================
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
