import cv2
from ultralytics import YOLO
import time
import numpy as np

# 1. Khởi tạo model
model = YOLO('yolov8n-pose.pt').to('cuda')
cap = cv2.VideoCapture(0)
prev_time = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    # Tính FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    results = model.predict(frame, device=0, verbose=False)

    for r in results:
        annotated_frame = r.plot(labels=False, boxes=True) 
        
        if r.boxes is not None and r.keypoints is not None:
            for box, kp_data in zip(r.boxes.xyxy, r.keypoints.data):
                # kp[index] = [x, y, confidence]
                kp = kp_data.cpu().numpy()
                x1, y1, x2, y2 = box.cpu().numpy()

                try:
                    # 2. TRÍCH XUẤT CÁC ĐIỂM QUAN TRỌNG
                    # Hông: 11(L), 12(R) | Đầu gối: 13(L), 14(R) | Cổ chân: 15(L), 16(R)
                    l_hip, r_hip = kp[11], kp[12]  
                    l_knee, r_knee = kp[13], kp[14]
                    l_ankle, r_ankle = kp[15], kp[16]
                    
                    # 3. TÍNH GÓC ĐẦU GỐI (Dùng công thức Vector trực tiếp)
                    # Tính cho chân trái
                    v1 = np.array([l_hip[0] - l_knee[0], l_hip[1] - l_knee[1]])
                    v2 = np.array([l_ankle[0] - l_knee[0], l_ankle[1] - l_knee[1]])
                    
                    # Công thức Cosine: cos(a) = (v1.v2) / (|v1|*|v2|)
                    unit_v1 = v1 / np.linalg.norm(v1)
                    unit_v2 = v2 / np.linalg.norm(v2)
                    angle_rad = np.arccos(np.clip(np.dot(unit_v1, unit_v2), -1.0, 1.0))
                    knee_angle = np.degrees(angle_rad)

                    # 4. THUẬT TOÁN TỐI ƯU
                    status = "Unknown"
                    color = (255, 255, 255) # Trắng

                    # Lấy chiều cao/rộng của Bounding Box
                    box_h = y2 - y1
                    box_w = x2 - x1

                    # Ưu tiên kiểm tra NẰM/NGÃ (Dựa vào tỉ lệ khung hình Box)
                    if box_w > box_h:
                        status = "Lying/Falling"
                        color = (0, 0, 255) # Đỏ
                    
                    # Nếu Box đứng, phân biệt ĐỨNG hay NGỒI bằng góc đầu gối
                    else:
                        # Nếu góc gối > 150 độ là chân thẳng -> ĐỨNG
                        if knee_angle > 150:
                            status = "Standing"
                            color = (0, 255, 0) # Xanh lá
                        # Nếu góc gối nhỏ (thường < 130) -> NGỒI
                        elif knee_angle < 130:
                            status = "Sitting"
                            color = (255, 165, 0) # Cam
                        else:
                            status = "Bending" # Đang khom người/cúi
                            color = (0, 255, 255) # Vàng

                    # 5. VẼ NHÃN TỰ CHỈNH
                    label = f"{status}"
                    (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    cv2.rectangle(annotated_frame, (int(x1), int(y1) - h_text - 10), (int(x1) + w_text, int(y1)), color, -1)
                    cv2.putText(annotated_frame, label, (int(x1), int(y1) - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
                                
                except Exception:
                    # Nếu không đủ điểm (ví dụ bị che khuất), bỏ qua người đó
                    continue

        # Hiển thị FPS và Kết quả
        cv2.putText(annotated_frame, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow("Optimized Action Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == 27: break # Nhấn ESC để thoát

cap.release()
cv2.destroyAllWindows()