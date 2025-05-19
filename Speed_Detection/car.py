import os
import time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import imutils
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# Video dosyasının tam yolunu belirtelim
current_dir = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(current_dir, "car.mp4")
output_path = os.path.join(current_dir, "car_tracked_output.mp4")
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise ValueError(f"Video dosyası açılamadı: {video_path}")

model_name = "yolov8n.pt"
model = YOLO(model_name)

vehicle_id = 2  # Araç sınıfı (örneğin araba)
track_history = defaultdict(lambda: [])
previous_positions = defaultdict(lambda: None)
speed_history = defaultdict(lambda: [])
vehicle_colors = {}

fps = cap.get(cv2.CAP_PROP_FPS)
# FPS değeri 0 olduğunda varsayılan değer olarak 30 kullanalım
if fps <= 0:
    fps = 30.0
    print(f"Uyarı: FPS değeri 0 olarak okundu. Varsayılan değer {fps} kullanılıyor.")

frame_time = 1 / fps
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Çıktı boyutları geçersizse varsayılan değerler kullanalım
if width <= 0 or height <= 0:
    width, height = 1280, 720
    print(f"Uyarı: Video boyutları geçersiz. Varsayılan değerler kullanılıyor: {width}x{height}")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

scale_factor = 0.05  # km/piksel

# Etiket konumlarını takip etmek için liste
label_positions = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = imutils.resize(frame, width=1280)

    results = model.track(frame, persist=True, verbose=False)[0]
    bboxes = np.array(results.boxes.data.tolist(), dtype="int")

    label_positions.clear()  # Her karede konumları sıfırla

    for box in bboxes:
        x1, y1, x2, y2, track_id, score, class_id = box
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        if class_id == vehicle_id:
            if track_id not in vehicle_colors:
                vehicle_roi = frame[y1:y2, x1:x2]
                avg_color_per_row = np.average(vehicle_roi, axis=0)
                avg_color = np.average(avg_color_per_row, axis=0).astype(int)

                avg_color_bgr = np.uint8([[avg_color]])
                avg_color_hsv = cv2.cvtColor(avg_color_bgr, cv2.COLOR_BGR2HSV)
                h, s, v = avg_color_hsv[0][0]

                s = min(s + 150, 255)
                v = min(v + 100, 255)

                vehicle_colors[track_id] = cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0][0]

            color = tuple(map(int, vehicle_colors[track_id]))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

            if previous_positions[track_id] is not None:
                prev_cx, prev_cy = previous_positions[track_id]
                distance = np.sqrt((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2) * scale_factor
                speed_kmh = (distance / frame_time) * 15.6

                speed_history[track_id].append(speed_kmh)
                if len(speed_history[track_id]) > 5:
                    speed_history[track_id].pop(0)

                avg_speed_kmh = np.mean(speed_history[track_id])

                text = f"ID:{track_id} CAR, Speed: {avg_speed_kmh:.2f} km/h"

                # Etiketin yeri başka bir etiketle çakışıyorsa kaydır
                label_y = y1 - 10
                while any(abs(label_y - pos[1]) < 20 for pos in label_positions):  # 20 piksel fark
                    label_y -= 20  # Etiketi yukarı kaydır

                label_positions.append((x1, label_y))  # Bu konumu ekle
                cv2.putText(frame, text, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 3)
                cv2.arrowedLine(frame, (prev_cx, prev_cy), (cx, cy), color, 2, tipLength=0.3)

            previous_positions[track_id] = (cx, cy)

            track = track_history[track_id]
            track.append((cx, cy))
            if len(track) > 15:
                track.pop(0)

            points = np.hstack(track).astype("int32").reshape((-1, 1, 2))
            cv2.polylines(frame, [points], isClosed=False, color=color, thickness=2)

    out.write(frame)

    cv2.imshow("Car_Detection_New", frame)
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
