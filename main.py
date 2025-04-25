import cv2
import time
import threading

from modules.face_recognition import detect_faces
from modules.gaze_tracking import get_gaze_direction
from modules.object_detection import detect_objects
from modules.utils import (
    save_log,
    log_to_csv,
    play_alarm,
    calculate_face_match_score
)

# Setup kamera
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

frame_count = 0
face_reference = None
last_detected_objects = []
yolo_lock = threading.Lock()

# Fungsi background YOLO (non-blocking)
def run_object_detection(frame):
    global last_detected_objects
    detections = detect_objects(frame)
    with yolo_lock:
        last_detected_objects = detections

prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Deteksi wajah
    faces, face_roi = detect_faces(frame)

    if face_reference is None and face_roi is not None:
        face_reference = face_roi.copy()

    match_score = calculate_face_match_score(face_reference, face_roi)

    # Deteksi arah pandangan
    gaze = get_gaze_direction(frame)

    # Deteksi wajah ganda
    if len(faces) == 0:
        name = "Unknown"
    elif len(faces) == 1:
        name = "Detected"
    else:
        name = "Multiple Faces"
        save_log(name, "Multiple Face Detected", frame)
        log_to_csv("Multiple Face", "More than 1 face detected")
        play_alarm()

    # Jalankan YOLO setiap 10 frame (non-blocking)
    if frame_count % 10 == 0:
        threading.Thread(target=run_object_detection, args=(frame.copy(),)).start()

    with yolo_lock:
        current_objects = list(last_detected_objects)

    # Proses hasil object detection
    for label, conf, x1, y1, x2, y2 in current_objects:
        if label in ["cell phone", "laptop", "remote"]:
            cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            save_log(name, f"Detected object: {label}", frame)
            log_to_csv("Gadget Detected", label)
            play_alarm()

    # Status cheating dari arah pandangan
    if gaze in ["Looking Down","Looking Down (Head)", "Looking Right", "Looking Left"]:
        save_log(name, gaze, frame)
        log_to_csv("Gaze Cheating", gaze)
        play_alarm()

    # Tampilkan info
    cv2.putText(frame, f"Status: {name}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.putText(frame, f"Gaze: {gaze}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"Match Score: {match_score:.2f}%", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Hitung FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if curr_time - prev_time > 0 else 0
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

    cv2.imshow("AI Proctoring System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
