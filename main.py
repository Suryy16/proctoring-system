import cv2
import time
import threading

from recognition_scripts.dataset_processor import DatasetProcessor
import recognition_scripts.face_utils as face_utils
from modules.gaze_tracking import get_gaze_direction
from modules.object_detection import detect_objects
from modules.utils import (
    save_log,
    log_to_csv,
    play_alarm,
    calculate_face_match_score
)

# Inisialisasi kamera
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Inisialisasi detektor wajah (YOLO + face recognition)
detector = DatasetProcessor()
frame_count = 0
recognition_interval = 5
previous_results = []

face_reference = None
last_detected_objects = []
yolo_lock = threading.Lock()
prev_time = 0

# Fungsi deteksi objek (non-blocking)
def run_object_detection(frame):
    global last_detected_objects
    detections = detect_objects(frame)
    with yolo_lock:
        last_detected_objects = detections

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    faces = detector.detect_faces(frame)
    face_roi = None

    # Ambil ROI wajah pertama (jika ada) untuk referensi
    if faces:
        x, y, w, h = faces[0]
        face_roi = frame[y:y+h, x:x+w]

        if face_reference is None:
            face_reference = face_roi.copy()

    match_score = calculate_face_match_score(face_reference, face_roi)

    # Lakukan pengenalan wajah setiap N frame
    if frame_count % recognition_interval == 0:
        previous_results = []
        for (x, y, w, h) in faces:
            identity, similarity = face_utils.recognize_face(frame, (x, y, w, h))
            label = f"{identity} ({similarity * 100:.1f}%)" if identity != "Unknown" else "Unknown"
            previous_results.append(((x, y, w, h), label))
    else:
        if len(previous_results) != len(faces):
            previous_results = [((x, y, w, h), "Unknown") for (x, y, w, h) in faces]

    # Deteksi arah pandangan
    gaze = get_gaze_direction(frame)

    # Deteksi kondisi wajah
    if len(faces) == 0:
        name_status = "Unknown"
    elif len(faces) == 1:
        name_status = previous_results[0][1].split('(')[0].strip() if previous_results else "Unknown"
    else:
        name_status = "Multiple Faces"
        save_log(name_status, "Multiple Face Detected", frame)
        log_to_csv("Multiple Face", "More than 1 face detected")
        play_alarm()

    # Jalankan YOLO setiap 10 frame
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
            save_log(name_status, f"Detected object: {label}", frame)
            log_to_csv("Gadget Detected", label)
            play_alarm()

    # Deteksi arah pandangan mencurigakan
    if gaze in ["Looking Down", "Looking Down (Head)", "Looking Right", "Looking Left"]:
        save_log(name_status, gaze, frame)
        log_to_csv("Gaze Cheating", gaze)
        play_alarm()

    # Tampilkan informasi
    cv2.putText(frame, f"Status: {name_status}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.putText(frame, f"Gaze: {gaze}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"Match Score: {match_score:.2f}%", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if curr_time - prev_time > 0 else 0
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    for i, (x, y, w, h) in enumerate(faces):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
        label = previous_results[i][1] if i < len(previous_results) else "Unknown"
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow("AI Proctoring System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
