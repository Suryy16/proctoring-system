import os
from datetime import datetime
import cv2
import csv
import threading
import platform

def calculate_face_match_score(face_ref, face_curr):
    if face_ref is None or face_curr is None:
        return 0.0

    try:
        face_ref_gray = cv2.cvtColor(face_ref, cv2.COLOR_BGR2GRAY)
        face_curr_gray = cv2.cvtColor(face_curr, cv2.COLOR_BGR2GRAY)

        face_ref_gray = cv2.resize(face_ref_gray, (100, 100))
        face_curr_gray = cv2.resize(face_curr_gray, (100, 100))

        hist_ref = cv2.calcHist([face_ref_gray], [0], None, [256], [0, 256])
        hist_curr = cv2.calcHist([face_curr_gray], [0], None, [256], [0, 256])

        score = cv2.compareHist(hist_ref, hist_curr, cv2.HISTCMP_CORREL)
        return round(score * 100, 2)  # jadi skala 0–100
    except:
        return 0.0


def save_log(name, gaze, frame):
    os.makedirs("data/logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"data/logs/{name}_{gaze}_{timestamp}.jpg"
    cv2.imwrite(filename, frame)

def log_to_csv(event, label, file='data/logs/events.csv'):
    import time
    last_log_time = 0  # Tambahkan variabel cache

    timestamp = time.time()
    if timestamp - last_log_time < 2:  # Hanya log setiap 2 detik
        return

    last_log_time = timestamp
    # Jika file belum ada, tulis header
    file_exists = os.path.isfile(file)
    with open(file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Event", "Label"])
        writer.writerow([last_log_time, event, label])


def play_alarm():
    def _alarm():
        if platform.system() == "Windows":
            import winsound
            winsound.Beep(1000, 500)  # frequency, duration (ms)
        else:
            # Beep for Linux (requires `beep` installed)
            os.system('play -nq -t alsa synth 0.5 sine 1000')
    threading.Thread(target=_alarm).start()