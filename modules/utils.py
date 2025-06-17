import os
import cv2
from datetime import datetime
import csv

def save_log(name, message, frame, path='data/logs'):
    os.makedirs(path, exist_ok=True)
    filename = os.path.join(path, f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
    cv2.imwrite(filename, frame)

def log_to_csv(event, label, file='data/logs/events.csv'):
    os.makedirs(os.path.dirname(file), exist_ok=True)
    file_exists = os.path.exists(file)
    with open(file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Timestamp', 'Event', 'Label'])
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), event, label])

def play_alarm():
    # optional: implement playsound here or print warning
    print("[ALERT] Cheating detected!")

def calculate_face_match_score(ref, new):
    if ref is None or new is None:
        return 0.0
    ref = cv2.resize(ref, (100, 100))
    new = cv2.resize(new, (100, 100))
    diff = cv2.absdiff(ref, new)
    score = 100 - (cv2.mean(diff)[0] / 255 * 100)
    return max(0.0, min(100.0, score))