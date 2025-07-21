import cv2
import time
import threading
import joblib
import os
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np

# Modules
from modules.gaze_tracking import get_gaze_direction
from modules.gaze_paper import get_eye_gaze_ratios
from modules.head_pose import estimate_head_pose
from modules.feature_logger import save_features
from modules.face_recognition import detect_faces
from recognition_scripts.dataset_processor import DatasetProcessor
import recognition_scripts.face_utils as FaceRecognizer
from modules.gaze_tracking import get_gaze_direction
from modules.object_detection import detect_objects
from modules.utils import (
    save_log,
    log_to_csv,
    play_alarm,
    calculate_face_match_score
)

# Define CNN model class
class TransferCNN(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(TransferCNN, self).__init__()
        self.backbone = models.resnet18(pretrained=pretrained)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

# Load models with CNN as primary and Random Forest as fallback
CNN_MODEL_PATH = "model/transfer_learning_cnn.pth"
SCALER_PATH = "model/scaler.pkl"
USE_CNN = os.path.exists(CNN_MODEL_PATH) and os.path.exists(SCALER_PATH)

if USE_CNN:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cnn_model = TransferCNN(num_classes=2, pretrained=False)
    cnn_model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=device))
    cnn_model.eval()
    scaler = joblib.load(SCALER_PATH)
    print("✅ Transfer Learning CNN model loaded")
else:
    # Fallback to Random Forest
    MODEL_PATH = "D:\\sursat\\kuliah\\Semester 6\\PKL SE\\proctoring-system\\models\\model_rf.pkl"
    USE_PREDICTION = os.path.exists(MODEL_PATH)
    model = joblib.load(MODEL_PATH) if USE_PREDICTION else None
    print("⚠️ Using Random Forest fallback")

# Initialize camera
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

frame_count = 0
face_reference = None
last_detected_objects = []
yolo_lock = threading.Lock()
prev_time = 0
recognition_interval = 5
previous_results = []

detector = DatasetProcessor()

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

    if faces:
        x, y, w, h = faces[0]
        face_roi = frame[y:y+h, x:x+w]
        if face_reference is None:
            face_reference = face_roi.copy()

    match_score = calculate_face_match_score(face_reference, face_roi)

    if frame_count % recognition_interval == 0:
        previous_results = []
        for (x, y, w, h) in faces:
            identity, similarity = FaceRecognizer.recognize_face(frame, (x, y, w, h))
            label = f"{identity} ({similarity * 100:.1f}%)" if identity != "Unknown" else "Unknown"
            previous_results.append(((x, y, w, h), label))
    else:
        if len(previous_results) != len(faces):
            previous_results = [((x, y, w, h), "Unknown") for (x, y, w, h) in faces]

    hr, vr = get_eye_gaze_ratios(frame)
    pitch = yaw = None
    label = "Unknown"
    gaze = get_gaze_direction(frame)

    if hr is not None and vr is not None:
        if faces:
            x, y, w, h = faces[0]
            face_landmarks = {
                1: (x + w // 2, y + h // 2),
                152: (x + w // 2, y + h),
                33: (x, y + h // 3),
                263: (x + w, y + h // 3),
                61: (x + w // 3, y + (2 * h) // 3),
                291: (x + (2 * w) // 3, y + (2 * h) // 3),
            }
            pitch, yaw = estimate_head_pose(face_landmarks, frame.shape)

        save_features(hr, vr, pitch if pitch else 0, yaw)

        # CNN Prediction
        if USE_CNN and pitch is not None and yaw is not None:
            try:
                features = np.array([[hr, vr, pitch, yaw]], dtype=np.float32)
                features = scaler.transform(features)
                features = torch.tensor(features, dtype=torch.float32).to(device)
                
                with torch.no_grad():
                    outputs = cnn_model(features)
                    probabilities = torch.softmax(outputs, dim=1)
                    _, predicted = torch.max(outputs, 1)
                
                prediction = predicted.cpu().numpy()[0]
                confidence = probabilities.cpu().numpy()[0][prediction]
                
                label = f"CNN: {'CHEATING' if prediction == 1 else 'OK'} ({confidence:.2f})"
                
                if prediction == 1:
                    save_log(label, f"CNN: Cheating detected (confidence: {confidence:.2f})", frame)
                    log_to_csv("ML Cheating", "CNN predicted Cheating")
                    play_alarm()
                    
            except Exception as e:
                print(f"CNN prediction error: {e}")
                label = "CNN Error"
        
        # Fallback to Random Forest if CNN not available
        elif USE_PREDICTION and pitch is not None and yaw is not None:
            pred = model.predict([[hr, vr, pitch, yaw]])[0]
            label = f"RF: {'CHEATING' if pred == 1 else 'OK'}"
            if pred == 1:
                save_log(label, "RF: Cheating detected", frame)
                log_to_csv("ML Cheating", "RF predicted Cheating")
                play_alarm()
        else:
            label = "Logged Only"

    name_status = "Unknown"
    if len(faces) == 0:
        name_status = "Unknown"
    elif len(faces) == 1:
        name_status = previous_results[0][1].split('(')[0].strip() if previous_results else "Unknown"
    else:
        name_status = "Multiple Faces"
        save_log(name_status, "Multiple Face Detected", frame)
        log_to_csv("Multiple Face", "More than 1 face detected")
        play_alarm()

    if frame_count % 10 == 0:
        threading.Thread(target=run_object_detection, args=(frame.copy(),)).start()

    with yolo_lock:
        current_objects = list(last_detected_objects)

    for label_obj, conf, x1, y1, x2, y2 in current_objects:
        if label_obj in ["cell phone", "laptop", "remote"]:
            cv2.putText(frame, f"{label_obj} ({conf:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            save_log(name_status, f"Detected object: {label_obj}", frame)
            log_to_csv("Gadget Detected", label_obj)
            play_alarm()

    if gaze in ["Looking Down", "Looking Down (Head)", "Looking Right", "Looking Left"]:
        save_log(name_status, gaze, frame)
        log_to_csv("Gaze Cheating", gaze)
        play_alarm()

    # Display information on frame
    cv2.putText(frame, f"Status: {name_status}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.putText(frame, f"ML: {label}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 255) if "CHEATING" in label else (0, 255, 0), 2)
    cv2.putText(frame, f"Gaze: {gaze}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    if pitch is not None:
        cv2.putText(frame, f"Pitch: {pitch:.2f}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    if yaw is not None:
        cv2.putText(frame, f"Yaw: {yaw:.2f}", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(frame, f"Match Score: {match_score:.2f}%", (10, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if curr_time - prev_time > 0 else 0
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    for i, (x, y, w, h) in enumerate(faces):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
        label_display = previous_results[i][1] if i < len(previous_results) else "Unknown"
        cv2.putText(frame, label_display, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow("AI Proctoring System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()