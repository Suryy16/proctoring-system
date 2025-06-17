import sys
import os
import glob
import cv2

# Pastikan modul bisa diakses
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.gaze_paper import get_eye_gaze_ratios
from modules.head_pose import estimate_head_pose
from modules.feature_logger import save_features
from modules.face_recognition import detect_faces

# Cari semua video di ./data/raw/ dengan berbagai ekstensi umum
video_paths = []
for ext in ('*.mp4', '*.mkv', '*.mov', '*.avi', '*.flv', '*.wmv', '*.webm'):
    video_paths.extend(glob.glob(f'./data/raw/dataset_{ext}'))

if not video_paths:
    print("🚫 No dataset videos found in ./data/raw/")
    sys.exit(1)

for video_path in video_paths:
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"📹 Processing: {video_path} ({total_frames} frames)")

    frame_count = 0
    success_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        hr, vr = get_eye_gaze_ratios(frame)
        pitch = yaw = None
        faces, _ = detect_faces(frame)

        if hr is None or vr is None:
            print(f"Frame {frame_count}: ❌ HR/VR not detected")
            continue
        if not faces:
            print(f"Frame {frame_count}: ❌ Face not detected")
            continue

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

        save_features(hr, vr, pitch if pitch is not None else 0, yaw if yaw is not None else 0)
        success_count += 1

        if frame_count % 30 == 0:
            print(f"✅ Processed {frame_count}/{total_frames} frames, saved {success_count}")

    cap.release()
    print(f"🎉 Finished {video_path} — total saved: {success_count}\n")
