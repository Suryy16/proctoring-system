import sys
import os
import glob
import cv2
import csv
from datetime import datetime, timedelta
import pandas as pd

# Pastikan modul bisa diakses
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.gaze_paper import get_eye_gaze_ratios
from modules.head_pose import estimate_head_pose
from modules.face_recognition import detect_faces

def save_features_with_image(hr, vr, pitch, yaw, image_filename, timestamp, label=None):
    """Save features with image filename reference for hybrid model"""
    path = '../data/hybrid_features.csv'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.exists(path)
    
    with open(path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "image_filename", "HR", "VR", "Pitch", "Yaw", "Label"])
        writer.writerow([timestamp, image_filename, hr, vr, pitch, yaw, label if label is not None else ""])

def get_label_from_elapsed_time(elapsed_seconds):
    """
    Determine label based on elapsed time from video start
    Menggunakan mapping yang sama seperti di label_and_split.py
    """
    if 0 <= elapsed_seconds < 30:
        return 0  # normal
    elif 30 <= elapsed_seconds < 45:
        return 1  # cheat left
    elif 45 <= elapsed_seconds < 60:
        return 1  # cheat right
    elif 60 <= elapsed_seconds < 75:
        return 1  # cheat down
    elif 75 <= elapsed_seconds < 90:
        return 0  # blink (normal behavior)
    elif 90 <= elapsed_seconds < 105:
        return 1  # fake cheat
    elif 105 <= elapsed_seconds < 120:
        return 0  # back to normal
    else:
        # Untuk video yang lebih panjang, ulangi pola atau set ke normal
        cycle_time = elapsed_seconds % 120
        return get_label_from_elapsed_time(cycle_time)

# Create directories for images
print("📁 Creating directories...")
os.makedirs('../data/images/train', exist_ok=True)
os.makedirs('../data/images/val', exist_ok=True)
os.makedirs('../data', exist_ok=True)

# Cari semua video di ./data/raw/ dengan berbagai ekstensi umum
video_paths = []
for ext in ('*.mp4', '*.mkv', '*.mov', '*.avi', '*.flv', '*.wmv', '*.webm'):
    video_paths.extend(glob.glob(f'../data/raw/dataset*{ext}'))

if not video_paths:
    print("🚫 No dataset videos found in ../data/raw/")
    print("Expected format: ../data/raw/dataset_*.mp4")
    sys.exit(1)

print(f"📹 Found {len(video_paths)} video(s)")

# Global counter untuk filename unik
global_frame_counter = 0

# Process each video
for video_idx, video_path in enumerate(video_paths):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"\n📹 Processing video {video_idx + 1}/{len(video_paths)}: {video_path}")
    print(f"   Total frames: {total_frames}, FPS: {fps:.2f}")

    frame_count = 0
    success_count = 0
    
    # Base timestamp untuk video ini
    video_start_time = datetime.now()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        
        # Calculate elapsed time berdasarkan frame count dan FPS
        elapsed_seconds = frame_count / fps if fps > 0 else frame_count / 30.0
        
        # Skip beberapa frame untuk mengurangi data (ambil setiap 5 frame)
        if frame_count % 5 != 0:
            continue
            
        # Extract features
        hr, vr = get_eye_gaze_ratios(frame)
        pitch = yaw = None
        faces, _ = detect_faces(frame)

        if hr is None or vr is None:
            if frame_count % 150 == 0:  # Print setiap 150 frame untuk mengurangi spam
                print(f"Frame {frame_count}: ❌ HR/VR not detected")
            continue
            
        if not faces:
            if frame_count % 150 == 0:
                print(f"Frame {frame_count}: ❌ Face not detected")
            continue

        # Get face region
        x, y, w, h = faces[0]
        
        # Ensure face region is valid
        if w < 50 or h < 50:  # Skip too small faces
            continue
            
        # Extract face crop dengan padding
        padding = 20
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame.shape[1], x + w + padding)
        y2 = min(frame.shape[0], y + h + padding)
        
        face_crop = frame[y1:y2, x1:x2]
        
        # Resize face untuk consistency
        if face_crop.shape[0] > 0 and face_crop.shape[1] > 0:
            face_resized = cv2.resize(face_crop, (224, 224))
        else:
            continue
        
        # Calculate head pose
        face_landmarks = {
            1: (x + w // 2, y + h // 2),
            152: (x + w // 2, y + h),
            33: (x, y + h // 3),
            263: (x + w, y + h // 3),
            61: (x + w // 3, y + (2 * h) // 3),
            291: (x + (2 * w) // 3, y + (2 * h) // 3),
        }
        pitch, yaw = estimate_head_pose(face_landmarks, frame.shape)
        
        # Generate timestamp untuk record ini
        current_timestamp = video_start_time + timedelta(seconds=elapsed_seconds)
        timestamp_str = current_timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        # Generate unique image filename
        image_filename = f"video{video_idx:02d}_frame_{global_frame_counter:08d}_t{elapsed_seconds:.1f}s.jpg"
        
        # Save face image
        image_path = os.path.join('../data/images/train', image_filename)
        cv2.imwrite(image_path, face_resized)
        
        # Get label berdasarkan elapsed time
        label = get_label_from_elapsed_time(elapsed_seconds)
        
        # Save features dengan image reference dan label
        save_features_with_image(
            hr=hr, 
            vr=vr, 
            pitch=pitch if pitch is not None else 0, 
            yaw=yaw if yaw is not None else 0,
            image_filename=image_filename,
            timestamp=timestamp_str,
            label=label
        )
        
        global_frame_counter += 1
        success_count += 1

        # Progress update
        if frame_count % 150 == 0:
            print(f"✅ Processed {frame_count}/{total_frames} frames, saved {success_count} samples")
            print(f"   Current time: {elapsed_seconds:.1f}s, Label: {label}")

    cap.release()
    print(f"🎉 Finished {video_path}")
    print(f"   Total saved samples: {success_count}")

print(f"\n🎊 All videos processed!")
print(f"📊 Total samples saved: {global_frame_counter}")

# Load dan display dataset info
if os.path.exists('../data/hybrid_features.csv'):
    df = pd.read_csv('./data/hybrid_features.csv')
    print(f"\n📈 Dataset Summary:")
    print(f"   Total records: {len(df)}")
    print(f"   Label distribution:")
    print(df['Label'].value_counts().sort_index())
    print(f"   Columns: {list(df.columns)}")
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"   Missing values: {missing[missing > 0].to_dict()}")
    else:
        print(f"   ✅ No missing values")
        
    print(f"\n📁 Images saved in: ./data/images/train/")
    print(f"📊 Features saved in: ./data/hybrid_features.csv")
    print(f"\n🚀 Ready for hybrid model training!")
    print("Next step: Run 'python train_hybrid.py' to train the model")

else:
    print("❌ No features were saved. Check video files and face detection.")