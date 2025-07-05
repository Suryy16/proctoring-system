import asyncio
from collections import deque
import json
import shutil
import threading
import time
from fastapi import FastAPI, File, Form, Response, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
from deepface import DeepFace
from modules.gaze_tracking import get_gaze_direction
from modules.object_detection import detect_objects
from modules.utils import calculate_face_match_score, log_to_csv, play_alarm, save_log

import numpy as np

from recognition_scripts.dataset_processor import DatasetProcessor
from recognition_scripts.video_to_dataset import VideoFrameExtractor
from recognition_scripts.face_utils import FaceRecognizer
import cv2
from dotenv import load_dotenv
import os


load_dotenv()
logger = logging.getLogger(__name__)
app = FastAPI()
ROOT_DATABASE_DIR = os.getenv('ROOT_DATABASE_DIR')
DEEPF_DATABASE_DIR = os.getenv('DEEPF_DATABASE_DIR')

# Global objects
frame_processor = DatasetProcessor()
yolo_lock = threading.Lock()
last_detected_objects = []

# CORS configuration
app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

#load setup
try:
  processor = DatasetProcessor()
  recognizer = FaceRecognizer()
  extractor = VideoFrameExtractor()
  logger.info(f"Setup Import Completed")
except Exception as e:
  logger.error(f"Failed to import setup: {str(e)}")
  raise


@app.post("/register-face")
def register_face(
    personName: str = Form(...),
    video: UploadFile = File(...)
):
    try:
        # Validate inputs
        if not personName or not video:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "Video and personName are required"}
            )

        # Sanitize input
        safe_personName = "".join(c for c in personName if c.isalnum() or c in ("_", "-", " ")).rstrip().replace(" ", "_")

        # Define directory structure
        base_dir = ROOT_DATABASE_DIR
        data_dir = DEEPF_DATABASE_DIR
        raw_data_dir = os.path.join(base_dir, "raw_data", safe_personName)
        dataset_dir = os.path.join(base_dir, "dataset", safe_personName)
        processed_dataset_dir = os.path.join(base_dir, DEEPF_DATABASE_DIR, safe_personName)

        # Create directories
        os.makedirs(raw_data_dir, exist_ok=True)
        os.makedirs(dataset_dir, exist_ok=True)
        os.makedirs(processed_dataset_dir, exist_ok=True)

        # Save the uploaded video
        video_filename = "raw_face.mp4"
        video_path = os.path.join(raw_data_dir, video_filename)
        try:
            with open(video_path, "wb") as buffer:
                shutil.copyfileobj(video.file, buffer)
        except Exception as e:
            logger.error(f"Failed to save video: {str(e)}")
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "Failed to save uploaded video", "error": str(e)}
            )

        # Clear previous dataset frames
        for path in [dataset_dir, processed_dataset_dir]:
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path, exist_ok=True)

        # Extract frames from video
        try:
            frames = extractor.extract_uniform_frames(video_path, num_frames=28)
            logger.info(f"Extracted {len(frames)} frames from video")
        except Exception as e:
            logger.error(f"Frame extraction failed: {str(e)}")
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "Video processing failed", "error": str(e)}
            )

        # Save extracted frames to dataset
        saved_frames = []
        for i, frame in enumerate(frames):
            frame_filename = f"frame_{i:04d}.jpg"
            frame_path = os.path.join(dataset_dir, frame_filename)
            if cv2.imwrite(frame_path, frame):
                saved_frames.append(frame_path)

        if not saved_frames:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "No frames were successfully saved"}
            )

        # Process frames and save aligned faces to processed_dataset
        try:
            faces, processed_count = processor.process_dataset(dataset_dir, processed_dataset_dir)
            logger.info(f"Processed {processed_count} frames, found {len(faces)} faces")

            if len(faces) < 28:
                return JSONResponse(
                    status_code=400,
                    content={
                        "status": "error",
                        "message": f"Only found {len(faces)} faces (minimum 28 required). Please ensure: "
                                   "1. Your face is clearly visible\n"
                                   "2. Good lighting conditions\n"
                                   "3. No obstructions\n"
                                   "4. Front-facing camera angle",
                        "faces_found": len(faces),
                        "frames_processed": processed_count
                    }
                )
        except Exception as e:
            logger.error(f"Face detection failed: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": "Face detection processing failed", "error": str(e)}
            )

        return JSONResponse({
            "status": "success",
            "message": "Face registration completed successfully",
            "personName": personName,
            "details": {
                "video_saved": video_filename,
                "frames_extracted": len(frames),
                "frames_saved": len(saved_frames),
                "faces_detected": len(faces),
                "raw_data_dir": raw_data_dir,
                "dataset_dir": dataset_dir,
                "processed_dataset_dir": processed_dataset_dir
            }
        })

    except Exception as e:
        logger.error(f"Face registration failed: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "Face registration failed due to server error",
                "error": str(e)
            }
        )
    
@app.post("/delete-face")
async def delete_face(
    faceName: str = Form(...)
):
    try:
        # Sanitize input
        safe_personName = "".join(c for c in faceName if c.isalnum() or c in ("_", "-", " ")).rstrip().replace(" ", "_")

        # Define paths to delete
        base_dir = ROOT_DATABASE_DIR
        raw_path = os.path.join(base_dir, "raw_data", safe_personName)
        dataset_path = os.path.join(base_dir, "dataset", safe_personName)
        processed_path = os.path.join(base_dir, DEEPF_DATABASE_DIR, safe_personName)

        # Delete directories if they exist
        for path in [raw_path, dataset_path, processed_path]:
            if os.path.exists(path):
                shutil.rmtree(path)
                logger.info(f"Deleted: {path}")
            else:
                logger.warning(f"Path not found: {path}")

        return JSONResponse({
            "status": "success",
            "message": f"Deleted data for '{safe_personName}'",
            "faceName": safe_personName
        })

    except Exception as e:
        logger.error(f"Delete face failed: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "Delete face failed due to server error",
                "error": str(e)
            }
        )


# Object detection thread
def run_object_detection(frame):
    global last_detected_objects
    try:
        detections = detect_objects(frame)
        print(f"[YOLO DEBUG] Detected: {detections}")  # or use logger.info()
        with yolo_lock:
            last_detected_objects = detections
    except Exception as e:
        print(f"[YOLO ERROR] Failed: {e}")


@app.websocket("/face-recognition")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    frame_count = 0
    recognition_interval = 5
    face_reference = None
    previous_results = []
    prev_time = time.time()

    try:
        while True:
            try:
                data = await websocket.receive_bytes()
                nparr = np.frombuffer(data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if frame is None:
                    continue

                frame_count += 1
                faces = frame_processor.detect_faces(frame)
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

                gaze = get_gaze_direction(frame)

                if len(faces) == 0:
                    name_status = "Unknown"
                elif len(faces) == 1:
                    name_status = previous_results[0][1].split('(')[0].strip() if previous_results else "Unknown"
                else:
                    name_status = "Multiple Faces"
                    save_log(name_status, "Multiple Face Detected", frame)
                    log_to_csv("Multiple Face", "More than 1 face detected")
                    play_alarm()

                if frame_count % 15 == 0:
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, run_object_detection, frame.copy())

                with yolo_lock:
                    allowed_labels = ["cell phone", "laptop", "remote"]  # hanya ini yang kamu mau deteksi
                    current_objects = [obj for obj in last_detected_objects if obj[0].lower() in allowed_labels]

                alerts = []
                detected_objects = []  # to send to frontend

                for label, conf, x1, y1, x2, y2 in current_objects:
                    detected_objects.append({
                        "label": label,
                        "confidence": float(conf),
                        "bbox": [int(x1), int(y1), int(x2), int(y2)]
                    })

                    # Deteksi objek mencurigakan
                    if label.lower() in ["cell phone", "laptop", "remote"]:
                        log_message = f"Detected object: {label}"
                        save_log(name_status, log_message, frame)
                        log_to_csv("Gadget Detected", label)
                        play_alarm()

                        alerts.append({
                            "type": "object",
                            "severity": "high",
                            "label": label,
                            "message": log_message
                        })

                if gaze in ["Looking Down", "Looking Down (Head)", "Looking Right", "Looking Left"]:
                    alerts.append({"type": "gaze", "direction": gaze})
                    save_log(name_status, gaze, frame)
                    log_to_csv("Gaze Cheating", gaze)
                    play_alarm()

                curr_time = time.time()
                fps = 1 / (curr_time - prev_time) if curr_time - prev_time > 0 else 0
                prev_time = curr_time

                await websocket.send_json({
                    "status": name_status,
                    "faces": [
                        {
                            "bbox": [int(x), int(y), int(x+w), int(y+h)],
                            "label": label
                        } for ((x, y, w, h), label) in previous_results
                    ],
                    "gaze": gaze,
                    "match_score": float(match_score),
                    "fps": fps,
                    "alerts": alerts,
                    "detected_objects": detected_objects
                })

            except WebSocketDisconnect:
                break

    finally:
        await websocket.close()

# app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="info")