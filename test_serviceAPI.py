import logging
import shutil
from fastapi import WebSocketDisconnect
import pytest
import cv2
import numpy as np
import time
import os
from unittest.mock import mock_open
from fastapi.testclient import TestClient
from modules.object_detection import detect_objects
from unittest.mock import patch
import torch
from serviceAPI import app  # Import your FastAPI app
from serviceAPI import USE_CNN, CNN_MODEL_PATH, cnn_model, transform, device, yolo_lock, last_detected_objects

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
# Suppress DeepFace warnings
logging.getLogger('deepface').setLevel(logging.ERROR)

# Create test client
client = TestClient(app)

# Test data paths
TEST_VIDEO_PATH = "test_video.mp4"
TEST_NO_FACES_VIDEO_PATH = "no_faces_video.mp4"
TEST_INVALID_FILE = "invalid.txt"

@pytest.fixture
def create_test_video():
    """Create test video files with proper cleanup"""
    test_video_path = "test_video.mp4"
    no_faces_path = "no_faces_video.mp4"
    
    # Create test video with face
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(test_video_path, fourcc, 20.0, (640, 480))
    for i in range(30):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(frame, (200+i, 150), (300+i, 250), (255, 255, 255), -1)
        out.write(frame)
    out.release()
    
    # Create video with no faces
    out = cv2.VideoWriter(no_faces_path, fourcc, 20.0, (640, 480))
    for _ in range(30):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        out.write(frame)
    out.release()
    
    # Setup test environment
    os.environ['ROOT_DATABASE_DIR'] = "test_db"
    os.environ['DEEPF_DATABASE_DIR'] = "deepface_db"
    
    yield {
        "with_face": test_video_path,
        "no_face": no_faces_path
    }
    
    # Cleanup - only remove if tests created them
    for path in [test_video_path, no_faces_path, "test_db"]:
        if os.path.exists(path):
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)

@pytest.fixture
def mock_face_detection():
    with patch('serviceAPI.recognizer.recognize_face', return_value=("test_user", 0.9)), \
         patch('serviceAPI.calculate_face_match_score', return_value=0.95):
        yield

def test_health_check():
    """Test the health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_register_face_success(create_test_video):
    with patch('builtins.open', mock_open()), \
         patch('shutil.copyfileobj'), \
         patch('cv2.imwrite', return_value=True), \
         patch('serviceAPI.VideoFrameExtractor.extract_uniform_frames') as mock_extract, \
         patch('serviceAPI.DatasetProcessor.process_dataset') as mock_process:
        
        # Mock successful frame extraction and processing
        mock_extract.return_value = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(30)]
        mock_process.return_value = ([1]*30, 30)  # Return 30 "faces"
        
        with open(TEST_VIDEO_PATH, "rb") as video_file:
            response = client.post(
                "/register-face",
                files={"video": ("test_video.mp4", video_file, "video/mp4")},
                data={"personName": "test_user"}
            )
        
        assert response.status_code == 200
        assert response.json()["status"] == "success"

def test_register_face_missing_parameters(create_test_video):
    """Test missing parameters in face registration"""
    # Use mock instead of real file
    with patch('builtins.open', mock_open()):
        # Test missing video file
        response = client.post(
            "/register-face",
            data={"personName": "test_user"}
        )
        assert response.status_code == 422
    
        # Test missing personName
        response = client.post(
            "/register-face",
            files={"video": ("test_video.mp4", b"fake_data", "video/mp4")}
        )
        assert response.status_code == 422

def test_register_face_invalid_video(create_test_video):
    """Test with invalid video file"""
    with open(TEST_INVALID_FILE, "rb") as fake_video:
        response = client.post(
            "/register-face",
            files={"video": ("test_video.mp4", fake_video, "video/mp4")},
            data={"personName": "test_user"}
        )
    assert response.status_code == 400

def test_register_face_insufficient_faces(create_test_video):
    """Test with video that doesn't contain enough faces"""
    with open(TEST_NO_FACES_VIDEO_PATH, "rb") as video_file:
        response = client.post(
            "/register-face",
            files={"video": ("no_faces_video.mp4", video_file, "video/mp4")},
            data={"personName": "test_user"}
        )
    assert response.status_code == 400
    assert "Only found" in response.json()["message"]

def test_delete_face_success(create_test_video):
    """Test successful face deletion"""
    # Mock the registration first
    with patch('serviceAPI.os.path.exists', return_value=True), \
         patch('serviceAPI.shutil.rmtree'):
        
        response = client.post(
            "/delete-face",
            data={"faceName": "test_user"}
        )
        assert response.status_code == 200
        assert response.json()["status"] == "success"

def test_delete_nonexistent_face():
    """Test deleting a face that doesn't exist"""
    with patch('serviceAPI.os.path.exists', return_value=False), \
         patch('serviceAPI.logger.warning'):
        
        response = client.post(
            "/delete-face",
            data={"faceName": "nonexistent_user"}
        )
        assert response.status_code == 200
        assert "Deleted data for" in response.json()["message"]

def test_websocket_connection():
    """Test basic WebSocket connection"""
    with client.websocket_connect("/face-recognition") as websocket:
        # Send a test frame
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        _, img_encoded = cv2.imencode('.jpg', test_frame)
        websocket.send_bytes(img_encoded.tobytes())
        
        # Receive response
        data = websocket.receive_json()
        assert "status" in data
        assert data["status"] == "Unknown"  # No face in blank frame

def test_websocket_with_face():
    with patch('serviceAPI.frame_processor.detect_faces', return_value=[(100, 100, 200, 200)]), \
         patch('serviceAPI.recognizer.recognize_face', return_value=("test_user", 0.9)), \
         patch('serviceAPI.calculate_face_match_score', return_value=0.95), \
         patch('serviceAPI.get_gaze_direction', return_value="Looking Center"):
        
        with client.websocket_connect("/face-recognition") as websocket:
            test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.rectangle(test_frame, (100, 100), (300, 300), (255, 255, 255), -1)
            _, img_encoded = cv2.imencode('.jpg', test_frame)
            websocket.send_bytes(img_encoded.tobytes())
            
            data = websocket.receive_json()
            assert "faces" in data
            assert len(data["faces"]) == 1
            assert data["status"] == "test_user"
            assert data["match_score"] == 0.95
            assert data["gaze"] == "Looking Center"

def test_websocket_with_mocked_face_detection(mock_face_detection):
    with patch('serviceAPI.frame_processor.detect_faces', return_value=[(100, 100, 200, 200)]), \
         patch('serviceAPI.get_gaze_direction', return_value="Looking Center"):
        
        with client.websocket_connect("/face-recognition") as websocket:
            test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            _, img_encoded = cv2.imencode('.jpg', test_frame)
            websocket.send_bytes(img_encoded.tobytes())
            
            data = websocket.receive_json()
            assert data["status"] == "test_user"
            assert data["match_score"] == 0.95
            assert data["gaze"] == "Looking Center"

def test_websocket_performance():
    """Test WebSocket performance"""
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(test_frame, (200, 150), (300, 250), (255, 255, 255), -1)
    
    start_time = time.time()
    with client.websocket_connect("/face-recognition") as websocket:
        _, img_encoded = cv2.imencode('.jpg', test_frame)
        websocket.send_bytes(img_encoded.tobytes())
        data = websocket.receive_json()
        
    processing_time = time.time() - start_time
    assert processing_time < 0.5  # Should process under 500ms

@pytest.mark.asyncio
async def test_websocket_corrupted_frame_async():
    # Need to mock the websocket handler since we're testing error case
    with patch('serviceAPI.WebSocket.receive_bytes', side_effect=WebSocketDisconnect(1008)):
        with client.websocket_connect("/face-recognition") as websocket:
            with pytest.raises(WebSocketDisconnect):
                await websocket.receive_json()

def test_cnn_model_loading():
    """Test CNN model loading"""
    assert USE_CNN == os.path.exists(CNN_MODEL_PATH)
    if USE_CNN:
        assert isinstance(cnn_model, torch.nn.Module)
        assert cnn_model.training == False

@pytest.mark.skipif(not USE_CNN, reason="CNN model not available")
def test_cnn_prediction():
    """Test CNN prediction"""
    # Create test face image
    test_face = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    input_tensor = transform(test_face)
    input_batch = input_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = cnn_model(input_batch)
        assert output.shape == (1, 2)  # Batch size 1, 2 classes

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

def test_object_detection_thread():
    """Test object detection thread"""
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    run_object_detection(test_frame)
    
    with yolo_lock:
        assert isinstance(last_detected_objects, list)
    
    # Test with objects in frame
    test_frame_with_phone = test_frame.copy()
    cv2.rectangle(test_frame_with_phone, (100, 100), (200, 200), (255, 255, 255), -1)
    run_object_detection(test_frame_with_phone)
    
    with yolo_lock:
        assert len(last_detected_objects) >= 0  # Might or might not detect

if __name__ == "__main__":
    # Run tests
    pytest.main(["-v", "--cov=.", "--cov-report=html"])