import os
import streamlit as st
import requests
import json
import cv2
import numpy as np
from PIL import Image
import io
import time
import atexit
import websockets
import asyncio
import pandas as pd
from datetime import datetime
import logging
import threading
from threading import Thread  # Add this import
from queue import Queue
import concurrent.futures
from functools import partial
from collections import deque  # Add this import too


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

IS_DOCKER = os.getenv('IS_DOCKER', 'false').lower() == 'true'
CAMERA_INDEX = int(os.getenv('CAMERA_INDEX', '0'))
WEBSOCKET_HOST = os.getenv('WEBSOCKET_HOST', 'localhost')
API_HOST = os.getenv('API_HOST', 'localhost')
frame_queue = Queue(maxsize=1)  # Stores the latest frame

# Set page config
st.set_page_config(
    page_title="Exam Proctoring System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #f5f5f5; }
    .sidebar .sidebar-content { background-color: #e8f4f8; }
    .stButton>button { width: 100%; border-radius: 5px; font-weight: bold; }
    .stAlert { border-radius: 5px; }
    .metric-box { border: 1px solid #ddd; border-radius: 5px; padding: 15px; margin-bottom: 15px; background-color: white; }
    .header-box { background-color: #2c3e50; color: white; padding: 10px; border-radius: 5px; margin-bottom: 15px; }
    </style>
""", unsafe_allow_html=True)

ROOT_DATABASE_DIR = os.getenv('ROOT_DATABASE_DIR', 'database')

def test_camera_access():
    try:
        if IS_DOCKER:
            # Check if video devices exist in Docker
            if not os.path.exists('/dev/video0'):
                st.error("No camera devices found in /dev/")
                return False
        
        # Try opening a camera temporarily
        cap = cv2.VideoCapture(CAMERA_INDEX)
        if cap.isOpened():
            cap.release()
            return True
        return False
    except Exception as e:
        st.error(f"Camera test failed: {str(e)}")
        return False

def show_camera_preview():
    """Display camera preview in main thread"""
    preview_placeholder = st.empty()
    cap = get_video_capture()
    
    if not cap or not cap.isOpened():
        st.error("Camera not available")
        return False

    try:
        start_time = time.time()
        while (time.time() - start_time) < 5:  # Show for 5 seconds
            ret, frame = cap.read()
            if ret:
                preview_placeholder.image(frame, channels="BGR", use_container_width=True)
            time.sleep(0.05)
        return True
    finally:
        cap.release()

def start_proctoring_session():
    """Handles the complete proctoring startup sequence"""
    status = st.empty()
    
    # Step 1: Initialize camera
    status.info("🔌 Initializing camera...")
    cap = get_video_capture()
    if not cap:
        status.error("❌ Camera initialization failed")
        return False

    # Step 2: Verify camera stream
    status.info("📷 Verifying camera stream...")
    test_frames = 0
    for _ in range(10):  # Try 10 frames
        ret, frame = cap.read()
        if ret:
            test_frames += 1
            # Quick preview
            st.image(frame, channels="BGR", use_container_width=True)
            time.sleep(0.1)
    
    if test_frames < 3:
        status.error("⚠️ Camera stream unstable")
        cap.release()
        return False

    # Step 3: Start proctoring session
    try:
        status.info("🚀 Starting proctoring session...")
        st.session_state.cap = cap
        st.session_state.recognition_active = True
        st.session_state.exam_start_time = time.time()
        
        # Start keepalive thread
        threading.Thread(
            target=camera_keepalive,
            args=(cap,),
            daemon=True
        ).start()
        
        # Start processing
        asyncio.run(run_proctoring())
        return True
        
    except Exception as e:
        status.error(f"❌ Failed to start: {str(e)}")
        if cap.isOpened():
            cap.release()
        return False

def get_video_capture():
    """Initialize camera with optimized settings for higher FPS"""
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)  # Use DirectShow for Windows
    
    if cap.isOpened():
        # Reduce resolution for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Reduced from higher resolutions
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Set MJPG codec which is typically faster
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        
        # Request higher FPS (may not be supported by all cameras)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Reduce buffer size to minimize latency
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Disable auto-focus and other auto-settings that cause delays
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # 1 = manual
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)  # Disable auto white balance
    return cap

def camera_thread(stop_event):
    """Thread for continuous frame capture"""
    cap = get_video_capture()
    if not cap or not cap.isOpened():
        logger.error("Failed to initialize camera")
        return

    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if ret:
                if frame_queue.full():
                    frame_queue.get_nowait()
                frame_queue.put(frame)
            time.sleep(0.033)  # ~30 FPS
    finally:
        cap.release()
        logger.info("Camera thread stopped")

def show_webcam_troubleshooting():
    """Help users test and fix webcam issues"""
    with st.expander("🛠️ Webcam Troubleshooting", expanded=True):
        st.markdown("""
        ### Fixes for Built-in Webcam Issues:
        1. **Close other apps** using the webcam (Zoom, Skype, etc.)
        2. **Restart your laptop** (resets camera drivers)
        3. **Update webcam drivers** (Device Manager → Cameras)
        """)
        
        if st.button("🎥 Test Webcam Now"):
            test_webcam()

def test_webcam():
    """Test the webcam and show preview"""
    cap = None
    try:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Force DirectShow
        
        if not cap.isOpened():
            st.error("❌ Webcam not detected")
            return
            
        st.info("🔍 Testing webcam... Say cheese!")
        preview = st.empty()
        
        # Show 5-sec preview
        start_time = time.time()
        while (time.time() - start_time) < 5:
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (640, 480))
                preview.image(frame, channels="BGR", use_container_width=True)
            else:
                st.warning("⚠️ Webcam disconnected during test")
                break
            time.sleep(0.05)
        
        st.success("✅ Webcam works! Proceed to proctoring")
    except Exception as e:
        st.error(f"Error: {str(e)}")
    finally:
        if cap:
            cap.release()

def process_webcam_frames(cap, stop_event):
    """Process frames in a thread-safe way"""
    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                logger.warning("Frame capture failed")
                time.sleep(0.1)
                continue
            
            # Process frame (e.g., face detection)
            processed_frame = process_frame(frame)
            
            # Update the latest frame in session state
            st.session_state.latest_frame = processed_frame
            time.sleep(0.033)  # ~30 FPS
            
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
    finally:
        cap.release()
        stop_event.set()

def process_frame(frame):
    """Example frame processing function"""
    # Convert to grayscale for demonstration
    processed = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
    return processed

def display_camera_feed():
    """Display the live camera feed"""
    video_placeholder = st.empty()
    stop_event = threading.Event()
    
    # Start camera thread
    threading.Thread(
        target=camera_thread,
        args=(stop_event,),
        daemon=True
    ).start()

    try:
        while st.session_state.recognition_active:
            if not frame_queue.empty():
                frame = frame_queue.get_nowait()
                video_placeholder.image(frame, channels="BGR", use_container_width=True)
            time.sleep(0.05)
    finally:
        stop_event.set()

def camera_keepalive(cap):
    while hasattr(st.session_state, 'recognition_active') and st.session_state.recognition_active:
        if cap and cap.isOpened():
            ret, _ = cap.read()
            if not ret:
                print("Webcam stream disconnected, reconnecting...")
                cap.release()
                st.session_state.cap = get_video_capture()
                cap = st.session_state.cap
        time.sleep(0.5)

def verify_camera_access():
    try:
        if IS_DOCKER:
            # Check if video devices exist in container
            if not any(os.path.exists(f'/dev/video{i}') for i in [0,1]):
                st.error("No camera devices found in /dev/")
                return False
        
        # Test actual capture
        test_cap = get_video_capture()
        if test_cap is None:
            return False
        if test_cap.isOpened():
            test_cap.release()
            return True
        return False
    except Exception as e:
        st.error(f"Camera verification failed: {str(e)}")
        return False
    
class VideoCaptureManager:
    def __init__(self, camera_index):
        self.camera_index = camera_index
        self.cap = None
        self.max_attempts = 5
        self.attempt_delay = 1  # seconds
        
    def __enter__(self):
        attempts = 0
        while attempts < self.max_attempts:
            try:
                if IS_DOCKER:
                    # In Docker, try all possible camera indices
                    for i in range(3):
                        self.cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
                        if self.cap.isOpened():
                            st.success(f"Successfully opened camera at index {i}")
                            return self.cap
                else:
                    # Local development, use specified index
                    self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
                    if self.cap.isOpened():
                        st.success(f"Successfully opened camera at index {self.camera_index}")
                        return self.cap
                
                attempts += 1
                time.sleep(self.attempt_delay)
            except Exception as e:
                st.warning(f"Camera attempt {attempts} failed: {str(e)}")
                attempts += 1
                time.sleep(self.attempt_delay)
        
        st.error("Failed to initialize camera after multiple attempts")
        return None
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cap and self.cap.isOpened():
            self.cap.release()
            st.session_state.cap = None

def get_registered_students():
    try:
        dataset_path = os.path.join(ROOT_DATABASE_DIR, "dataset")
        if os.path.exists(dataset_path):
            return sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
        return []
    except Exception as e:
        st.error(f"Error accessing student directory: {str(e)}")
        return []

# Initialize session state
# Session state initialization
if 'recognition_active' not in st.session_state:
    st.session_state.recognition_active = False
if 'alerts' not in st.session_state:
    st.session_state.alerts = []
if 'alert_keys' not in st.session_state:
    st.session_state.alert_keys = set()
if 'metrics' not in st.session_state:
    st.session_state.metrics = {
        "cheat_detection": {"normal": 0, "detected": 0},  # Changed from face_recognition
        "gaze_detection": {"normal": 0, "abnormal": 0},
        "object_detection": {"allowed": 0, "suspicious": 0}
    }
if 'cap' not in st.session_state:
    st.session_state.cap = None
if 'websocket' not in st.session_state:
    st.session_state.websocket = None
if 'exam_start_time' not in st.session_state:
    st.session_state.exam_start_time = None
if 'exam_logs' not in st.session_state:
    st.session_state.exam_logs = []

API_BASE_URL = f"http://{API_HOST}:5000"

# Main tabs
tab1, tab2, tab3 = st.tabs(["Registration", "Proctoring", "Alerts"])

def stop_proctoring_session():
    """Properly stop the proctoring session"""
    if 'stop_event' in st.session_state:
        st.session_state.stop_event.set()
    
    if 'cap' in st.session_state and st.session_state.cap:
        st.session_state.cap.release()
        del st.session_state.cap
    
    if 'latest_frame' in st.session_state:
        del st.session_state.latest_frame
    
    st.session_state.recognition_active = False
    time.sleep(0.5)  # Allow threads to clean up

# [Previous imports remain the same...]

# Add these new imports if not already present
import queue
from collections import deque

# Initialize session state with optimized settings
if 'metrics' not in st.session_state:
    st.session_state.metrics = {
        "cheat_detection": {"normal": 0, "detected": 0},
        "gaze_detection": {"normal": 0, "abnormal": 0},
        "object_detection": {"allowed": 0, "suspicious": 0}
    }

class CameraStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.stream.set(cv2.CAP_PROP_FPS, 30)
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        self.Q = queue.Queue(maxsize=32)  # Smaller queue for lower latency

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while not self.stopped:
            if not self.Q.full():
                (grabbed, frame) = self.stream.read()
                if grabbed:
                    self.Q.put(frame)
            else:
                time.sleep(0.01)  # Prevent busy waiting

    def read(self):
        return self.Q.get()

    def stop(self):
        self.stopped = True
        self.stream.release()

async def run_proctoring():
    """Optimized proctoring function with better FPS handling"""
    ws_url = f"ws://{WEBSOCKET_HOST}:5000/face-recognition"
    
    # Initialize camera stream
    cam_stream = CameraStream(CAMERA_INDEX).start()
    st.session_state.camera_stream = cam_stream
    
    try:
        # Create layout
        col1, col2 = st.columns([2, 1])
        with col1:
            video_placeholder = st.empty()
            status_display = st.empty()
        with col2:
            metrics_placeholder = st.empty()
            alerts_placeholder = st.empty()

        # Performance tracking
        frame_times = deque(maxlen=30)
        last_frame_time = time.time()
        
        async with websockets.connect(ws_url) as websocket:
            st.session_state.websocket = websocket
            
            while st.session_state.recognition_active:
                start_time = time.time()
                
                # Get frame from camera thread
                frame = cam_stream.read()
                
                # Calculate FPS
                frame_times.append(time.time())
                fps = len(frame_times) / (frame_times[-1] - frame_times[0]) if len(frame_times) > 1 else 0
                
                # Process at reduced resolution
                small_frame = cv2.resize(frame, (320, 240))
                _, buffer = cv2.imencode('.jpg', small_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                
                try:
                    # Send frame with timeout
                    await asyncio.wait_for(websocket.send(buffer.tobytes()), timeout=0.1)
                    
                    # Get results with timeout
                    data = await asyncio.wait_for(websocket.recv(), timeout=0.5)
                    data = json.loads(data)
                    
                    # Process status and alerts
                    status_info = process_status_data(data, fps)
                    
                    # Handle cheating detection
                    if "CHEATING" in status_info['ml_prediction']:
                        handle_cheating_alert(buffer, status_info, data)
                    
                    # Process other alerts
                    if data.get('alerts'):
                        process_alerts(data, buffer, status_info)
                    
                    # Update display
                    update_display(
                        frame, 
                        data, 
                        status_info, 
                        video_placeholder, 
                        status_display,
                        metrics_placeholder,
                        alerts_placeholder
                    )
                    
                except asyncio.TimeoutError:
                    # Display frame even if processing timed out
                    video_placeholder.image(frame, channels="BGR")
                    continue
                    
                except Exception as e:
                    st.error(f"Processing error: {str(e)}")
                    break
                
                # Control frame rate
                elapsed = time.time() - start_time
                if elapsed < 0.033:  # ~30fps
                    await asyncio.sleep(0.033 - elapsed)
                    
    except Exception as e:
        st.error(f"Proctoring error: {str(e)}")
    finally:
        if hasattr(st.session_state, 'camera_stream'):
            st.session_state.camera_stream.stop()
        if st.session_state.get('websocket'):
            await st.session_state.websocket.close()
        st.session_state.recognition_active = False
        st.rerun()

def process_status_data(data, fps):
    """Extract and format status information"""
    return {
        'status': data.get("status", "Unknown"),
        'ml_prediction': data.get("ml_prediction", "N/A"),
        'gaze': data.get("gaze", "Unknown"),
        'match_score': float(data.get("match_score", 0)),
        'fps': fps,
        'timestamp': datetime.now().isoformat()
    }

def handle_cheating_alert(buffer, status_info, data):
    """Handle cheating detection from ML model"""
    alert_key = f"cheating_{status_info['ml_prediction']}_{time.time()}"
    
    if alert_key not in st.session_state.get('alert_keys', set()):
        alert_entry = {
            **status_info,
            'type': 'cheating',
            'message': f"Cheating detected: {status_info['ml_prediction']}",
            'severity': 'high',
            'frame': buffer.tobytes(),
            'data': data
        }
        
        # Initialize if needed
        if 'alerts' not in st.session_state:
            st.session_state.alerts = []
        if 'alert_keys' not in st.session_state:
            st.session_state.alert_keys = set()
        
        st.session_state.alerts.append(alert_entry)
        st.session_state.alert_keys.add(alert_key)
        st.session_state.metrics['cheat_detection']['detected'] += 1

def process_alerts(data, buffer, status_info):
    """Process other alerts (gaze, objects)"""
    for alert in data.get('alerts', []):
        alert_key = f"{alert.get('type')}_{alert.get('message')}_{time.time()}"
        
        if alert_key not in st.session_state.get('alert_keys', set()):
            alert_entry = {
                **status_info,
                'type': alert.get('type', 'unknown'),
                'message': alert.get('message', 'No details'),
                'severity': alert.get('severity', 'medium'),
                'frame': buffer.tobytes(),
                'data': data
            }
            
            # Add direction/label if available
            if 'direction' in alert:
                alert_entry['direction'] = alert['direction']
            if 'label' in alert:
                alert_entry['label'] = alert['label']
            
            st.session_state.alerts.append(alert_entry)
            st.session_state.alert_keys.add(alert_key)
            
            # Update metrics
            alert_type = alert_entry['type']
            if alert_type == 'object':
                st.session_state.metrics['object_detection']['suspicious'] += 1
            elif alert_type == 'gaze':
                st.session_state.metrics['gaze_detection']['abnormal'] += 1

def update_display(frame, data, status_info, video_placeholder, status_display, metrics_placeholder, alerts_placeholder):
    """Update all display elements efficiently"""
    # Draw detections on frame
    processed_frame = frame.copy()
    
    # Draw faces
    for face in data.get("faces", []):
        x1, y1, x2, y2 = face["bbox"]
        cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(processed_frame, face["label"], (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Draw objects
    for obj in data.get("detected_objects", []):
        x1, y1, x2, y2 = obj["bbox"]
        cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(processed_frame, f"{obj['label']}", (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Update video display
    video_placeholder.image(processed_frame, channels="BGR")
    
    # Update status
    status_text = f"""
    **Status:** {status_info['status']}  
    **ML:** {status_info['ml_prediction']}  
    **Gaze:** {status_info['gaze']}  
    **Match:** {status_info['match_score']:.1f}%  
    **FPS:** {status_info['fps']:.1f}
    """
    status_display.markdown(status_text)
    
    # Update metrics
    with metrics_placeholder.container():
        st.markdown("### 📊 Live Metrics")
        cols = st.columns(3)
        with cols[0]:
            st.metric("Cheat Det", 
                     f"{st.session_state.metrics['cheat_detection']['normal']} ✔️", 
                     f"{st.session_state.metrics['cheat_detection']['detected']} ✖️")
        with cols[1]:
            st.metric("Gaze", 
                     f"{st.session_state.metrics['gaze_detection']['normal']} 👀", 
                     f"{st.session_state.metrics['gaze_detection']['abnormal']} ⚠️")
        with cols[2]:
            st.metric("Objects", 
                     f"{st.session_state.metrics['object_detection']['allowed']} ✔️", 
                     f"{st.session_state.metrics['object_detection']['suspicious']} ✖️")
    
    # Update alerts
    with alerts_placeholder.container():
        st.markdown("### 🚨 Active Alerts")
        if st.session_state.get('alerts'):
            for alert in reversed(st.session_state.alerts[-3:]):  # Show last 3 alerts
                with st.expander(f"{alert['type'].upper()} - {alert['timestamp']}"):
                    cols = st.columns([1, 3])
                    with cols[0]:
                        try:
                            img = cv2.imdecode(np.frombuffer(alert['frame'], np.uint8), cv2.IMREAD_COLOR)
                            if img is not None:
                                st.image(img, channels="BGR", width=150)
                        except:
                            st.warning("Couldn't display image")
                    with cols[1]:
                        st.write(f"**Type:** {alert['type'].upper()}")
                        st.write(f"**Message:** {alert['message']}")
                        st.write(f"**Time:** {alert['timestamp']}")
        else:
            st.info("No alerts detected")

# Tab 1: Registration (unchanged)
with tab1:
    st.header("Student Registration")
    col1, col2 = st.columns(2)
    
    with col1:
        with st.form("register_form"):
            st.subheader("Register New Student")
            person_name = st.text_input("Full Name")
            student_id = st.text_input("Student ID")
            video_file = st.file_uploader("Upload Face Video", type=["mp4", "mov"])
            if st.form_submit_button("Register"):
                if person_name and video_file and student_id:
                    try:
                        files = {"video": video_file}
                        data = {"personName": f"{student_id}_{person_name}"}
                        with st.spinner("Registering..."):
                            response = requests.post(f"{API_BASE_URL}/register-face", files=files, data=data)
                            if response.json().get("status") == "success":
                                st.success("✅ Student registered!")
                            else:
                                st.error("Registration failed")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
    
    with col2:
        st.subheader("Remove Student")
        registered_students = get_registered_students()
        if registered_students:
            with st.form("delete_form"):
                student = st.selectbox("Select student", registered_students)
                if st.form_submit_button("Remove"):
                    try:
                        response = requests.post(f"{API_BASE_URL}/delete-face", data={"faceName": student})
                        if response.json().get("status") == "success":
                            st.success(f"Removed {student}")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")


# Tab 2: Proctoring
with tab2:
    st.header("Live Proctoring")
    
    if st.session_state.recognition_active:
        if st.button("🛑 Stop Proctoring"):
            stop_proctoring_session()
            st.rerun()
        
        # Display the proctoring session
        asyncio.run(run_proctoring())
    else:
        if st.button("🎥 Test Webcam"):
            if show_camera_preview():
                st.success("Camera working properly!")
        
        if st.button("▶️ Start Proctoring", type="primary"):
            st.session_state.recognition_active = True
            st.rerun()

# Tab 3: Alerts (unchanged)
with tab3:
    st.header("Suspicious Activity Alerts")
    
    # Display metrics summary
    with st.expander("📊 Detection Summary", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Cheat Detection", 
                     f"{st.session_state.metrics['cheat_detection']['normal']} ✅", 
                     f"{st.session_state.metrics['cheat_detection']['detected']} ❌")
        with col2:
            st.metric("Gaze Detection", 
                     f"{st.session_state.metrics['gaze_detection']['normal']} 👀", 
                     f"{st.session_state.metrics['gaze_detection']['abnormal']} ⚠️")
        with col3:
            st.metric("Object Detection", 
                     f"{st.session_state.metrics['object_detection']['allowed']} ✔️", 
                     f"{st.session_state.metrics['object_detection']['suspicious']} ✖️")
    
    # Display detailed alerts
    st.subheader("📜 Alert History")
    if st.session_state.alerts:
        # Summary count by type
        alert_counts = {}
        for alert in st.session_state.alerts:
            alert_type = alert.get('type', 'unknown')
            alert_counts[alert_type] = alert_counts.get(alert_type, 0) + 1
        
        st.write("### Alert Statistics")
        cols = st.columns(len(alert_counts))
        for idx, (alert_type, count) in enumerate(alert_counts.items()):
            cols[idx].metric(f"{alert_type.title()} Alerts", count)
        
        # Detailed alert list
        st.write("### Detailed Alerts")
        for idx, alert in enumerate(reversed(st.session_state.alerts)):
            with st.expander(f"⚠️ {alert.get('type', 'Alert').upper()} - {alert.get('timestamp')}", expanded=False):
                cols = st.columns([1, 3])
                with cols[0]:
                    if 'frame' in alert and alert['frame']:
                        try:
                            # Properly decode the image from bytes
                            img_array = np.frombuffer(alert['frame'], np.uint8)
                            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                            if img is not None and len(img.shape) in (2, 3):
                                st.image(img, channels="BGR", caption="Alert snapshot")
                            else:
                                st.warning("Invalid image data")
                        except Exception as e:
                            st.warning(f"Couldn't display image: {str(e)}")
                with cols[1]:
                    st.write(f"**Type:** {alert.get('type', 'Unknown')}")
                    st.write(f"**Severity:** {alert.get('severity', 'medium')}")
                    st.write(f"**Details:** {alert.get('message', 'No details')}")
                    if 'direction' in alert:
                        st.write(f"**Direction:** {alert['direction']}")
                    if 'label' in alert:
                        st.write(f"**Object:** {alert['label']}")
                    
                    # Show technical details with unique index-based key
                    if st.button(f"Show Technical Details", key=f"tech_details_{idx}"):
                        st.json(alert.get('data', {}))
    else:
        st.info("No alerts detected yet. The system will show suspicious activity here.")
    

def cleanup():
    if 'cap' in st.session_state and st.session_state.cap.isOpened():
        st.session_state.cap.release()

atexit.register(cleanup)