import asyncio
import json
import time
from datetime import datetime
from collections import deque
import cv2
import numpy as np
import websockets
import streamlit as st

from camera.manager import CameraStream
from proctoring.alerts import handle_cheating_alert, process_alerts, update_alerts_display
from proctoring.metrics import update_metrics_display
from config import WEBSOCKET_HOST

async def run_proctoring(camera_index):
    """Optimized proctoring function with better FPS handling"""
    ws_url = f"ws://{WEBSOCKET_HOST}:5000/face-recognition"
    
    # Initialize camera stream
    cam_stream = CameraStream(camera_index).start()
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
        'match_score': float(data.get("match_score", 0)) * 100,
        'fps': fps,
        'timestamp': datetime.now().isoformat()
    }

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
    
    # Update metrics and alerts
    update_metrics_display(metrics_placeholder)
    update_alerts_display(alerts_placeholder)