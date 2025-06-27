import os
import streamlit as st
import requests
import json
import cv2
import numpy as np
from PIL import Image
import io
import time
import websockets
import asyncio
import pandas as pd
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Exam Proctoring System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .sidebar .sidebar-content {
        background-color: #e8f4f8;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        font-weight: bold;
    }
    .stAlert {
        border-radius: 5px;
    }
    .metric-box {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 15px;
        background-color: white;
    }
    .header-box {
        background-color: #2c3e50;
        color: white;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

ROOT_DATABASE_DIR = os.getenv('ROOT_DATABASE_DIR', 'database')
def get_registered_students():
    """Get list of registered students from the dataset directory"""
    try:
        dataset_path = os.path.join(ROOT_DATABASE_DIR, "dataset")
        if os.path.exists(dataset_path):
            students = [d for d in os.listdir(dataset_path) 
                       if os.path.isdir(os.path.join(dataset_path, d))]
            return sorted(students)  # Return sorted list for consistency
        return []
    except Exception as e:
        st.error(f"Error accessing student directory: {str(e)}")
        return []

# App title
st.markdown('<div class="header-box"><h1 style="color:white;text-align:center;">Exam Proctoring System</h1></div>', unsafe_allow_html=True)

# Initialize session state
if 'recognition_active' not in st.session_state:
    st.session_state.recognition_active = False
if 'websocket' not in st.session_state:
    st.session_state.websocket = None
if 'alerts' not in st.session_state:
    st.session_state.alerts = []
if 'metrics' not in st.session_state:
    st.session_state.metrics = {
        "face_recognition": {"success": 0, "fail": 0},
        "gaze_detection": {"normal": 0, "abnormal": 0},
        "object_detection": {"allowed": 0, "suspicious": 0}
    }
if 'exam_start_time' not in st.session_state:
    st.session_state.exam_start_time = None
if 'exam_logs' not in st.session_state:
    st.session_state.exam_logs = []

# API configuration
API_BASE_URL = "http://localhost:5000"  # Update with your API URL

# Sidebar - Exam Configuration
with st.sidebar:
    st.header("Exam Configuration")
    exam_name = st.text_input("Exam Name", "Midterm Exam")
    exam_duration = st.number_input("Duration (minutes)", min_value=1, value=60)
    allowed_objects = st.multiselect(
        "Allowed Objects",
        ["None", "Pen", "Paper", "Calculator"],
        default=["Pen", "Paper"]
    )
    
    st.markdown("---")
    st.header("System Status")
    if st.session_state.recognition_active:
        st.success("Proctoring Active")
        if st.session_state.exam_start_time:
            elapsed_time = time.time() - st.session_state.exam_start_time
            remaining_time = max(0, exam_duration * 60 - elapsed_time)
            minutes, seconds = divmod(int(remaining_time), 60)
            st.metric("Time Remaining", f"{minutes:02d}:{seconds:02d}")
    else:
        st.warning("Proctoring Inactive")

# Main Content - Tab Layout
tab1, tab2, tab3, tab4 = st.tabs(["Registration", "Proctoring", "Alerts", "Reports"])

with tab1:
    # Register Face Section
    st.header("Student Registration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.form("register_form"):
            st.subheader("Register New Student")
            person_name = st.text_input("Full Name", key="person_name")
            student_id = st.text_input("Student ID", key="student_id")
            email = st.text_input("Email", key="email")
            video_file = st.file_uploader("Upload Face Video (5-10 seconds)", type=["mp4", "mov"], key="video_file")
            register_submit = st.form_submit_button("Register Student")

        if register_submit:
            if person_name and video_file and student_id:
                try:
                    files = {"video": video_file}
                    data = {"personName": f"{student_id}_{person_name}"}
                    
                    with st.spinner("Registering face... This may take a moment..."):
                        response = requests.post(f"{API_BASE_URL}/register-face", files=files, data=data)
                        result = response.json()
                    
                    if result.get("status") == "success":
                        st.success("✅ Student registered successfully!")
                        st.balloons()
                        
                        # Log registration
                        st.session_state.exam_logs.append({
                            "timestamp": datetime.now().isoformat(),
                            "event": "registration",
                            "student_id": student_id,
                            "name": person_name,
                            "status": "success",
                            "details": result["details"]
                        })
                    else:
                        st.error("Registration failed: " + result.get("message", "Unknown error"))
                        if "faces_found" in result:
                            st.warning(f"Only found {result['faces_found']} faces (minimum 28 required)")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
            else:
                st.warning("Please provide name, student ID, and video file")
    
    with col2:
        st.subheader("Remove Student")
        
        # Always get fresh list of students
        registered_students = get_registered_students()
        
        if not registered_students:
            st.warning("No registered students found in the database")
            st.info(f"Database path: {os.path.join(ROOT_DATABASE_DIR, 'dataset')}")
        else:
            with st.form(key="delete_form", clear_on_submit=True):
                student_to_delete = st.selectbox(
                    "Select student to remove",
                    options=registered_students,
                    key="student_to_delete"
                )
                
                delete_submit = st.form_submit_button("Remove Student")
                
                if delete_submit:
                    try:
                        with st.spinner(f"Removing {student_to_delete}..."):
                            response = requests.post(
                                f"{API_BASE_URL}/delete-face",
                                data={"faceName": student_to_delete},
                                timeout=10
                            )
                            result = response.json()
                        
                        if result.get("status") == "success":
                            st.success(f"Successfully removed {student_to_delete}")
                            
                            # Update logs
                            if "exam_logs" not in st.session_state:
                                st.session_state.exam_logs = []
                            
                            st.session_state.exam_logs.append({
                                "timestamp": datetime.now().isoformat(),
                                "event": "deletion",
                                "student_id": student_to_delete.split("_")[0],
                                "name": "_".join(student_to_delete.split("_")[1:]),
                                "status": "success"
                            })
                            
                            # Force UI refresh to show updated list
                            st.rerun()
                        else:
                            st.error(f"Failed to remove student: {result.get('message', 'Unknown error')}")
                    
                    except requests.exceptions.RequestException as e:
                        st.error(f"Network error: {str(e)}")
                    except Exception as e:
                        st.error(f"Unexpected error: {str(e)}")
with tab2:
    # Real-time Proctoring Section
    st.header("Live Proctoring")
    
    # Initialize video capture
    def get_video_capture():
        return cv2.VideoCapture(0)

    if 'cap' not in st.session_state:
        st.session_state.cap = get_video_capture()

    # WebSocket connection for real-time proctoring
    async def run_proctoring():
        ws_url = f"ws://localhost:5000/face-recognition"
        
        try:
            async with websockets.connect(ws_url) as websocket:
                st.session_state.websocket = websocket
                
                # Initialize placeholders
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    video_placeholder = st.empty()
                    status_placeholder = st.empty()
                
                with col2:
                    metrics_placeholder = st.empty()
                    alerts_placeholder = st.empty()
                
                # Start exam timer
                st.session_state.exam_start_time = time.time()
                
                while st.session_state.recognition_active:
                    # Capture frame
                    ret, frame = st.session_state.cap.read()
                    if not ret:
                        st.error("Failed to capture frame")
                        break
                    
                    # Convert frame to JPEG
                    _, buffer = cv2.imencode('.jpg', frame)
                    jpg_bytes = buffer.tobytes()
                    
                    # Send frame via WebSocket
                    await websocket.send(jpg_bytes)
                    
                    try:
                        # Get results with timeout
                        result = await asyncio.wait_for(websocket.recv(), timeout=0.1)
                        data = json.loads(result)
                        
                        # Process results
                        if data.get("status"):
                            identity = data["status"]
                            if identity != "Unknown":
                                st.session_state.metrics["face_recognition"]["success"] += 1
                            else:
                                st.session_state.metrics["face_recognition"]["fail"] += 1
                        
                        if data.get("gaze"):
                            gaze = data["gaze"]
                            if gaze == "Looking Center":
                                st.session_state.metrics["gaze_detection"]["normal"] += 1
                            else:
                                st.session_state.metrics["gaze_detection"]["abnormal"] += 1
                                
                                # Log abnormal gaze
                                alert = {
                                    "timestamp": datetime.now().isoformat(),
                                    "type": "gaze",
                                    "severity": "medium",
                                    "message": f"Abnormal gaze detected: {gaze}",
                                    "frame": cv2.imencode('.jpg', frame)[1].tobytes()
                                }
                                st.session_state.alerts.append(alert)
                                st.session_state.exam_logs.append(alert)
                        
                        if data.get("alerts"):
                            for alert in data["alerts"]:
                                if alert["type"] == "object":
                                    st.session_state.metrics["object_detection"]["suspicious"] += 1
                                    
                                    # Log suspicious object
                                    alert_obj = {
                                        "timestamp": datetime.now().isoformat(),
                                        "type": "object",
                                        "severity": "high",
                                        "message": f"Suspicious object detected: {alert['label']}",
                                        "frame": cv2.imencode('.jpg', frame)[1].tobytes()
                                    }
                                    st.session_state.alerts.append(alert_obj)
                                    st.session_state.exam_logs.append(alert_obj)
                                else:
                                    st.session_state.metrics["object_detection"]["allowed"] += 1
                        
                        # Display frame with annotations
                        if data.get("faces"):
                            for face in data["faces"]:
                                x1, y1, x2, y2 = face["bbox"]
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(frame, face["label"], (x1, y1-10), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # Display frame
                        video_placeholder.image(frame, channels="BGR", use_column_width=True)
                        
                        # In the WebSocket result processing section (tab2):
                        match_score = data.get("match_score", 0)
                        # Ensure score is between 0 and 1 before converting to percentage
                        if isinstance(match_score, (int, float)):
                            if match_score > 1:  # If it's already a percentage (e.g., 0-100)
                                match_score = match_score / 100  # Convert to 0-1 range
                            match_score = max(0, min(1, match_score))  # Clamp between 0 and 1
                        else:
                            match_score = 0

                        status_text = f"""
                        **Identity Verification:** {data.get("status", "Unknown")}  
                        **Gaze Direction:** {data.get("gaze", "Unknown")}  
                        **Face Match Score:** {match_score*100:.1f}%  
                        **FPS:** {data.get("fps", 0):.1f}  
                        """
                        status_placeholder.markdown(status_text)
                        
                        # Update metrics
                        metrics_html = """
                        <div style="margin-top:20px;">
                            <h4>Proctoring Metrics</h4>
                            <div class="metric-box">
                                <h5>Face Recognition</h5>
                                <p>✅ Verified: {face_success}</p>
                                <p>❌ Unknown: {face_fail}</p>
                            </div>
                            <div class="metric-box">
                                <h5>Gaze Detection</h5>
                                <p>👀 Normal: {gaze_normal}</p>
                                <p>⚠️ Abnormal: {gaze_abnormal}</p>
                            </div>
                            <div class="metric-box">
                                <h5>Object Detection</h5>
                                <p>🟢 Allowed: {obj_allowed}</p>
                                <p>🔴 Suspicious: {obj_suspicious}</p>
                            </div>
                        </div>
                        """.format(
                            face_success=st.session_state.metrics["face_recognition"]["success"],
                            face_fail=st.session_state.metrics["face_recognition"]["fail"],
                            gaze_normal=st.session_state.metrics["gaze_detection"]["normal"],
                            gaze_abnormal=st.session_state.metrics["gaze_detection"]["abnormal"],
                            obj_allowed=st.session_state.metrics["object_detection"]["allowed"],
                            obj_suspicious=st.session_state.metrics["object_detection"]["suspicious"]
                        )
                        metrics_placeholder.markdown(metrics_html, unsafe_allow_html=True)
                        
                    except asyncio.TimeoutError:
                        # Display frame even if no results yet
                        video_placeholder.image(frame, channels="BGR", use_column_width=True)
                        continue
                    
                    # Small delay to limit FPS
                    await asyncio.sleep(0.05)
        
        except Exception as e:
            st.error(f"WebSocket error: {str(e)}")
        finally:
            st.session_state.recognition_active = False
            st.session_state.websocket = None
            if 'cap' in st.session_state:
                st.session_state.cap.release()
            st.experimental_rerun()

    # Start/stop proctoring controls
    if not st.session_state.recognition_active:
        if st.button("Start Proctoring Session", key="start_proctoring"):
            st.session_state.recognition_active = True
            st.session_state.cap = get_video_capture()
            st.session_state.alerts = []
            st.session_state.metrics = {
                "face_recognition": {"success": 0, "fail": 0},
                "gaze_detection": {"normal": 0, "abnormal": 0},
                "object_detection": {"allowed": 0, "suspicious": 0}
            }
            st.session_state.exam_start_time = time.time()
            st.session_state.exam_logs.append({
                "timestamp": datetime.now().isoformat(),
                "event": "session_start",
                "exam_name": exam_name,
                "duration_minutes": exam_duration
            })
            asyncio.run(run_proctoring())
    else:
        if st.button("Stop Proctoring Session", key="stop_proctoring"):
            st.session_state.recognition_active = False
            if st.session_state.websocket:
                asyncio.get_event_loop().run_until_complete(st.session_state.websocket.close())
            st.session_state.cap.release()
            st.session_state.exam_logs.append({
                "timestamp": datetime.now().isoformat(),
                "event": "session_end",
                "duration_seconds": time.time() - st.session_state.exam_start_time
            })
            st.experimental_rerun()

with tab3:
    # Alerts Dashboard
    st.header("Suspicious Activity Alerts")
    
    if not st.session_state.alerts:
        st.info("No alerts detected yet. Start a proctoring session to monitor for suspicious activity.")
    else:
        st.warning(f"🚨 {len(st.session_state.alerts)} suspicious events detected")
        
        for i, alert in enumerate(st.session_state.alerts[-10:][::-1]):  # Show last 10 alerts, newest first
            with st.expander(f"{i+1}. {alert['message']} - {alert['timestamp']}"):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    # Display the alert frame
                    img = cv2.imdecode(np.frombuffer(alert['frame'], cv2.IMREAD_COLOR))
                    st.image(img, channels="BGR", caption="Alert Frame")
                
                with col2:
                    # Alert details
                    st.markdown(f"""
                    **Timestamp:** {alert['timestamp']}  
                    **Type:** {alert['type'].title()}  
                    **Severity:** {alert['severity'].title()}  
                    **Message:** {alert['message']}
                    """)
                    
                    # Add action buttons
                    if st.button("Acknowledge", key=f"ack_{i}"):
                        st.session_state.alerts[i]["acknowledged"] = True
                        st.success("Alert acknowledged")
                        st.experimental_rerun()
                    
                    if st.button("Flag for Review", key=f"flag_{i}"):
                        st.session_state.alerts[i]["flagged"] = True
                        st.warning("Alert flagged for review")
                        st.experimental_rerun()

with tab4:
    # Reporting Section
    st.header("Exam Session Reports")
    
    if not st.session_state.exam_logs:
        st.info("No exam data available yet. Start a proctoring session to generate reports.")
    else:
        # Convert logs to DataFrame for display
        df_logs = pd.DataFrame(st.session_state.exam_logs)
        
        # Filter out frame data for display
        display_logs = df_logs.drop(columns=['frame'], errors='ignore')
        
        # Show summary metrics
        st.subheader("Session Summary")
        col1, col2, col3 = st.columns(3)
        
        # Calculate metrics
        total_alerts = len([log for log in st.session_state.exam_logs if log.get('type') in ['gaze', 'object']])
        gaze_alerts = len([log for log in st.session_state.exam_logs if log.get('type') == 'gaze'])
        object_alerts = len([log for log in st.session_state.exam_logs if log.get('type') == 'object'])
        
        col1.metric("Total Alerts", total_alerts)
        col2.metric("Gaze Alerts", gaze_alerts)
        col3.metric("Object Alerts", object_alerts)
        
        # Show detailed logs
        st.subheader("Detailed Logs")
        st.dataframe(display_logs)
        
        # Export options
        st.subheader("Export Report")
        
        if st.button("Generate CSV Report"):
            csv = df_logs.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"proctoring_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        if st.button("Generate JSON Report"):
            json_report = df_logs.to_json(orient='records', indent=2)
            st.download_button(
                label="Download JSON",
                data=json_report,
                file_name=f"proctoring_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

# Clean up on app exit
def cleanup():
    if 'cap' in st.session_state:
        st.session_state.cap.release()
    if st.session_state.recognition_active:
        st.session_state.recognition_active = False

import atexit
atexit.register(cleanup)