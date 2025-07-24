import atexit
import streamlit as st
import time

def initialize_session_state():
    """Initialize session state variables"""
    if 'recognition_active' not in st.session_state:
        st.session_state.recognition_active = False
    if 'alerts' not in st.session_state:
        st.session_state.alerts = []
    if 'alert_keys' not in st.session_state:
        st.session_state.alert_keys = set()
    if 'metrics' not in st.session_state:
        st.session_state.metrics = {
            "cheat_detection": {"normal": 0, "detected": 0},
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

def cleanup():
    """Cleanup function registered with atexit"""
    if 'cap' in st.session_state and st.session_state.cap.isOpened():
        st.session_state.cap.release()

atexit.register(cleanup)