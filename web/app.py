import streamlit as st
import asyncio
from datetime import datetime
import numpy as np
import cv2

from config import CAMERA_INDEX
from camera.utils import test_webcam
from proctoring.core import run_proctoring
from registration.views import show_registration_tab
from utils.common import initialize_session_state, stop_proctoring_session

# Initialize session state
initialize_session_state()

# Set page config
st.set_page_config(
    page_title="Exam Proctoring System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
with open("static/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Main tabs
tab1, tab2, tab3 = st.tabs(["Registration", "Proctoring", "Alerts"])

with tab1:
    show_registration_tab()

with tab2:
    st.header("Live Proctoring")
    
    if st.session_state.recognition_active:
        if st.button("🛑 Stop Proctoring"):
            stop_proctoring_session()
            st.rerun()
        
        # Display the proctoring session
        asyncio.run(run_proctoring(CAMERA_INDEX))
    else:
        if st.button("🎥 Test Webcam"):
            test_webcam(CAMERA_INDEX)
        
        if st.button("▶️ Start Proctoring", type="primary"):
            st.session_state.recognition_active = True
            st.rerun()

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
                    
                    if st.button(f"Show Technical Details", key=f"tech_details_{idx}"):
                        st.json(alert.get('data', {}))
    else:
        st.info("No alerts detected yet. The system will show suspicious activity here.")