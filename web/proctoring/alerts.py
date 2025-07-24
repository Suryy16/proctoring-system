import time
import cv2
import numpy as np
import streamlit as st

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

def update_alerts_display(placeholder):
    """Update alerts display"""
    with placeholder.container():
        st.markdown("### 🚨 Active Alerts")
        if st.session_state.get('alerts'):
            for alert in reversed(st.session_state.alerts[-3:]):
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