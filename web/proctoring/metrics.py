import streamlit as st

def update_metrics_display(placeholder):
    """Update metrics display"""
    with placeholder.container():
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