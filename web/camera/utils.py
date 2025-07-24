import cv2
import streamlit as st
import time

def get_video_capture(camera_index):
    """Initialize camera with optimized settings for higher FPS"""
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    return cap

def test_webcam(camera_index):
    """Test the webcam and show preview"""
    cap = None
    try:
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        
        if not cap.isOpened():
            st.error("❌ Webcam not detected")
            return False
            
        st.info("🔍 Testing webcam... Say cheese!")
        preview = st.empty()
        
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
        return True
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return False
    finally:
        if cap:
            cap.release()