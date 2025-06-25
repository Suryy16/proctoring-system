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

# Set page config
st.set_page_config(page_title="Face Recognition System", layout="wide")

# App title
st.title("Face Recognition System")

# Initialize session state
if 'recognition_active' not in st.session_state:
    st.session_state.recognition_active = False
if 'websocket' not in st.session_state:
    st.session_state.websocket = None

# API configuration
API_BASE_URL = "http://localhost:5000"  # Update with your API URL

# Register Face Section
st.header("Register New Face")
with st.form("register_form"):
    person_name = st.text_input("Your Name", key="person_name")
    video_file = st.file_uploader("Upload Video (5-10 seconds)", type=["mp4", "mov"], key="video_file")
    register_submit = st.form_submit_button("Register Face")

if register_submit:
    if person_name and video_file:
        try:
            files = {"video": video_file}
            data = {"personName": person_name}
            
            response = requests.post(f"{API_BASE_URL}/register-face", files=files, data=data)
            result = response.json()
            
            if result.get("status") == "success":
                st.success(result["message"])
                st.json(result["details"])
            else:
                st.error(result.get("message", "Registration failed"))
                if "faces_found" in result:
                    st.warning(f"Only found {result['faces_found']} faces (minimum 28 required)")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.warning("Please provide both name and video file")

# Delete Face Section
st.header("Delete Registered Face")
with st.form("delete_form"):
    face_to_delete = st.text_input("Name to Delete", key="face_to_delete")
    delete_submit = st.form_submit_button("Delete Face")

if delete_submit and face_to_delete:
    try:
        response = requests.post(
            f"{API_BASE_URL}/delete-face",
            data={"faceName": face_to_delete}
        )
        result = response.json()
        
        if result.get("status") == "success":
            st.success(result["message"])
        else:
            st.error(result.get("message", "Deletion failed"))
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Real-time Recognition Section
st.header("Real-time Face Recognition")

# Initialize video capture
def get_video_capture():
    return cv2.VideoCapture(0)

if 'cap' not in st.session_state:
    st.session_state.cap = get_video_capture()

# WebSocket connection for real-time recognition
async def recognize_faces():
    user_email = st.session_state.get("user_email", "anonymous@example.com")
    ws_url = f"ws://localhost:5000/face-recognition"  # Update with your WebSocket URL
    
    try:
        async with websockets.connect(ws_url) as websocket:
            st.session_state.websocket = websocket
            
            # Send initialization message
            await websocket.send(json.dumps({
                "type": "init",
                "email": user_email
            }))
            
            # Initialize image placeholder
            image_placeholder = st.empty()
            results_placeholder = st.empty()
            
            while st.session_state.recognition_active:
                # Capture frame-by-frame
                ret, frame = st.session_state.cap.read()
                if not ret:
                    st.error("Failed to capture frame")
                    break
                
                # Convert frame to JPEG
                _, buffer = cv2.imencode('.jpg', frame)
                jpg_bytes = buffer.tobytes()
                
                # Send frame via WebSocket
                await websocket.send(jpg_bytes)
                
                # Display the resulting frame
                image_placeholder.image(frame, channels="BGR", use_column_width=True)
                
                # Try to receive results
                try:
                    result = await asyncio.wait_for(websocket.recv(), timeout=0.1)
                    data = json.loads(result)
                    
                    if data.get("status") == "success":
                        results_placeholder.success(
                            f"Verified: {data['label']} "
                            f"(Confidence: {data['confidence']*100:.1f}%, "
                            f"Match: {data['match_score']*100:.1f}%)"
                        )
                    elif data.get("results"):
                        results = data["results"]
                        results_text = "### Detection Results:\n"
                        for i, res in enumerate(results):
                            results_text += (
                                f"- Face {i+1}: {res['label']} "
                                f"(Confidence: {res['confidence']*100:.1f}%, "
                                f"Match: {res['match_score']*100:.1f}%)\n"
                            )
                        results_placeholder.markdown(results_text)
                
                except asyncio.TimeoutError:
                    pass
                except json.JSONDecodeError:
                    pass
                
                # Small delay to limit FPS
                await asyncio.sleep(0.2)
    
    except Exception as e:
        st.error(f"WebSocket error: {str(e)}")
    finally:
        st.session_state.recognition_active = False
        st.session_state.websocket = None

# Start/stop recognition
col1, col2 = st.columns(2)
with col1:
    user_email = st.text_input("Your Email (for tracking)", key="user_email")

if not st.session_state.recognition_active:
    if st.button("Start Recognition"):
        if user_email:
            st.session_state.recognition_active = True
            st.session_state.cap = get_video_capture()  # Reinitialize capture
            asyncio.run(recognize_faces())
        else:
            st.warning("Please enter your email")
else:
    if st.button("Stop Recognition"):
        st.session_state.recognition_active = False
        if st.session_state.websocket:
            asyncio.get_event_loop().run_until_complete(st.session_state.websocket.close())
        st.session_state.cap.release()
        st.experimental_rerun()

# Clean up on app exit
def cleanup():
    if 'cap' in st.session_state:
        st.session_state.cap.release()
    if st.session_state.recognition_active:
        st.session_state.recognition_active = False

import atexit
atexit.register(cleanup)