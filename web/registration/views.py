import os
import requests
import streamlit as st

from config import API_BASE_URL, ROOT_DATABASE_DIR

def get_registered_students():
    try:
        dataset_path = os.path.join(ROOT_DATABASE_DIR, "dataset")
        if os.path.exists(dataset_path):
            return sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
        return []
    except Exception as e:
        st.error(f"Error accessing student directory: {str(e)}")
        return []

def show_registration_tab():
    """Display the registration tab"""
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