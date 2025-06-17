import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,  # WAJIB agar landmark iris tersedia
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Landmark indeks untuk iris & eye reference
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
LEFT_EYE = [33, 133]
RIGHT_EYE = [362, 263]

def get_eye_gaze_ratios(frame):
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        print("[DEBUG] No face landmarks detected")
        return None, None

    mesh = results.multi_face_landmarks[0].landmark

    # Hitung HR (horizontal ratio)
    left_iris_x = mesh[LEFT_IRIS[0]].x * w
    left_eye_left = mesh[LEFT_EYE[0]].x * w
    left_eye_right = mesh[LEFT_EYE[1]].x * w
    left_hr = (left_iris_x - left_eye_left) / (left_eye_right - left_eye_left + 1e-6)

    right_iris_x = mesh[RIGHT_IRIS[0]].x * w
    right_eye_left = mesh[RIGHT_EYE[0]].x * w
    right_eye_right = mesh[RIGHT_EYE[1]].x * w
    right_hr = (right_iris_x - right_eye_left) / (right_eye_right - right_eye_left + 1e-6)

    # Hitung VR (vertical ratio) sederhana
    left_iris_y = mesh[LEFT_IRIS[0]].y * h
    left_eye_top = mesh[159].y * h
    left_eye_bottom = mesh[145].y * h
    left_vr = (left_iris_y - left_eye_top) / (left_eye_bottom - left_eye_top + 1e-6)

    right_iris_y = mesh[RIGHT_IRIS[0]].y * h
    right_eye_top = mesh[386].y * h
    right_eye_bottom = mesh[374].y * h
    right_vr = (right_iris_y - right_eye_top) / (right_eye_bottom - right_eye_top + 1e-6)

    avg_hr = (left_hr + right_hr) / 2
    avg_vr = (left_vr + right_vr) / 2

    print(f"[DEBUG] HR: {avg_hr:.2f}, VR: {avg_vr:.2f}")
    return avg_hr, avg_vr
