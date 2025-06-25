import mediapipe as mp
import cv2
from modules.head_pose import estimate_head_pose

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

LEFT_IRIS = 468
LEFT_EYE = [33, 133]
LEFT_TOP_BOTTOM = [159, 145]

def get_gaze_direction(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)

    if not results.multi_face_landmarks:
        return "Face Not Detected"

    face_landmarks = results.multi_face_landmarks[0].landmark
    h, w, _ = frame.shape

    if len(face_landmarks) < 478:
        return "Landmark Error"

    # Landmark pixel
    iris = face_landmarks[LEFT_IRIS]
    eye_left = face_landmarks[LEFT_EYE[0]]
    eye_right = face_landmarks[LEFT_EYE[1]]
    eye_top = face_landmarks[LEFT_TOP_BOTTOM[0]]
    eye_bottom = face_landmarks[LEFT_TOP_BOTTOM[1]]

    iris_x, iris_y = int(iris.x * w), int(iris.y * h)
    eye_left_x = int(eye_left.x * w)
    eye_right_x = int(eye_right.x * w)
    eye_top_y = int(eye_top.y * h)
    eye_bottom_y = int(eye_bottom.y * h)

    eye_width = eye_right_x - eye_left_x
    eye_height = eye_bottom_y - eye_top_y

    rel_x = (iris_x - eye_left_x) / (eye_width + 1e-6)
    rel_y = (iris_y - eye_top_y) / (eye_height + 1e-6)

    # Head pose estimation
    landmark_px = {i: (int(face_landmarks[i].x * w), int(face_landmarks[i].y * h)) for i in [1, 152, 33, 263, 61, 291]}
    pitch, _ = estimate_head_pose(landmark_px, frame.shape)


    # Debug
    #cv2.putText(frame, f"Pitch: {pitch:.2f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

    # Enhanced logic
    if eye_height < 4:
        if pitch < 5:
            return "Looking Down (Head)"
        else:
            return "Blinking"

    if pitch > 20:  # extreme head tilt
        return "Looking Down (Head)"

    if rel_x < 0.35:
        return "Looking Right"
    elif rel_x > 0.65:
        return "Looking Left"
    elif rel_y > 0.65:
        return "Looking Down"
    else:
        return "Looking Center"

