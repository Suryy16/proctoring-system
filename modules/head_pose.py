import cv2
import numpy as np

model_points = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye left corner
    (225.0, 170.0, -135.0),      # Right eye right corner
    (-150.0, -150.0, -125.0),    # Left mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
], dtype="double")

def estimate_head_pose(landmarks, image_shape):
    h, w = image_shape[:2]

    image_points = np.array([
        landmarks[1],    # Nose tip
        landmarks[152],  # Chin
        landmarks[33],   # Left eye left corner
        landmarks[263],  # Right eye right corner
        landmarks[61],   # Left mouth
        landmarks[291]   # Right mouth
    ], dtype="double")

    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4,1))
    success, rotation_vector, _ = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
    if not success:
        return None, None  # Gagal

    rmat, _ = cv2.Rodrigues(rotation_vector)

    sy = np.sqrt(rmat[0, 0]**2 + rmat[1, 0]**2)

    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(rmat[2, 1], rmat[2, 2])  # roll
        y = np.arctan2(-rmat[2, 0], sy)         # pitch
        z = np.arctan2(rmat[1, 0], rmat[0, 0])  # yaw
    else:
        x = np.arctan2(-rmat[1, 2], rmat[1, 1])
        y = np.arctan2(-rmat[2, 0], sy)
        z = 0

    pitch = y * 180 / np.pi
    yaw = z * 180 / np.pi
    roll = x * 180 / np.pi

    return pitch, yaw  # bisa juga return pitch, yaw, roll
