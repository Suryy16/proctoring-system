import mediapipe as mp
import cv2

mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

def detect_faces(frame):
    """
    Mendeteksi wajah pada sebuah frame gambar.

    Args:
        frame (numpy.ndarray): Frame gambar dalam format BGR.

    Returns:
        tuple: 
            - faces (list of tuple): Daftar bounding box wajah yang terdeteksi dalam format (x, y, w, h).
            - face_roi (numpy.ndarray or None): Region of interest (ROI) dari wajah pertama yang terdeteksi, atau None jika tidak ada wajah.

    Penjelasan kode baris per baris:
    1. Mengubah format warna frame dari BGR ke RGB.
    2. Memproses gambar RGB untuk mendeteksi wajah menggunakan face_detector.
    3. Membuat list kosong untuk menyimpan bounding box wajah.
    4. Menginisialisasi face_roi sebagai None.
    5. Jika ada deteksi wajah:
        6. Melakukan iterasi pada setiap deteksi wajah.
        7. Mengambil bounding box relatif dari deteksi.
        8. Mendapatkan tinggi dan lebar frame.
        9-12. Menghitung koordinat dan ukuran bounding box dalam piksel.
        13. Menambahkan bounding box ke list faces.
        14. Mengambil ROI wajah dari frame berdasarkan bounding box (hanya satu wajah).
    15. Mengembalikan daftar bounding box wajah dan ROI wajah pertama (atau None jika tidak ada wajah).
    """
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detector.process(image_rgb)
    faces = []
    face_roi = None

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x = int(bboxC.xmin * iw)
            y = int(bboxC.ymin * ih)
            w = int(bboxC.width * iw)
            h = int(bboxC.height * ih)
            faces.append((x, y, w, h))
            face_roi = frame[y:y+h, x:x+w]  # hanya ambil satu wajah

    return faces, face_roi
