# Konfigurasi sistem
CONFIG = {
    "database_path": "database/processed_dataset",
    "model_name": "Facenet512",
    "detector_backend": "opencv",
    "threshold": 0.1,  # Threshold untuk menentukan kecocokan wajah
    "save_unknown_faces": True,  # Menyimpan wajah yang tidak dikenali
    "camera_index": 0,  # Index kamera (biasanya 0 untuk webcam utama)
    "display_width": 800,
    "display_height": 600
}