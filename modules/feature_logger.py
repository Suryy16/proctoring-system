import csv
import os
from datetime import datetime

def save_features(hr, vr, pitch, yaw, label=None, path='data/features_log.csv'):
    """
    Menyimpan fitur-fitur (HR, VR, Pitch, Yaw, Label) ke dalam file CSV.
    Args:
        hr (float): Nilai HR (Horizontal Ratio) yang akan disimpan.
        vr (float): Nilai VR (Vertical Ratio) yang akan disimpan.
        pitch (float): Nilai pitch yang akan disimpan.
        yaw (float): Nilai yaw yang akan disimpan.
        label (str, optional): Label tambahan untuk data, default None.
        path (str, optional): Path file CSV untuk menyimpan data, default 'data/features_log.csv'.
    Baris kode:
            # Membuat direktori tujuan jika belum ada.
            # Mengecek apakah file CSV sudah ada.
            # Membuka file CSV dalam mode append.
                # Membuat objek writer untuk menulis ke file CSV.
                    # Jika file baru, tulis header kolom.
                # Menulis baris data baru dengan timestamp saat ini dan fitur-fitur yang diberikan.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.exists(path)

    with open(path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "HR", "VR", "Pitch", "Yaw", "Label"])
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), hr, vr, pitch, yaw, label if label is not None else ""])
