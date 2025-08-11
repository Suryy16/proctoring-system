# 📚 Proctoring Application

Aplikasi Proctoring ini terdiri dari dua komponen utama:

1. **Service API** – Berjalan menggunakan Docker.
2. **UI** – Dibuat dengan Streamlit menggunakan Python virtual environment.

---

## 🚀 Menjalankan Service API

1. Pastikan **Docker** dan **Docker Compose** sudah terinstal.
2. Jalankan perintah:

```bash
docker-compose up --build
```

API akan otomatis dibangun dan dijalankan.

---

## 💻 Menjalankan UI (Streamlit)

1. Pastikan setidaknya **Python 3.10** sudah terinstal.
2. Buat virtual environment:

```bash
python -m venv ui-venv
```

4. Aktifkan virtual environment:

**Windows:**

```bash
.\ui-venv\Scripts\activate
```

**Mac/Linux:**

```bash
source ui-venv/bin/activate
```

5. Install dependencies:

```bash
pip install -r requirements/requirements-ui.txt
```

6. Jalankan aplikasi Streamlit:

```bash
streamlit run streamlit_app.py
```

---

## 📌 Catatan

* Pastikan **Service API** sudah berjalan sebelum menjalankan UI.
* Untuk menghentikan Docker service, gunakan `Ctrl + C` atau:

```bash
docker-compose down
```

* Jika ada perubahan pada kode API, ulangi proses build dengan:

```bash
docker-compose up --build
```

---

## 📄 Lisensi

Proyek ini dilisensikan di bawah MIT License. Lihat File LICENSE untuk detail lebih lanjut.



