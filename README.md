# Jersey Virtual Try-On - How to Run

## Prerequisites
- Python 3.8 atau lebih tinggi
- Webcam (untuk fitur Live Streaming)
- GPU dengan CUDA support (opsional, untuk performa lebih cepat)

## Installation

### 1. Clone atau Download Project
```bash
cd TubesJersey
```

### 2. Buat Virtual Environment (Recommended)
```bash
python -m venv .venv
```

### 3. Aktivasi Virtual Environment

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
.venv\Scripts\activate.bat
```

**Linux/Mac:**
```bash
source .venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

**Note:** Instalasi pertama kali akan memakan waktu karena download model AI (transformers, rembg, dll)

## Running the Application

### 1. Jalankan Server Flask
```bash
python app.py
```

### 2. Buka Browser
Akses aplikasi di: **http://localhost:5000**

### 3. Stop Server
Tekan `Ctrl + C` di terminal

## Fitur Aplikasi

### Upload Photo Mode
1. Klik menu **"Upload Photo"**
2. Upload foto Anda (pastikan badan bagian atas terlihat jelas)
3. Pilih jersey yang diinginkan
4. Klik **"Process"**
5. Hasil akan ditampilkan dengan opsi download

### Live Streaming Mode
1. Klik menu **"Live Streaming"**
2. Izinkan akses webcam
3. Pilih jersey dari panel kanan
4. Jersey akan langsung ter-apply secara real-time

## Struktur Folder Jersey

Jersey disimpan di:
```
Assets/Jerseys/PremierLeague/Home_NOBG/
```

Format file yang didukung:
- PNG (dengan transparansi/alpha channel)
- JPG/JPEG

## Troubleshooting

### Error: Module not found
```bash
pip install -r requirements.txt --upgrade
```

### Webcam tidak terdeteksi
- Pastikan webcam terhubung dan tidak digunakan aplikasi lain
- Cek permission webcam di browser
- Restart browser dan aplikasi

### Performance lambat
- Gunakan GPU jika tersedia
- Kurangi resolusi webcam
- Tutup aplikasi lain yang berat

### Port 5000 sudah digunakan
Edit `app.py` dan ubah port:
```python
app.run(debug=True, port=5001)  # Ganti 5000 ke 5001
```

## Tools Tambahan

### Jersey Annotator
Tool untuk membuat metadata jersey baru:
```bash
python jersey_anotator.py
```

### Curve Jersey Annotator
Tool untuk membuat curve annotation (3D effect):
```bash
python curve_jersey_annotator.py
```

## Notes
- Pastikan pencahayaan cukup untuk hasil terbaik
- Gunakan background yang tidak terlalu ramai
- Posisikan badan menghadap kamera secara langsung
