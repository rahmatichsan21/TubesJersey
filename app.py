import os
import cv2
import json
import time
import numpy as np
from flask import Flask, render_template, request, jsonify, url_for, Response
from werkzeug.utils import secure_filename

# Import processor kita
# Pastikan file ini ada di folder processor/photo_processor_ai.py
from processor.photo_processor_ai import PhotoProcessor
from processor.live_processor import LiveStreamProcessor
from processor.cylindrical_warp import apply_cylindrical_to_jersey

app = Flask(__name__)

# --- KONFIGURASI ---
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
JERSEY_FOLDER = 'Assets/Jerseys/PremierLeague/Home_NOBG' # Sesuaikan path ini
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Buat folder jika belum ada
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load Jersey Metadata & Landmarks
METADATA_FILE = 'jersey_metadata.json'
LANDMARKS_FILE = 'jersey_landmarks.json'

with open(METADATA_FILE, 'r') as f:
    jersey_metadata = json.load(f)
    
with open(LANDMARKS_FILE, 'r') as f:
    jersey_landmarks = json.load(f)

# Inisialisasi AI Processor (Load Model Sekali Saja di Awal)
print("="*60)
print("[INIT] Virtual Try-On System Starting...")
print("="*60)

# Check GPU availability
try:
    import torch
    if torch.cuda.is_available():
        print(f"[SYSTEM] GPU Acceleration: ENABLED ✓")
        print(f"[SYSTEM] GPU Device: {torch.cuda.get_device_name(0)}")
    else:
        print("[SYSTEM] GPU Acceleration: DISABLED (CPU mode)")
except ImportError:
    print("[SYSTEM] PyTorch not available")

print("\n[INIT] Loading AI Models...")
processor = PhotoProcessor()
print("\n" + "="*60)
print("[INIT] ✓ System Ready!")
print("="*60 + "\n")

# Helper: Cek ekstensi file
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Helper: Load Daftar Jersey
def get_jerseys():
    jerseys = []
    if os.path.exists(JERSEY_FOLDER):
        for f in os.listdir(JERSEY_FOLDER):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                jerseys.append(f)
    return sorted(jerseys)

# --- ROUTES ---

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    jerseys = get_jerseys()
    
    if request.method == 'POST':
        # 1. Cek Upload Foto User
        if 'user_image' not in request.files:
            return "No file part", 400
        file = request.files['user_image']
        if file.filename == '':
            return "No selected file", 400
        
        # 2. Simpan Foto User
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            user_img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(user_img_path)
            
            # 3. Ambil Pilihan Jersey
            selected_jersey = request.form.get('jersey_select')
            jersey_path = os.path.join(JERSEY_FOLDER, selected_jersey)
            
            # 4. Baca Gambar
            user_image = cv2.imread(user_img_path)
            jersey_image = cv2.imread(jersey_path, cv2.IMREAD_UNCHANGED) # Baca Alpha Channel
            
            if user_image is None or jersey_image is None:
                return "Gagal membaca gambar", 500
            
            # 4.5 CYLINDRICAL WARP: Terapkan efek 3D ke jersey
            print(f"[3D PHOTO] Applying cylindrical warp to: {selected_jersey}")
            jersey_image = apply_cylindrical_to_jersey(jersey_image, strength=0.7)

            # 5. PROSES AWAL (Auto-Detect)
            # Kita panggil get_body_landmarks dulu untuk dikirim ke frontend
            # Convert BGR ke RGB untuk MediaPipe
            user_rgb = cv2.cvtColor(user_image, cv2.COLOR_BGR2RGB)
            raw_points = processor.get_body_landmarks(user_rgb)
            
            # Jika AI gagal deteksi pose, beri nilai default (tengah gambar)
            if raw_points is None:
                h, w, _ = user_image.shape
                raw_points = np.array([
                    [w*0.3, h*0.2], [w*0.4, h*0.2], [w*0.6, h*0.2], [w*0.7, h*0.2], # Bahu/Leher
                    [w*0.2, h*0.4], [w*0.8, h*0.4], # Lengan
                    [w*0.3, h*0.6], [w*0.7, h*0.6]  # Pinggang
                ])
            
            # Konversi numpy array ke list python biasa (biar bisa jadi JSON)
            landmarks_list = raw_points.tolist()

            # Lakukan Processing Pertama (Pakai Auto Landmarks)
            result_image = processor.process_photo(
                user_image.copy(), 
                jersey_image, 
                meta={'fileName': selected_jersey},
                custom_body_points=None # Pakai auto dulu
            )
            
            # Simpan Hasil
            result_filename = f"result_{int(time.time())}.jpg"
            result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
            cv2.imwrite(result_path, result_image)
            
            # 6. RENDER HALAMAN DENGAN DATA LANDMARK
            return render_template('upload.html', 
                                   jerseys=jerseys,
                                   result_image=result_path,
                                   user_image_path=user_img_path,     # Path asli (untuk re-process)
                                   selected_jersey=selected_jersey,   # Nama jersey
                                   landmarks=json.dumps(landmarks_list) # Data titik untuk JS
                                   )

    return render_template('upload.html', jerseys=jerseys)

# --- API BARU UNTUK UPDATE MANUAL ---
@app.route('/update_warp', methods=['POST'])
def update_warp():
    data = request.json
    points = data['points']       # List [[x,y], [x,y]...] dari JS
    jersey_name = data['jersey']
    user_img_path = data['user_image_path'] # Kita kirim balik path ini dari frontend
    
    # Validasi path (Security check sederhana)
    if 'static' not in user_img_path: return jsonify({'error': 'Invalid path'}), 400

    # 1. Load Gambar
    user_image = cv2.imread(user_img_path)
    jersey_path = os.path.join(JERSEY_FOLDER, jersey_name)
    jersey_image = cv2.imread(jersey_path, cv2.IMREAD_UNCHANGED)
    
    # 1.5 Apply Cylindrical Warp
    print(f"[3D ADJUST] Applying cylindrical warp to: {jersey_name}")
    jersey_image = apply_cylindrical_to_jersey(jersey_image, strength=0.7)
    
    # 2. Re-Process dengan CUSTOM POINTS
    # Prosesor akan memakai titik yang dikirim user, bukan MediaPipe
    result_image = processor.process_photo(
        user_image.copy(), 
        jersey_image, 
        meta={'fileName': jersey_name},
        custom_body_points=points 
    )
    
    # 3. Simpan & Return URL
    new_filename = f"adjusted_{int(time.time())}.jpg"
    new_path = os.path.join(app.config['RESULT_FOLDER'], new_filename)
    cv2.imwrite(new_path, result_image)
    
    return jsonify({
        'status': 'ok', 
        'image_url': new_path + '?t=' + str(time.time()) # Timestamp agar tidak cache
    })

@app.route('/live', methods=['GET'])
def live():
    """Halaman Live Camera untuk Virtual Try-On Real-time"""
    jerseys = get_jerseys()
    return render_template('live.html', jerseys=jerseys)

# Cache untuk menyimpan gambar jersey yang sudah di-load agar cepat
# Format: {'NamaJersey.png': (image_data, metadata)}
JERSEY_CACHE = {}
camera = None

def get_camera():
    """Get atau initialize camera dengan retry mechanism"""
    global camera
    if camera is None:
        print("[CAMERA] Initializing camera...")
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
    
    # Jika camera tidak terbuka, coba buka ulang
    if not camera.isOpened():
        print("[CAMERA] Camera not opened, trying to reopen...")
        camera.release()
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
    return camera

def generate_frames(jersey_name=None):
    """Generator function dengan Local Processor (Anti-Jitter per User)"""
    cam = get_camera()
    
    # 1. INIT PROCESSOR LOKAL (Agar punya memori smoothing sendiri per user)
    # Import sudah ada di atas: from processor.live_processor import LiveStreamProcessor
    local_live_processor = LiveStreamProcessor()
    print(f"[SESSION] Created new LiveStreamProcessor for session")
    
    # Hard check - pastikan camera benar-benar terbuka
    if not cam.isOpened():
        print("[FATAL] Kamera tidak terdeteksi atau dikunci aplikasi lain!")
        # Kirim frame error
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_frame, "Camera Error", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        ret, buffer = cv2.imencode('.jpg', error_frame)
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        return
    
    # 1. Siapkan Jersey (Ambil dari Cache atau Load Baru)
    active_jersey_img = None
    active_jersey_meta = None
    
    if jersey_name and jersey_name != 'null' and jersey_name != 'None':
        # Cek apakah ada di cache?
        if jersey_name in JERSEY_CACHE:
            active_jersey_img, active_jersey_meta = JERSEY_CACHE[jersey_name]
            print(f"[CACHE HIT] Using cached jersey: {jersey_name}")
        else:
            # Jika belum ada, load dari disk lalu simpan ke cache
            path = os.path.join(JERSEY_FOLDER, jersey_name)
            if os.path.exists(path):
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                
                # SAFETY: Jika JPG (3 channel), tambah alpha channel
                if img is not None and len(img.shape) == 3 and img.shape[2] == 3:
                    print(f"[INFO] Converting JPG to RGBA for: {jersey_name}")
                    b, g, r = cv2.split(img)
                    alpha = np.ones_like(b) * 255
                    img = cv2.merge((b, g, r, alpha))
                
                # CYLINDRICAL WARP: Terapkan efek 3D (dilakukan sekali saat load)
                if img is not None:
                    print(f"[3D] Applying cylindrical warp to: {jersey_name}")
                    img = apply_cylindrical_to_jersey(img, strength=0.7)
                
                meta = jersey_metadata.get(jersey_name, None)
                if img is not None:
                    JERSEY_CACHE[jersey_name] = (img, meta)  # Simpan ke cache
                    active_jersey_img = img
                    active_jersey_meta = meta
                    print(f"[CACHE MISS] Loaded & Cached with 3D Effect: {jersey_name}")
    
    print(f"[INFO] Camera streaming dimulai... Jersey: {jersey_name or 'None'}")
    frame_count = 0
    
    while True:
        success, frame = cam.read()
        if not success:
            print(f"[ERROR] Gagal membaca frame dari camera")
            break
        
        frame_count += 1
        if frame_count % 90 == 0:  # Log setiap 90 frame (3 detik @ 30fps)
            print(f"[INFO] Frame #{frame_count} - Jersey: {jersey_name or 'None'}")
        
        # 2. Proses frame menggunakan jersey yang diminta USER INI
        if active_jersey_img is not None:
            try:
                processed = local_live_processor.process_frame(
                    user_frame=frame,
                    jersey_img=active_jersey_img,
                    meta=active_jersey_meta,
                    scales=(0.0, 0.2, 0.1)
                )
                if processed is not None:
                    frame = processed
            except Exception as e:
                print(f"[ERROR] Processing frame: {e}")
                import traceback
                traceback.print_exc()
        
        # Encode frame ke JPEG
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            print(f"[ERROR] Gagal encode frame ke JPEG")
            continue
        
        frame_bytes = buffer.tobytes()
        
        # Yield frame dalam format multipart
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Video streaming route untuk live camera (Stateless)"""
    # Ambil nama jersey dari URL parameter (contoh: /video_feed?jersey=Arsenal.png)
    selected_jersey = request.args.get('jersey')
    
    return Response(
        generate_frames(selected_jersey),  # Kirim nama jersey ke generator
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)