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
print("[INIT] Loading AI Models...")
processor = PhotoProcessor()
live_processor = LiveStreamProcessor()
print("[INIT] Ready!")

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

# Global variables untuk live streaming
camera = None
current_jersey = None
current_jersey_img = None
current_jersey_meta = None

def get_camera():
    """Get atau initialize camera"""
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return camera

def generate_frames():
    """Generator function untuk video streaming"""
    global current_jersey_img
    
    cam = get_camera()
    
    if not cam.isOpened():
        print("[ERROR] Camera tidak dapat dibuka!")
        return
    
    print("[INFO] Camera streaming dimulai...")
    frame_count = 0
    
    while True:
        success, frame = cam.read()
        if not success:
            print(f"[ERROR] Gagal membaca frame dari camera")
            break
        
        frame_count += 1
        if frame_count % 30 == 0:  # Log setiap 30 frame
            print(f"[INFO] Frame #{frame_count} - Shape: {frame.shape}")
        
        # Jika ada jersey yang dipilih, overlay jersey
        if current_jersey_img is not None:
            try:
                frame = live_processor.process_frame(
                    user_frame=frame,
                    jersey_img=current_jersey_img,
                    meta=current_jersey_meta,  # Pass metadata untuk collar handling
                    scales=(0.0, 0.2, 0.1)
                )
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
    """Video streaming route untuk live camera"""
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/set_jersey', methods=['POST'])
def set_jersey():
    """Set jersey yang akan dioverlay pada live stream"""
    global current_jersey, current_jersey_img, current_jersey_meta
    
    data = request.get_json()
    jersey_name = data.get('jersey')
    
    if not jersey_name:
        current_jersey = None
        current_jersey_img = None
        current_jersey_meta = None
        return jsonify({'status': 'ok', 'message': 'Jersey cleared'})
    
    jersey_path = os.path.join(JERSEY_FOLDER, jersey_name)
    
    if not os.path.exists(jersey_path):
        return jsonify({'status': 'error', 'message': 'Jersey not found'})
    
    # Load jersey dengan alpha channel
    jersey_img = cv2.imread(jersey_path, cv2.IMREAD_UNCHANGED)
    
    if jersey_img is None:
        return jsonify({'status': 'error', 'message': 'Failed to load jersey'})
    
    # Load metadata untuk jersey ini
    meta = jersey_metadata.get(jersey_name, None)
    
    current_jersey = jersey_name
    current_jersey_img = jersey_img
    current_jersey_meta = meta
    
    return jsonify({'status': 'ok', 'jersey': jersey_name})

@app.route('/change_jersey/<jersey_name>', methods=['GET'])
def change_jersey(jersey_name):
    """Endpoint untuk mengganti jersey di live stream (dipanggil dari live.html)"""
    global current_jersey, current_jersey_img, current_jersey_meta
    
    jersey_path = os.path.join(JERSEY_FOLDER, jersey_name)
    
    if not os.path.exists(jersey_path):
        return jsonify({'success': False, 'message': 'Jersey not found'})
    
    # Load jersey dengan alpha channel
    jersey_img = cv2.imread(jersey_path, cv2.IMREAD_UNCHANGED)
    
    if jersey_img is None:
        return jsonify({'success': False, 'message': 'Failed to load jersey'})
    
    # Load metadata untuk jersey ini
    meta = jersey_metadata.get(jersey_name, None)
    if meta is None:
        print(f"[WARNING] No metadata found for {jersey_name}")
    else:
        print(f"[INFO] Loaded metadata for {jersey_name}")
    
    current_jersey = jersey_name
    current_jersey_img = jersey_img
    current_jersey_meta = meta
    
    return jsonify({'success': True, 'jersey': jersey_name})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)