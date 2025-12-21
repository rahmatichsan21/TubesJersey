import cv2
import json
import os
import glob
import numpy as np

# --- KONFIGURASI ---
JERSEY_FOLDER = "Assets/Jerseys/PremierLeague/Home_NOBG"  
OUTPUT_FILE = "jersey_landmarks.json"

# Daftar Titik
LANDMARKS = [
    "1. Bahu Kiri (Left Shoulder)",
    "2. Leher Kiri (Left Neck)",
    "3. Leher Kanan (Right Neck)",
    "4. Bahu Kanan (Right Shoulder)",
    "5. Ujung Lengan Kiri (Left Sleeve)",
    "6. Ujung Lengan Kanan (Right Sleeve)",
    "7. Pinggang Kiri Bawah (Left Hip)",
    "8. Pinggang Kanan Bawah (Right Hip)"
]

current_points = []
current_img = None
base_img = None
mouse_pos = (0, 0) # Untuk kaca pembesar

def redraw_points():
    """Menggambar ulang UI: Titik, Garis, dan Kaca Pembesar"""
    global current_img, base_img, current_points, mouse_pos
    
    # 1. Reset ke gambar bersih
    current_img = base_img.copy()
    
    # 2. Gambar Titik & Garis
    for i, pt in enumerate(current_points):
        x, y = pt
        cv2.circle(current_img, (x, y), 4, (0, 0, 255), -1) # Titik Merah
        cv2.circle(current_img, (x, y), 2, (0, 255, 255), -1) # Tengah Kuning
        cv2.putText(current_img, str(i+1), (x+8, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        if i > 0: # Garis konektor hijau
            prev_pt = current_points[i-1]
            cv2.line(current_img, (prev_pt[0], prev_pt[1]), (x, y), (0, 255, 0), 1)

    # 3. Fitur Kaca Pembesar (Magnifier) di Pojok Kiri Atas
    h, w = current_img.shape[:2]
    mx, my = mouse_pos
    
    # Ukuran kotak zoom
    zoom_size = 100 
    scale = 2
    
    # Pastikan koordinat aman (tidak keluar batas gambar)
    x1 = max(0, mx - zoom_size // 2)
    y1 = max(0, my - zoom_size // 2)
    x2 = min(w, mx + zoom_size // 2)
    y2 = min(h, my + zoom_size // 2)
    
    if (x2 - x1 > 0) and (y2 - y1 > 0):
        # Ambil potongan area di sekitar mouse dari gambar BERSIH (base_img)
        roi = base_img[y1:y2, x1:x2]
        
        # Perbesar roi
        roi_zoomed = cv2.resize(roi, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        
        # Gambar crosshair di tengah zoom
        zh, zw = roi_zoomed.shape[:2]
        cv2.line(roi_zoomed, (zw//2, 0), (zw//2, zh), (0, 0, 255), 1)
        cv2.line(roi_zoomed, (0, zh//2), (zw, zh//2), (0, 0, 255), 1)
        
        # Tempelkan di pojok kiri atas layar (dengan border putih)
        try:
            current_img[10:10+zh, 10:10+zw] = roi_zoomed
            cv2.rectangle(current_img, (10, 10), (10+zw, 10+zh), (255, 255, 255), 2)
            cv2.putText(current_img, "ZOOM", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
        except:
            pass # Skip jika gambar terlalu kecil untuk zoom box

def click_event(event, x, y, flags, params):
    global current_points, mouse_pos
    
    # Update posisi mouse untuk zoom
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_pos = (x, y)
        redraw_points()
        cv2.imshow("Jersey Annotator", current_img)
    
    # Klik Kiri: Tambah Titik
    elif event == cv2.EVENT_LBUTTONDOWN:
        if len(current_points) >= len(LANDMARKS):
            print("[INFO] Titik lengkap! Tekan SPASI (Simpan) atau 'u' (Undo).")
            return

        current_points.append([x, y])
        redraw_points()
        cv2.imshow("Jersey Annotator", current_img)
        print(f"[KLIK] Poin {len(current_points)}: ({x}, {y})")

def annotate_jerseys():
    global current_img, base_img, current_points
    
    files = glob.glob(os.path.join(JERSEY_FOLDER, "*.png"))
    if not files: print(f"[ERROR] Tidak ada PNG di {JERSEY_FOLDER}"); return

    all_data = {}
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r') as f: all_data = json.load(f)
        print(f"[INFO] Database dimuat: {len(all_data)} jersey.")

    print("\n=== CONTROLS ===")
    print("[ Klik ] : Tambah titik")
    print("[  u   ] : UNDO (Hapus titik terakhir)")
    print("[Enter ] : SIMPAN & Lanjut")
    print("[  r   ] : RESET (Hapus semua titik di gambar ini)")
    print("[  s   ] : SKIP (Lewati gambar ini)")
    print("[  q   ] : QUIT (Keluar)")
    print("================\n")

    for filepath in files:
        filename = os.path.basename(filepath)
        img = cv2.imread(filepath)
        if img is None: continue
        
        base_img = img.copy()
        current_img = img.copy()
        current_points = []
        
        status_msg = "BARU"
        if filename in all_data:
            status_msg = "EDIT"
            current_points = all_data[filename]
        
        # Init UI
        redraw_points()
        cv2.namedWindow("Jersey Annotator")
        cv2.setMouseCallback("Jersey Annotator", click_event)
        
        print(f"--> Mengedit: {filename} [{status_msg}]")
        
        while True:
            # Judul Window Dinamis
            idx = len(current_points)
            if idx < len(LANDMARKS):
                msg = f"KLIK {idx+1}: {LANDMARKS[idx]}"
            else:
                msg = "LENGKAP! Tekan SPASI untuk Simpan."
            
            cv2.setWindowTitle("Jersey Annotator", f"[{status_msg}] {filename} - {msg}")
            cv2.imshow("Jersey Annotator", current_img)
            
            key = cv2.waitKey(1) & 0xFF
            
            # --- CONTROLS ---
            
            # Undo ('u')
            if key == ord('u'):
                if len(current_points) > 0:
                    popped = current_points.pop()
                    print(f"[UNDO] Menghapus titik terakhir: {popped}")
                    redraw_points()
                else:
                    print("[WARN] Belum ada titik untuk dihapus.")

            # Simpan (Enter/Spasi)
            elif (key == 13 or key == 32):
                if len(current_points) == len(LANDMARKS):
                    all_data[filename] = current_points
                    print(f"[SIMPAN] {filename} OK!")
                    with open(OUTPUT_FILE, 'w') as f: json.dump(all_data, f, indent=4)
                    break
                else:
                    print(f"[WARN] Belum lengkap ({len(current_points)}/8).")

            # Reset ('r')
            elif key == ord('r'):
                current_points = []
                redraw_points()
                print("[RESET] Titik dihapus.")

            # Skip ('s')
            elif key == ord('s'):
                print(f"[SKIP] {filename} dilewati.")
                break

            # Quit ('q')
            elif key == ord('q'):
                print("[KELUAR] Bye!")
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()
    print("[SELESAI] Semua tersimpan.")

if __name__ == "__main__":
    annotate_jerseys()