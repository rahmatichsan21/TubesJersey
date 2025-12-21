import cv2
import json
import os
import glob
import numpy as np
from scipy.interpolate import CubicSpline, splprep, splev

# --- KONFIGURASI ---
ASSET_FOLDER = "Assets/Jerseys/PremierLeague/Home_NOBG/"
OUTPUT_FILE = "jersey_metadata.json"

# Variabel Global
current_image = None
display_image = None
points = []
zoom_level = 1.0
pan_x = 0
pan_y = 0

# Urutan klik (TOTAL 7 TITIK untuk kerah yang smooth):
# 1. Bahu Kiri (Titik paling kiri bahu)
# 2. Kerah Kiri Atas (Antara bahu kiri dan tengah)
# 3. Kerah Tengah Atas (Bagian tengah atas kerah)
# 4. Kerah Bawah (Titik paling bawah kerah - untuk V-neck atau round neck)
# 5. Kerah Tengah Kanan (Bagian tengah kanan atas kerah)
# 6. Kerah Kanan Atas (Antara tengah dan bahu kanan)
# 7. Bahu Kanan (Titik paling kanan bahu)
# 8. Ujung Lengan Kiri
# 9. Ujung Lengan Kanan

POINT_NAMES = [
    "Bahu Kiri", 
    "Kerah Kiri-Atas",
    "Kerah Tengah-Atas", 
    "Kerah Bawah",
    "Kerah Tengah-Kanan",
    "Kerah Kanan-Atas",
    "Bahu Kanan",
    "Lengan Kiri", 
    "Lengan Kanan"
]

POINT_COLORS = [
    (0, 255, 0),    # Hijau - Bahu Kiri
    (0, 255, 255),  # Cyan - Kerah Kiri Atas
    (255, 255, 0),  # Kuning - Kerah Tengah Atas
    (0, 0, 255),    # Merah - Kerah Bawah (CENTER)
    (255, 255, 0),  # Kuning - Kerah Tengah Kanan
    (0, 255, 255),  # Cyan - Kerah Kanan Atas
    (0, 255, 0),    # Hijau - Bahu Kanan
    (255, 0, 0),    # Biru - Lengan Kiri
    (255, 0, 0)     # Biru - Lengan Kanan
]

def interpolate_collar_curve(collar_points):
    """
    Buat kurva smooth dari titik-titik kerah menggunakan spline interpolation
    """
    if len(collar_points) < 3:
        return collar_points
    
    # Convert to numpy array
    points_array = np.array(collar_points)
    
    # Parametric spline interpolation
    try:
        # Parameter untuk smoothness (s=0 untuk exact fit, s>0 untuk smoothing)
        tck, u = splprep([points_array[:, 0], points_array[:, 1]], s=0, k=min(3, len(collar_points)-1))
        
        # Generate smooth curve dengan lebih banyak titik
        u_new = np.linspace(0, 1, 50)  # 50 titik untuk kurva smooth
        x_new, y_new = splev(u_new, tck)
        
        return np.column_stack([x_new, y_new])
    except:
        # Fallback: simple linear interpolation
        return collar_points


def update_display():
    """Update tampilan dengan zoom, pan, dan kurva smooth"""
    global display_image, current_image, points, zoom_level, pan_x, pan_y
    
    if current_image is None:
        return
    
    h, w = current_image.shape[:2]
    
    # Apply zoom
    new_w = int(w * zoom_level)
    new_h = int(h * zoom_level)
    zoomed = cv2.resize(current_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Apply pan (crop)
    start_x = max(0, pan_x)
    start_y = max(0, pan_y)
    end_x = min(new_w, start_x + w)
    end_y = min(new_h, start_y + h)
    
    # Jika area crop lebih kecil dari original, pad dengan hitam
    display_image = np.zeros((h, w, 3), dtype=np.uint8)
    crop = zoomed[start_y:end_y, start_x:end_x]
    display_image[:crop.shape[0], :crop.shape[1]] = crop[:, :, :3] if crop.shape[2] > 3 else crop
    
    # Gambar kurva kerah smooth jika sudah ada 4+ titik kerah
    if len(points) >= 4:
        collar_points = []
        for i in range(min(7, len(points))):  # Maksimal 7 titik kerah
            x = int(points[i][0] * zoom_level - pan_x)
            y = int(points[i][1] * zoom_level - pan_y)
            if 0 <= x < w and 0 <= y < h:
                collar_points.append([x, y])
        
        if len(collar_points) >= 4:
            # Interpolate kurva smooth
            smooth_curve = interpolate_collar_curve(collar_points)
            curve_pts = smooth_curve.astype(np.int32)
            
            # Filter titik yang di dalam bounds
            valid_pts = []
            for pt in curve_pts:
                if 0 <= pt[0] < w and 0 <= pt[1] < h:
                    valid_pts.append(pt)
            
            if len(valid_pts) > 1:
                # Gambar kurva dengan gradient warna
                for i in range(len(valid_pts) - 1):
                    cv2.line(display_image, tuple(valid_pts[i]), tuple(valid_pts[i+1]), 
                            (0, 255, 255), 3)
    
    # Gambar titik-titik yang sudah diklik
    for i, pt in enumerate(points):
        # Convert original coordinate to zoomed coordinate
        zoomed_x = int(pt[0] * zoom_level - pan_x)
        zoomed_y = int(pt[1] * zoom_level - pan_y)
        
        if 0 <= zoomed_x < w and 0 <= zoomed_y < h:
            # Gambar circle dengan size berbeda untuk titik penting
            radius = 10 if i in [0, 3, 6] else 7  # Bahu dan kerah bawah lebih besar
            cv2.circle(display_image, (zoomed_x, zoomed_y), radius, POINT_COLORS[i], -1)
            cv2.circle(display_image, (zoomed_x, zoomed_y), radius + 2, (255, 255, 255), 2)
            
            # Gambar label
            label = f"{i+1}"
            label_size = 0.6 if i in [0, 3, 6] else 0.4
            cv2.putText(display_image, label, (zoomed_x + 15, zoomed_y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, label_size, (255, 255, 255), 2)
            cv2.putText(display_image, label, (zoomed_x + 15, zoomed_y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, label_size, POINT_COLORS[i], 1)
    
    # Gambar garis bantu vertikal tengah
    center_x = w // 2
    cv2.line(display_image, (center_x, 0), (center_x, h), (128, 128, 128), 1, cv2.LINE_AA)
    
    # Info panel MINIMAL - hanya info penting
    info_text = [
        f"{len(points)}/9 | Zoom:{zoom_level:.1f}x",
        f"[S]Save [R]Reset [U]Undo [H]Help"
    ]
    
    # Background panel kecil semi-transparan
    panel_height = 50
    overlay_panel = display_image.copy()
    cv2.rectangle(overlay_panel, (5, 5), (220, panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay_panel, 0.6, display_image, 0.4, 0, display_image)
    
    y_offset = 20
    for i, text in enumerate(info_text):
        color = (0, 255, 255) if i == 0 else (180, 180, 180)
        cv2.putText(display_image, text, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
        y_offset += 16
    
    # Petunjuk titik berikutnya - KECIL di kanan bawah
    if len(points) < 9:
        next_text = f"{len(points)+1}. {POINT_NAMES[len(points)]}"
        instruction_y = h - 15
        instruction_x = w - 200
        
        # Background kecil semi-transparan
        text_size = cv2.getTextSize(next_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        overlay = display_image.copy()
        cv2.rectangle(overlay, (instruction_x - 5, instruction_y - 18),
                     (instruction_x + text_size[0] + 5, instruction_y + 2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, display_image, 0.4, 0, display_image)
        
        cv2.putText(display_image, next_text, (instruction_x, instruction_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, POINT_COLORS[len(points)], 1, cv2.LINE_AA)
    
    cv2.imshow("Jersey Curve Annotator", display_image)


def click_event(event, x, y, flags, params):
    global points, zoom_level, pan_x, pan_y, current_image
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 9:
            # Convert display coordinate back to original coordinate
            orig_x = int((x + pan_x) / zoom_level)
            orig_y = int((y + pan_y) / zoom_level)
            
            # Clamp to image bounds
            h, w = current_image.shape[:2]
            orig_x = max(0, min(w - 1, orig_x))
            orig_y = max(0, min(h - 1, orig_y))
            
            points.append([orig_x, orig_y])
            print(f"✓ [{len(points)}/9] {POINT_NAMES[len(points)-1]} di ({orig_x}, {orig_y})")
            update_display()
    
    elif event == cv2.EVENT_MOUSEWHEEL:
        # Zoom in/out
        old_zoom = zoom_level
        if flags > 0:  # Scroll up = zoom in
            zoom_level = min(3.0, zoom_level * 1.2)
        else:  # Scroll down = zoom out
            zoom_level = max(0.5, zoom_level / 1.2)
        
        # Adjust pan to keep mouse position centered
        if zoom_level != old_zoom:
            pan_x = int(pan_x * (zoom_level / old_zoom))
            pan_y = int(pan_y * (zoom_level / old_zoom))
        
        update_display()
    
    elif event == cv2.EVENT_RBUTTONDOWN:
        # Start drag
        params['dragging'] = True
        params['drag_start'] = (x, y)
    
    elif event == cv2.EVENT_RBUTTONUP:
        params['dragging'] = False
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if params.get('dragging', False):
            dx = x - params['drag_start'][0]
            dy = y - params['drag_start'][1]
            pan_x = max(0, pan_x - dx)
            pan_y = max(0, pan_y - dy)
            params['drag_start'] = (x, y)
            update_display()


def show_collar_guide():
    """Tampilkan panduan lengkap anotasi kerah dengan kurva"""
    guide = np.zeros((700, 900, 3), dtype=np.uint8)
    
    # Title
    cv2.putText(guide, "PANDUAN ANOTASI KERAH LENGKUNG (7 TITIK)", (50, 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    # V-Neck Example dengan 7 titik
    cv2.putText(guide, "1. V-NECK COLLAR (Deep V)", (30, 90), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    collar_v = np.array([
        [100, 120],   # 1. Bahu Kiri
        [130, 125],   # 2. Kerah Kiri Atas
        [160, 135],   # 3. Kerah Tengah Atas
        [190, 200],   # 4. Kerah Bawah (DEEP)
        [220, 135],   # 5. Kerah Tengah Kanan
        [250, 125],   # 6. Kerah Kanan Atas
        [280, 120]    # 7. Bahu Kanan
    ])
    
    # Interpolate smooth curve
    smooth_v = interpolate_collar_curve(collar_v)
    for i in range(len(smooth_v) - 1):
        pt1 = tuple(smooth_v[i].astype(int))
        pt2 = tuple(smooth_v[i+1].astype(int))
        cv2.line(guide, pt1, pt2, (0, 255, 255), 2)
    
    # Draw points
    for i, pt in enumerate(collar_v):
        color = (0, 255, 0) if i in [0, 6] else (0, 0, 255) if i == 3 else (255, 255, 0)
        cv2.circle(guide, tuple(pt), 5, color, -1)
        cv2.putText(guide, str(i+1), (pt[0]+10, pt[1]-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Round Neck Example
    cv2.putText(guide, "2. ROUND NECK (O-Neck)", (30, 280), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    collar_round = np.array([
        [100, 310],   # 1. Bahu Kiri
        [130, 312],   # 2. Kerah Kiri Atas
        [160, 320],   # 3. Kerah Tengah Atas
        [190, 340],   # 4. Kerah Bawah (SHALLOW)
        [220, 320],   # 5. Kerah Tengah Kanan
        [250, 312],   # 6. Kerah Kanan Atas
        [280, 310]    # 7. Bahu Kanan
    ])
    
    smooth_round = interpolate_collar_curve(collar_round)
    for i in range(len(smooth_round) - 1):
        pt1 = tuple(smooth_round[i].astype(int))
        pt2 = tuple(smooth_round[i+1].astype(int))
        cv2.line(guide, pt1, pt2, (0, 255, 255), 2)
    
    for i, pt in enumerate(collar_round):
        color = (0, 255, 0) if i in [0, 6] else (0, 0, 255) if i == 3 else (255, 255, 0)
        cv2.circle(guide, tuple(pt), 5, color, -1)
        cv2.putText(guide, str(i+1), (pt[0]+10, pt[1]-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Polo Collar Example
    cv2.putText(guide, "3. POLO COLLAR (Kerah Kaku)", (30, 470), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    collar_polo = np.array([
        [100, 500],   # 1. Bahu Kiri
        [130, 502],   # 2. Kerah Kiri Atas
        [160, 505],   # 3. Kerah Tengah Atas
        [190, 510],   # 4. Kerah Bawah (VERY SHALLOW)
        [220, 505],   # 5. Kerah Tengah Kanan
        [250, 502],   # 6. Kerah Kanan Atas
        [280, 500]    # 7. Bahu Kanan
    ])
    
    smooth_polo = interpolate_collar_curve(collar_polo)
    for i in range(len(smooth_polo) - 1):
        pt1 = tuple(smooth_polo[i].astype(int))
        pt2 = tuple(smooth_polo[i+1].astype(int))
        cv2.line(guide, pt1, pt2, (0, 255, 255), 2)
    
    for i, pt in enumerate(collar_polo):
        color = (0, 255, 0) if i in [0, 6] else (0, 0, 255) if i == 3 else (255, 255, 0)
        cv2.circle(guide, tuple(pt), 5, color, -1)
        cv2.putText(guide, str(i+1), (pt[0]+10, pt[1]-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Tips section
    cv2.putText(guide, "TIPS PENTING:", (450, 90), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    tips = [
        "• 7 titik = kurva lengkung smooth",
        "• Jangan skip titik tengah!",
        "• Titik 1 & 7: Ujung bahu",
        "• Titik 4: Kerah paling bawah",
        "• Simetris kiri-kanan",
        "",
        "• Zoom in untuk presisi",
        "• [U] untuk undo jika salah",
        "• Garis abu = tengah simetri",
        "",
        "• Kurva cyan = preview hasil",
        "• Titik hijau = bahu",
        "• Titik merah = kerah bawah",
        "• Titik kuning = kurva",
        "• Titik biru = lengan"
    ]
    
    y_pos = 130
    for tip in tips:
        cv2.putText(guide, tip, (460, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        y_pos += 30
    
    # Urutan klik
    cv2.rectangle(guide, (450, y_pos), (880, y_pos + 180), (50, 50, 50), -1)
    cv2.putText(guide, "URUTAN KLIK (9 TITIK):", (460, y_pos + 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    sequence = [
        "1-7: Kerah (Bahu Kiri -> Kanan)",
        "8: Ujung Lengan Kiri",
        "9: Ujung Lengan Kanan"
    ]
    
    seq_y = y_pos + 55
    for seq in sequence:
        cv2.putText(guide, seq, (470, seq_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        seq_y += 30
    
    cv2.putText(guide, "Tekan sembarang key untuk mulai...", (250, 670), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow("Panduan Kerah Lengkung", guide)
    cv2.waitKey(0)
    cv2.destroyWindow("Panduan Kerah Lengkung")


def main():
    global current_image, display_image, points, zoom_level, pan_x, pan_y
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║         JERSEY CURVE ANNOTATOR - 7 POINT COLLAR             ║
║       Anotasi Kerah Lengkung dengan Kurva Smooth            ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Load database lama
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r') as f:
            try:
                database = json.load(f)
            except:
                database = {}
    else:
        database = {}
    
    # Cari semua file jersey
    image_files = glob.glob(os.path.join(ASSET_FOLDER, "*.png"))
    image_files += glob.glob(os.path.join(ASSET_FOLDER, "*.jpg"))
    image_files.sort()
    
    print(f"\nDitemukan {len(image_files)} jersey di: {ASSET_FOLDER}")
    print(f"Database saat ini: {len(database)} jersey sudah dianotasi\n")
    
    # Tampilkan panduan
    show_guide = input("Tampilkan panduan kerah lengkung? [Y/n]: ").strip().lower()
    if show_guide != 'n':
        show_collar_guide()
    
    # Mode
    mode = input("\nMode: [1] Skip yang sudah ada  [2] Re-annotate semua\nPilih [1/2]: ").strip()
    skip_existing = (mode != '2')
    
    mouse_params = {}
    success_count = 0
    skip_count = 0
    
    for img_idx, img_path in enumerate(image_files):
        filename = os.path.basename(img_path)
        
        print(f"\n{'='*60}")
        print(f"[{img_idx + 1}/{len(image_files)}] {filename}")
        print(f"{'='*60}")
        
        # Skip jika sudah ada
        if skip_existing and filename in database:
            print(f"[SKIP] Sudah ada di database.")
            skip_count += 1
            continue
        
        # Reset state
        points = []
        zoom_level = 1.0
        pan_x = 0
        pan_y = 0
        
        # Baca gambar
        current_image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if current_image is None:
            print(f"[ERROR] Gagal membaca: {img_path}")
            continue
        
        # Convert BGRA to BGR
        if current_image.shape[2] == 4:
            alpha = current_image[:, :, 3] / 255.0
            bgr = current_image[:, :, :3]
            white_bg = np.ones_like(bgr) * 255
            current_image = (bgr * alpha[:, :, np.newaxis] + 
                           white_bg * (1 - alpha[:, :, np.newaxis])).astype(np.uint8)
        
        h, w = current_image.shape[:2]
        print(f"Resolusi: {w}x{h}")
        print(f"Klik 9 titik: 7 titik kerah (kiri->kanan) + 2 lengan")
        
        update_display()
        cv2.setMouseCallback("Jersey Curve Annotator", click_event, mouse_params)
        
        # Main loop
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s'):  # SAVE
                if len(points) == 9:
                    # Normalize coordinates
                    norm_points = [[p[0]/w, p[1]/h] for p in points]
                    
                    # Generate smooth collar curve dari 7 titik pertama
                    collar_curve = interpolate_collar_curve(points[:7])
                    
                    # Simpan dengan format yang kompatibel + data kurva
                    database[filename] = {
                        # Format lama (kompatibel dengan processor existing)
                        "neck_left_top": norm_points[0],      # Bahu kiri
                        "neck_bottom": norm_points[3],        # Kerah bawah
                        "neck_right_top": norm_points[6],     # Bahu kanan
                        "sleeve_left": norm_points[7],        # Lengan kiri
                        "sleeve_right": norm_points[8],       # Lengan kanan
                        
                        # Format baru (untuk rendering lengkung)
                        "collar_curve_points": [norm_points[i] for i in range(7)],
                        "collar_curve_smooth": [[pt[0]/w, pt[1]/h] for pt in collar_curve]
                    }
                    
                    # Auto-save
                    with open(OUTPUT_FILE, 'w') as f:
                        json.dump(database, f, indent=4)
                    
                    print(f"[✓] DISIMPAN! (7 titik kerah + 2 lengan)")
                    success_count += 1
                    break
                else:
                    print(f"[!] Belum selesai! Butuh {9 - len(points)} titik lagi.")
            
            elif key == ord('r'):  # RESET
                points = []
                zoom_level = 1.0
                pan_x = 0
                pan_y = 0
                print("[RESET] Semua titik dihapus.")
                update_display()
            
            elif key == ord('u'):  # UNDO
                if len(points) > 0:
                    removed = points.pop()
                    print(f"[UNDO] Titik terakhir dihapus: {removed}")
                    update_display()
            
            elif key == ord('n'):  # NEXT
                print("[SKIP] Dilewati.")
                skip_count += 1
                break
            
            elif key == ord('h'):  # HELP
                show_collar_guide()
                update_display()
            
            elif key == ord('q'):  # QUIT
                cv2.destroyAllWindows()
                print(f"\n{'='*60}")
                print(f"SUMMARY:")
                print(f"  ✓ Berhasil: {success_count}")
                print(f"  ⊘ Dilewati: {skip_count}")
                print(f"  Total: {len(database)} jersey")
                print(f"{'='*60}")
                return
    
    cv2.destroyAllWindows()
    
    print(f"\n{'='*60}")
    print(f"SELESAI!")
    print(f"{'='*60}")
    print(f"✓ Berhasil: {success_count}")
    print(f"⊘ Dilewati: {skip_count}")
    print(f"📁 Total database: {len(database)} jersey")
    print(f"💾 File: {OUTPUT_FILE}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
