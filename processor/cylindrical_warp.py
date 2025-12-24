"""
Cylindrical Warp Utility
Untuk membuat jersey terlihat melengkung seperti menempel pada tubuh silinder
"""

import cv2
import numpy as np

def cylindrical_warp(img, focal_length=None, strength=1.0):
    """
    Melengkungkan gambar seolah-olah menempel pada tabung (Pseudo-3D Effect).
    
    Args:
        img: Input image (BGRA atau BGR)
        focal_length: Focal length untuk mengontrol derajat lengkungan.
                      None = auto (width * 0.9)
                      Lebih kecil = lebih melengkung
        strength: Multiplier untuk efek (0.0-1.0)
                  1.0 = full effect, 0.5 = subtle effect
    
    Returns:
        Warped image dengan efek cylindrical
    """
    if img is None:
        return None
    
    height, width = img.shape[:2]
    
    # 1. Tentukan Titik Pusat & Focal Length
    # Focal length menentukan seberapa 'cembung' lengkungannya.
    # Semakin kecil focal length = semakin melengkung ekstrem.
    if focal_length is None:
        focal_length = width * 0.9  # Default: subtle curve
    
    # Apply strength multiplier
    focal_length = focal_length / strength if strength > 0 else focal_length
    
    # 2. Membuat Grid Koordinat (Vectorization - Langkah Kunci!)
    # Kita buat array koordinat X dan Y sekaligus
    map_x, map_y = np.meshgrid(np.arange(width), np.arange(height))
    
    # 3. Matematika Silinder (Tanpa Loop for!)
    # Ubah koordinat pixel (0 s/d width) menjadi koordinat kartesius (-width/2 s/d width/2)
    # agar hitungannya dimulai dari tengah gambar.
    x_centered = map_x - (width / 2.0)
    
    # RUMUS INTI:
    # Memetakan layar datar kembali ke permukaan silinder
    # Z adalah kedalaman (jarak maya pixel ke mata)
    z = np.sqrt(focal_length**2 + x_centered**2)
    
    # Hitung koordinat X asli (sumber) yang harus diambil
    # Ini mensimulasikan pembiasan/lengkungan
    map_x_source = (focal_length * x_centered / z) + (width / 2.0)
    
    # Y tidak berubah (karena silinder tegak lurus)
    map_y_source = map_y.astype(np.float32)
    
    # 4. Konversi ke float32 (required by cv2.remap)
    map_x_source = map_x_source.astype(np.float32)
    
    # 5. Eksekusi Warping dengan cv2.remap
    # INTER_LINEAR membuat hasil halus
    # BORDER_CONSTANT agar pinggiran yang kosong jadi transparan
    border_value = (0, 0, 0, 0) if img.shape[2] == 4 else (0, 0, 0)
    
    result = cv2.remap(
        img, 
        map_x_source, 
        map_y_source,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value
    )
    
    return result


def apply_cylindrical_to_jersey(jersey_img, strength=0.7):
    """
    Wrapper function khusus untuk jersey dengan parameter optimal.
    
    Args:
        jersey_img: Jersey image (BGRA)
        strength: Tingkat lengkungan (0.5=subtle, 1.0=strong)
    
    Returns:
        Warped jersey
    """
    if jersey_img is None:
        return None
    
    # Apply cylindrical warp dengan parameter optimal untuk jersey
    warped = cylindrical_warp(jersey_img, focal_length=None, strength=strength)
    
    return warped
