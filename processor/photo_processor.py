import cv2
import numpy as np
from rembg import remove
from PIL import Image
import mediapipe as mp

class PhotoProcessor:
    def __init__(self):
        # Inisialisasi MediaPipe Face Mesh
        # Refine landmarks=True penting untuk akurasi mata/bibir
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

    def process_photo(self, user_image, jersey_img=None, meta=None, scales=(0.0, 0.2, 0.1)):
        """
        Input: user_image (OpenCV BGR)
        Output: Image dengan Overlay Merah (Kulit) dan Hijau (Baju)
        """
        try:
            h, w, _ = user_image.shape
            
            # ---------------------------------------------------------
            # 1. BODY SEGMENTATION (Rembg)
            # Ambil siluet seluruh badan
            # ---------------------------------------------------------
            user_rgb = cv2.cvtColor(user_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(user_rgb)
            output = remove(pil_image)
            output_np = np.array(output)
            
            if output_np.shape[2] == 4:
                body_mask = output_np[:, :, 3]
            else:
                gray = cv2.cvtColor(output_np, cv2.COLOR_RGB2GRAY)
                _, body_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
            
            # Pastikan mask biner bersih (0 atau 255)
            _, body_mask = cv2.threshold(body_mask, 127, 255, cv2.THRESH_BINARY)

            # ---------------------------------------------------------
            # 2. FACE DETECTION & SKIN SAMPLING (MediaPipe)
            # Ambil sampel warna kulit dari pipi
            # ---------------------------------------------------------
            results = self.face_mesh.process(user_rgb)
            
            face_mask = np.zeros((h, w), dtype=np.uint8)
            skin_Cr_values = []
            skin_Cb_values = []
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    points = []
                    for lm in face_landmarks.landmark:
                        points.append([int(lm.x * w), int(lm.y * h)])
                    points = np.array(points, dtype=np.int32)
                    
                    # A. Masker Wajah (Area Wajah PASTI Merah)
                    hull = cv2.convexHull(points)
                    cv2.fillConvexPoly(face_mask, hull, 255)
                    
                    # B. Sampling Warna YCrCb (Lebih akurat dari RGB/HSV)
                    # Kita ambil warna dari pipi kiri (idx 234) dan kanan (idx 454)
                    img_ycrcb = cv2.cvtColor(user_image, cv2.COLOR_BGR2YCrCb)
                    
                    for idx in [234, 454]:
                        if idx < len(points):
                            cx, cy = points[idx]
                            # Clip agar koordinat aman
                            cx, cy = np.clip(cx, 0, w-1), np.clip(cy, 0, h-1)
                            
                            # Ambil area 10x10 pixel di pipi
                            roi = img_ycrcb[max(0, cy-5):min(h, cy+5), max(0, cx-5):min(w, cx+5)]
                            if roi.size > 0:
                                # Simpan komponen Warna (Cr, Cb). Abaikan Terang/Gelap (Y)!
                                skin_Cr_values.append(np.mean(roi[:,:,1]))
                                skin_Cb_values.append(np.mean(roi[:,:,2]))

            # ---------------------------------------------------------
            # 3. DETEKSI KULIT BADAN (YCrCb Exclusion)
            # Ini kuncinya: Membedakan leher gelap vs baju hitam
            # ---------------------------------------------------------
            skin_mask_body = np.zeros((h, w), dtype=np.uint8)
            
            if len(skin_Cr_values) > 0:
                avg_Cr = np.mean(skin_Cr_values)
                avg_Cb = np.mean(skin_Cb_values)
                
                # Toleransi. Semakin kecil angka ini, semakin ketat deteksinya (tidak bocor ke baju)
                tol = 10 
                
                # Range Filter: Y (Luminance) kita set 0-255.
                # Artinya: Mau gelap (bayangan leher) atau terang, kalau warnanya (Cr/Cb) cocok, itu KULIT.
                # Baju hitam nilai Cr/Cb-nya beda jauh, jadi aman.
                lower_skin = np.array([0, avg_Cr - tol, avg_Cb - tol], dtype=np.uint8)
                upper_skin = np.array([255, avg_Cr + tol, avg_Cb + tol], dtype=np.uint8)
                
                # Proses deteksi
                img_ycrcb = cv2.cvtColor(user_image, cv2.COLOR_BGR2YCrCb)
                skin_mask_color = cv2.inRange(img_ycrcb, lower_skin, upper_skin)
                
                # Bersihkan noise bintik-bintik
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                skin_mask_color = cv2.morphologyEx(skin_mask_color, cv2.MORPH_OPEN, kernel, iterations=2)
                
                # Kulit Valid = (Warna Cocok) DAN (Ada di Body) DAN (Bukan Wajah)
                skin_mask_body = cv2.bitwise_and(skin_mask_color, body_mask)
                skin_mask_body = cv2.bitwise_and(skin_mask_body, cv2.bitwise_not(face_mask))

            # ---------------------------------------------------------
            # 4. PEMISAHAN FINAL (SHIRT vs SKIN)
            # ---------------------------------------------------------
            
            # Area MERAH (Protected) = Wajah + Kulit Badan
            protected_mask = cv2.bitwise_or(face_mask, skin_mask_body)
            
            # Sedikit dilate di wajah untuk menutup celah leher atas
            face_dilate = cv2.dilate(face_mask, np.ones((15,15), np.uint8), iterations=1)
            protected_mask = cv2.bitwise_or(protected_mask, face_dilate)

            # Area HIJAU (Baju) = Body - Protected
            # Logika: Apapun yang ada di badan TAPI bukan kulit, itu BAJU.
            shirt_mask = cv2.bitwise_and(body_mask, cv2.bitwise_not(protected_mask))
            
            # Rapikan mask baju (hilangkan lubang kecil)
            shirt_mask = cv2.morphologyEx(shirt_mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))

            # ---------------------------------------------------------
            # 5. VISUALISASI (Overlay Merah & Hijau)
            # Menggunakan cv2.addWeighted agar aman dan tidak crash
            # ---------------------------------------------------------
            final_view = user_image.copy()
            
            # Siapkan Canvas Warna
            green_canvas = np.zeros_like(user_image); green_canvas[:] = [0, 255, 0]
            red_canvas = np.zeros_like(user_image); red_canvas[:] = [0, 0, 255]
            
            # Konversi Mask ke 3 Channel
            prot_mask_3ch = cv2.cvtColor(protected_mask, cv2.COLOR_GRAY2BGR)
            shirt_mask_3ch = cv2.cvtColor(shirt_mask, cv2.COLOR_GRAY2BGR)
            
            # A. Tempel Overlay MERAH (Kulit) - Opacity 60%
            red_part = cv2.bitwise_and(red_canvas, prot_mask_3ch)
            orig_red = cv2.bitwise_and(final_view, prot_mask_3ch)
            blended_red = cv2.addWeighted(red_part, 0.6, orig_red, 0.4, 0)
            
            # B. Tempel Overlay HIJAU (Baju) - Opacity 60%
            green_part = cv2.bitwise_and(green_canvas, shirt_mask_3ch)
            orig_green = cv2.bitwise_and(final_view, shirt_mask_3ch)
            blended_green = cv2.addWeighted(green_part, 0.6, orig_green, 0.4, 0)
            
            # C. Gabungkan ke Gambar Asli
            # Hapus area mask dari gambar asli dulu
            total_mask = cv2.bitwise_or(prot_mask_3ch, shirt_mask_3ch)
            final_view = cv2.bitwise_and(final_view, cv2.bitwise_not(total_mask))
            
            # Masukkan hasil blend
            final_view = cv2.add(final_view, blended_red)
            final_view = cv2.add(final_view, blended_green)

            return final_view

        except Exception as e:
            print(f"Error processing photo: {e}")
            # Kembalikan gambar asli jika error, supaya aplikasi tidak crash
            return user_image