"""
Photo Processor - FINAL POLISH (Dilate & Overscan)
Perbaikan untuk menutup celah baju lama yang masih terlihat.
Strategi:
1. Perlebar titik target tubuh (agar jersey sedikit oversize).
2. Dilate (pertebal) masker baju lama untuk penghapusan agresif.
3. Komposit layer dengan strategi "Over-Paste".
"""

import cv2
import numpy as np
import json
import os
from rembg import remove
from PIL import Image
import torch
import torch.nn as nn
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
import mediapipe as mp

class PhotoProcessor:
    def __init__(self):
        print("[PHOTO] Initializing Final Polish Processor...")
        # (Setup model sama seperti sebelumnya...)
        MODEL_NAME = "matei-dorian/segformer-b5-finetuned-human-parsing"
        try:
            self.processor = SegformerImageProcessor.from_pretrained(MODEL_NAME)
            self.model = AutoModelForSemanticSegmentation.from_pretrained(MODEL_NAME)
            self.model.eval()
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
        except Exception as e: print(f"[ERROR] {e}"); self.processor=None
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=True, model_complexity=2)
        self.jersey_landmarks = {}
        if os.path.exists("jersey_landmarks.json"):
            with open("jersey_landmarks.json", 'r') as f: self.jersey_landmarks = json.load(f)

    def get_body_landmarks(self, image_rgb):
        h, w, _ = image_rgb.shape
        results = self.pose.process(image_rgb)
        
        if not results.pose_landmarks: return None
        
        lm = results.pose_landmarks.landmark
        def to_px(l): return np.array([l.x * w, l.y * h])

        # Ambil titik murni (Tulang)
        # 11=Left Shoulder (Kanan di Layar), 12=Right Shoulder (Kiri di Layar)
        sh_left = to_px(lm[11]) 
        sh_right = to_px(lm[12])
        elb_left = to_px(lm[13])
        elb_right = to_px(lm[14])
        hip_left = to_px(lm[23])
        hip_right = to_px(lm[24])

        # --- [LOGIC UPDATE] MANUAL OFFSET (GESER TITIK) ---
        
        # Hitung vektor lebar bahu (jarak horizontal & vertikal)
        vec_sh = sh_right - sh_left   # Vektor dari Kiri ke Kanan (User perspective)
        shoulder_width = np.linalg.norm(vec_sh)
        
        # 1. EXPANSION (Melebarkan ke Samping)
        # Kita geser keluar sebesar 25% dari lebar bahu
        EXPAND_X = 0.25 
        
        # 2. LIFT (Mengangkat ke Atas)
        # Jersey duduk di atas otot trapezius, bukan di sendi tulang.
        # Kita angkat sebesar 12% dari lebar bahu.
        LIFT_Y = 0.12 
        
        # Offset vektor untuk X (Samping)
        offset_x_vec = vec_sh * EXPAND_X
        
        # Offset vektor untuk Y (Atas Murni - Tegak Lurus)
        # Kita buat vektor (0, -1) dikali skala bahu
        offset_y_val = np.array([0, -shoulder_width * LIFT_Y])

        # --- TERAPKAN KE BAHU ---
        # Bahu Kiri (User Left / Screen Right): Geser ke Kanan(Samping) + Atas
        # Karena vec_sh arahnya ke kanan (screen left), kita kurangi untuk geser ke kanan screen (screen right?? Wait..)
        # Cek arah: sh_right(Screen Left) minus sh_left(Screen Right) = Vektor arah ke Kiri Screen (-X)
        
        # Bahu Kiri (Screen Right/High X): Kita mau geser ke Kanan Screen (Lebih High X)
        # Maka: sh_left - (vec_sh * 0.25) -> HighX - (-Value) = HighX + Value (Makin Kanan) -> OK
        sh_l_target = sh_left - offset_x_vec + offset_y_val
        
        # Bahu Kanan (Screen Left/Low X): Kita mau geser ke Kiri Screen (Lebih Low X)
        # Maka: sh_right + (vec_sh * 0.25) -> LowX + (-Value) = LowX - Value (Makin Kiri) -> OK
        sh_r_target = sh_right + offset_x_vec + offset_y_val
        
        # --- TERAPKAN KE LEHER ---
        # Leher juga perlu naik sedikit biar kerah tidak turun
        neck_l = sh_left + (vec_sh * 0.30) + (offset_y_val * 0.5) # Naik dikit (50% dari bahu)
        neck_r = sh_left + (vec_sh * 0.70) + (offset_y_val * 0.5)

        # --- TERAPKAN KE LENGAN ---
        # Lengan jersey biasanya lebih lebar dari tangan asli
        # Kita ambil posisi 65% menuju siku, lalu geser keluar (tegak lurus lengan)
        
        # Fungsi geser tegak lurus (perpendicular)
        def get_perp_outward(p_start, p_end, scale):
            v = p_end - p_start
            # Putar 90 derajat
            perp = np.array([-v[1], v[0]]) 
            norm = np.linalg.norm(perp)
            if norm == 0: return np.array([0,0])
            return (perp / norm) * (np.linalg.norm(v) * scale)

        # Lengan Kiri
        sleeve_l_base = sh_left + (elb_left - sh_left) * 0.65
        sleeve_l = sleeve_l_base + get_perp_outward(sh_left, elb_left, 0.2) # Geser keluar 20%
        
        # Lengan Kanan
        sleeve_r_base = sh_right + (elb_right - sh_right) * 0.65
        sleeve_r = sleeve_r_base - get_perp_outward(sh_right, elb_right, 0.2) # Arah sebaliknya

        # --- TERAPKAN KE PINGGANG ---
        # Pinggang geser keluar 15% dan Turun 5% (biar baju agak panjang)
        vec_hip = hip_right - hip_left
        hip_l_target = hip_left - (vec_hip * 0.15) + np.array([0, h*0.05])
        hip_r_target = hip_right + (vec_hip * 0.15) + np.array([0, h*0.05])

        # Return 8 Titik
        body_points = np.array([
            sh_l_target, neck_l, neck_r, sh_r_target,
            sleeve_l, sleeve_r, hip_l_target, hip_r_target
        ], dtype=np.float32)

        return body_points
    
    def warp_triangle(self, img1, img2, t1, t2):
        # (Fungsi ini sama persis seperti sebelumnya, tidak ada perubahan)
        r1 = cv2.boundingRect(np.float32([t1]))
        r2 = cv2.boundingRect(np.float32([t2]))
        t1_rect = []; t2_rect = []; t2_rect_int = []
        for i in range(0, 3):
            t1_rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
            t2_rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
            t2_rect_int.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
        img1_crop = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
        warp_mat = cv2.getAffineTransform(np.float32(t1_rect), np.float32(t2_rect))
        img2_crop = cv2.warpAffine(img1_crop, warp_mat, (r2[2], r2[3]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
        cv2.fillConvexPoly(mask, np.int32(t2_rect_int), (1.0, 1.0, 1.0), 16, 0)
        img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * (1 - mask) + img2_crop * mask

    # ... (kode inisialisasi __init__ tetap sama) ...

    # [MODIFIKASI] Tambahkan parameter 'custom_body_points'
    def process_photo(self, user_image, jersey_img=None, meta=None, custom_body_points=None):
        if self.model is None or jersey_img is None: return user_image
        h, w, _ = user_image.shape
        
        # 1. PARSING (Tetap jalan untuk masking)
        # ... (Kode Step 1 Parsing & Masking TETAP SAMA seperti sebelumnya) ...
        # (Copy paste bagian Step 1 dari kode terakhir Anda)
        pil_img = Image.fromarray(cv2.cvtColor(user_image, cv2.COLOR_BGR2RGB))
        rembg_out = remove(pil_img)
        body_mask = (np.array(rembg_out)[:,:,3] > 127).astype(np.uint8) * 255
        
        inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
        with torch.no_grad(): out = self.model(**inputs)
        logits = nn.functional.interpolate(out.logits.cpu(), size=(h,w), mode="bilinear", align_corners=False)
        seg = logits.argmax(dim=1)[0].numpy()

        shirt_mask_precise = np.isin(seg, [4, 7]).astype(np.uint8) * 255 
        shirt_mask_precise = cv2.bitwise_and(shirt_mask_precise, body_mask)
        kernel = np.ones((5, 5), np.uint8)
        shirt_mask_dilated = cv2.dilate(shirt_mask_precise, kernel, iterations=2)
        skin_mask = np.isin(seg, [1, 2, 11, 14, 15]).astype(np.uint8) * 255

        # 2. BODY LANDMARKS (LOGIKA BARU)
        if custom_body_points is not None:
            print("[INFO] Menggunakan Custom Landmarks dari Web!")
            # Konversi list dari JSON ke Numpy Array
            body_points = np.array(custom_body_points, dtype=np.float32)
        else:
            print("[INFO] Menggunakan Auto-Detect MediaPipe...")
            body_points = self.get_body_landmarks(cv2.cvtColor(user_image, cv2.COLOR_BGR2RGB))

        # 3. WARPING & COMPOSITING (Tetap Sama)
        # Ambil landmark jersey
        jersey_name = meta['fileName'] if meta else "Unknown"
        j_points = None
        for k, v in self.jersey_landmarks.items():
            if k in jersey_name or jersey_name in k:
                j_points = np.array(v, dtype=np.float32); break
        
        warped_jersey = np.zeros_like(user_image, dtype=np.uint8)

        if j_points is not None and body_points is not None:
            if jersey_img.shape[2] == 4: j_src = cv2.cvtColor(jersey_img, cv2.COLOR_BGRA2BGR)
            else: j_src = jersey_img
            indices = [[0, 1, 2], [0, 2, 3], [0, 4, 6], [0, 6, 1], [3, 5, 7], [3, 7, 2], [1, 6, 7], [1, 7, 2]]
            for ind in indices:
                # Safety check agar tidak crash jika poin manual ngaco
                try:
                    self.warp_triangle(j_src, warped_jersey, np.array([j_points[i] for i in ind]), np.array([body_points[i] for i in ind]))
                except Exception as e:
                    print(f"[WARN] Gagal warp segitiga: {e}")
        
        # ... (Kode Compositing Step 4 TETAP SAMA) ...
        # (Copy paste bagian Step 4 Compositing Over-Paste dari kode terakhir)
        
        bg_mask_inv = cv2.bitwise_not(shirt_mask_dilated)
        bg_clean = cv2.bitwise_and(user_image, user_image, mask=bg_mask_inv)
        final_jersey_cut = cv2.bitwise_and(warped_jersey, warped_jersey, mask=shirt_mask_precise)
        result_step1 = cv2.add(bg_clean, final_jersey_cut)
        skin_part = cv2.bitwise_and(user_image, user_image, mask=skin_mask)
        skin_mask_inv = cv2.bitwise_not(skin_mask)
        result_final = cv2.bitwise_and(result_step1, result_step1, mask=skin_mask_inv)
        result_final = cv2.add(result_final, skin_part)

        return result_final