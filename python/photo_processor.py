"""
Photo Processor
Untuk upload foto (fokus pada kualitas - REPLACE BAJU dengan jersey)
"""

import cv2
import numpy as np
import mediapipe as mp

class PhotoProcessor:
    def __init__(self):
        # High quality MediaPipe settings
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        
    def process_photo(self, user_image, jersey_img, meta=None, scales=(0.0, 0.2, 0.1), debug=True):
        """
        Process foto statis - REPLACE BAJU dengan jersey texture
        
        Args:
            user_image: Foto user (BGR)
            jersey_img: Jersey dengan alpha (BGRA, pre-warped)
            meta: Metadata anchor points
            scales: (scale_x, scale_neck_up, scale_length_down)
            debug: If True, tampilkan debug visualization
        
        Returns:
            Foto dengan baju di-REPLACE oleh jersey
        """
        try:
            print("\n" + "="*70)
            print("  PHOTO PROCESSOR - DEBUG MODE")
            print("="*70)
            
            _, scale_y_up, scale_y_down = scales
            h, w, _ = user_image.shape
            
            if jersey_img is None:
                return user_image
            
            jh, jw, _ = jersey_img.shape
            
            # ================================================================
            # STEP 1: MediaPipe Processing (High Quality Mode)
            # ================================================================
            pose_static = self.mp_pose.Pose(
                static_image_mode=True,
                model_complexity=2,
                enable_segmentation=True,
                min_detection_confidence=0.7
            )
            
            hands_detector = self.mp_hands.Hands(
                static_image_mode=True,
                max_num_hands=2,
                min_detection_confidence=0.7
            )
            
            user_rgb = cv2.cvtColor(user_image, cv2.COLOR_BGR2RGB)
            user_rgb.flags.writeable = False
            
            pose_results = pose_static.process(user_rgb)
            hands_results = hands_detector.process(user_rgb)
            
            user_rgb.flags.writeable = True
            
            if not pose_results.pose_landmarks or pose_results.segmentation_mask is None:
                print("[PHOTO] No pose detected!")
                return user_image
            
            lm = pose_results.pose_landmarks.landmark
            body_mask = pose_results.segmentation_mask
            
            # ================================================================
            # STEP 2: Extract Landmarks & Calculate Dimensions
            # ================================================================
            ls_pt = lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            rs_pt = lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            lh_pt = lm[self.mp_pose.PoseLandmark.LEFT_HIP]
            rh_pt = lm[self.mp_pose.PoseLandmark.RIGHT_HIP]
            
            # Arm/hand landmarks for masking
            left_elbow = lm[self.mp_pose.PoseLandmark.LEFT_ELBOW]
            right_elbow = lm[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
            left_wrist = lm[self.mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = lm[self.mp_pose.PoseLandmark.RIGHT_WRIST]
            
            if ls_pt.visibility < 0.5 or rs_pt.visibility < 0.5:
                return user_image
            
            ls = np.array([int(ls_pt.x * w), int(ls_pt.y * h)])
            rs = np.array([int(rs_pt.x * w), int(rs_pt.y * h)])
            
            print(f"[1] LANDMARKS DETECTED:")
            print(f"    - Left Shoulder:  ({ls[0]}, {ls[1]}) visibility={ls_pt.visibility:.2f}")
            print(f"    - Right Shoulder: ({rs[0]}, {rs[1]}) visibility={rs_pt.visibility:.2f}")
            
            if lh_pt.visibility > 0.5 and rh_pt.visibility > 0.5:
                hip_y_ref = int((lh_pt.y * h + rh_pt.y * h) / 2)
                print(f"    - Hips detected at y={hip_y_ref}")
            else:
                hip_y_ref = int(ls[1] + (w * 0.5))
                print(f"    - Hips NOT detected, using virtual y={hip_y_ref}")
            
            # ================================================================
            #rint(f"\n[2] TORSO WIDTH DETECTION:")
            print(f"    - Body left edge:  {body_left_x}px")
            print(f"    - Body right edge: {body_right_x}px")
            print(f"    - Detected width:  {body_width_px}px")
            
            padding_fit = int(body_width_px * 0.08)
            final_left_x = body_left_x - padding_fit
            final_right_x = body_right_x + padding_fit
            final_jersey_width = final_right_x - final_left_x
            
            print(f"    - With padding: {final_jersey_width}px (added {padding_fit}px each side)")
            
            pad_y_up = int(final_jersey_width * scale_y_up)
            
            final_ls = [final_right_x, ls[1] - pad_y_up]
            final_rs = [final_left_x, rs[1] - pad_y_up]
            
            print(f"\n[3] JERSEY CORNERS (after scaling):")
                print(f"    - Bottom-Right (Right Hip): {final_rh}")
                print(f"    - Bottom-Left (Left Hip):   {final_lh}")
            else:
                center_y = (ls[1] + rs[1]) / 2
                virtual_jersey_height = final_jersey_width * 1.4
                virtual_hip_y = center_y + virtual_jersey_height
                final_rh = [final_left_x, virtual_hip_y]
                final_lh = [final_right_x, virtual_hip_y]
                print(f"    - Bottom-Right (Virtual): {final_rh}")
                print(f"    - Bottom-Left (Virtual):  {final_lh}")
            
            # Calculate jersey dimensions
            jersey_height = int(max(final_lh[1], final_rh[1]) - min(final_ls[1], final_rs[1]))
            print(f"\n[4] JERSEY FINAL DIMENSIONS:")
            print(f"    - Width:  {final_jersey_width}px")
            print(f"    - Height: {jersey_height}px")
            print(f"\n[5] PERSPECTIVE TRANSFORM:")
            print(f"    - Source jersey: {jw}x{jh}px")
            
            src_pts = np.float32([[0, 0], [jw, 0], [jw, jh], [0, jh]])
            dst_pts = np.float32([final_rs, final_ls, final_lh, final_rh])
            
            print(f"    - Mapping: [{0},{0}] → {final_rs}")
            print(f"    - Mapping: [{jw},{0}] → {final_ls}")
            print(f"    - Mapping: [{jw},{jh}] → {final_lh}")
            print(f"    - Mapping: [{0},{jh}] → {final_rh}")
            
            final_rs = [final_left_x, rs[1] - pad_y_up]
            
            hips_visible = (lh_pt.visibility > 0.5) and (rh_pt.visibility > 0.5)
            
            if hips_visible:
                rh = np.array([int(rh_pt.x * w), int(rh_pt.y * h)])
                lh = np.array([int(lh_pt.x * w), int(lh_pt.y * h)])
                pad_y_down = int(final_jersey_width * scale_y_down)
                final_rh = [final_left_x, rh[1] + pad_y_down]
                final_lh = [final_right_x, lh[1] + pad_y_down]
            else:
                center_y = (ls[1] + rs[1]) / 2
                virtual_jersey_height = final_jersey_width * 1.4
                virtual_hip_y = center_y + virtual_jersey_height
                final_rh = [final_left_x, virtual_hip_y]
                final_lh = [final_right_x, virtual_hip_y]
            
            # ================================================================
            # STEP 4: Warp Jersey with Collar
            # ================================================================
            src_pts = np.float32([[0, 0], [jw, 0], [jw, jh], [0, jh]])
            dst_pts = np.float32([final_rs, final_ls, final_lh, final_rh])
            matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
            
            if meta is not None:
                jersey_working = jersey_img.copy()
                
                p1 = meta["neck_left_top"]
                p2 = meta["neck_bottom"]
                p3 = meta["neck_right_top"]
                
                pt1 = (int(p1[0] * jw), int(p1[1] * jh))
                pt2 = (int(p2[0] * jw), int(p2[1] * jh))
                pt3 = (int(p3[0] * jw), int(p3[1] * jh))
                
                back_collar_center = (int((pt1[0] + pt3[0])/2), pt1[1])
                back_collar_axes = (int(abs(pt1[0]-pt3[0])/2), int(abs(pt1[1]-pt2[1])/2))
                cv2.ellipse(jersey_working, back_collar_center, back_collar_axes, 0, 180, 360, (30, 30, 30, 255), -1)
                
                top_y_limit = -50
                pts_neck = np.array([pt1, pt2, pt3, (pt3[0], top_y_limit), (pt1[0], top_y_limit)], np.int32)
                cv2.fillPoly(jersey_working, [pts_neck], (0,0,0,0))
                
                warped_jersey = cv2.warpPerspective(jersey_working, matrix, (w, h))
            else:
                warped_jersey = cv2.warpPerspective(jersey_img, matrix, (w, h))
                
                neck_cx = int((final_ls[0] + final_rs[0]) / 2)
                neck_cy = int((final_ls[1] + final_rs[1]) / 2)
                nw = int(final_jersey_width * 0.18)
                nh = int(final_jersey_width * 0.12)
                
                cv2.ellipse(warped_jersey, (neck_cx, neck_cy - int(nh*0.5)), (nw, int(nh*0.8)), 0, 180, 360, (40,40,40,200), -1)
                cv2.ellipse(warped_jersey, (neck_cx, neck_cy), (nw, nh), 0, 0, 180, (0,0,0,0), -1)
            
            # ================================================================
            # STEP 5: CREATE TORSO MASK - GUNAKAN SEGMENTATION DARI MEDIAPIPE
            # ================================================================
            # PENTING: Gunakan body_mask dari MediaPipe (REAL body shape)
            # Konversi ke binary mask
            body_segmentation = (body_mask > 0.5).astype(np.uint8) * 255
            
            # Fokus pada TORSO area (shoulder ke hip)
            torso_mask = np.zeros((h, w), dtype=np.uint8)
            
            # Define torso region (exclude head and lower body)
            torso_top = int(min(final_ls[1], final_rs[1]) * 0.9)  # Sedikit di atas shoulder
            torso_bottom = int(max(final_lh[1], final_rh[1]) * 1.05)  # Sedikit di bawah hip
            
            # Copy body segmentation hanya di area torso
            torso_mask[torso_top:torso_bottom, :] = body_segmentation[torso_top:torso_bottom, :]
            
            # Clean up dengan morphology
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
            torso_mask = cv2.morphologyEx(torso_mask, cv2.MORPH_CLOSE, kernel_open)
            
            print(f"[PHOTO] Torso mask created from segmentation: {np.sum(torso_mask > 0)} pixels")
            
            # ================================================================
            # STEP 6: CREATE SKIN MASK - PRESERVE TANGAN, LEHER, WAJAH
            # ================================================================
            skin_mask = np.zeros((h, w), dtype=np.uint8)
            
            # A. Deteksi lengan/siku (PRESERVE - jangan replace dengan jersey)
            arm_radius = int(final_jersey_width * 0.12)
            
            arm_landmarks = [
                (left_elbow, "LEFT_ELBOW"),
                (right_elbow, "RIGHT_ELBOW"),
                (left_wrist, "LEFT_WRIST"),
                (right_wrist, "RIGHT_WRIST")
            ]
            
            for landmark, name in arm_landmarks:
                if landmark.visibility > 0.5:
                    lm_x = int(landmark.x * w)
                    lm_y = int(landmark.y * h)
                    cv2.circle(skin_mask, (lm_x, lm_y), arm_radius, 255, -1)
                    print(f"[PHOTO] Preserving {name} at ({lm_x}, {lm_y})")
            
            # B. Deteksi tangan menggunakan MediaPipe Hands
            if hands_results.multi_hand_landmarks:
                for hand_landmarks in hands_results.multi_hand_landmarks:
                    hand_points = []
                    for point in hand_landmarks.landmark:
                        px = int(point.x * w)
                        py = int(point.y * h)
                        hand_points.append([px, py])
                    
                    hand_points = np.array(hand_points, dtype=np.int32)
                    hull = cv2.convexHull(hand_points)
                    cv2.fillConvexPoly(skin_mask, hull, 255)
                    print(f"[PHOTO] Hand preserved at area")
            
            # Smooth skin mask
            if skin_mask.max() > 0:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
                skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)
                skin_mask = cv2.GaussianBlur(skin_mask, (21, 21), 0)
            
            # ================================================================
            # STEP 7: REPLACE BAJU dengan JERSEY - 100% REPLACE, BUKAN BLEND
            # ================================================================
            # Jersey mask = Torso mask MINUS skin mask
            jersey_replace_mask = cv2.bitwise_and(torso_mask, cv2.bitwise_not(skin_mask))
            
            # Also use jersey's alpha channel
            jersey_alpha = warped_jersey[:, :, 3]
            combined_mask = cv2.bitwise_and(jersey_replace_mask, jersey_alpha)
            
            # Smooth edges untuk transisi natural
            combined_mask = cv2.GaussianBlur(combined_mask, (7, 7), 0)
            mask_alpha = combined_mask.astype(float) / 255.0
            mask_alpha_3d = np.stack([mask_alpha] * 3, axis=-1)
            
            # ================================================================
            # ALGORITMA BARU: FULL REPLACE dengan Lighting Adaptation
            # ================================================================
            result = user_image.copy().astype(float)
            jersey_bgr = warped_jersey[:, :, :3].astype(float)
            
            # Step 1: Extract lighting dari foto asli (hanya intensitas)
            user_hsv = cv2.cvtColor(user_image, cv2.COLOR_BGR2HSV).astype(float)
            user_value = user_hsv[:, :, 2] / 255.0  # Value channel (brightness)
            
            # Step 2: Convert jersey ke HSV
            jersey_uint8 = jersey_bgr.astype(np.uint8)
            jersey_hsv = cv2.cvtColor(jersey_uint8, cv2.COLOR_BGR2HSV).astype(float)
            
            # ================================================================
            # DEBUG VISUALIZATION (if enabled)
            # ================================================================
            if debug:
                debug_img = user_image.copy()
                
                # Draw jersey corners
                cv2.circle(debug_img, tuple(map(int, final_rs)), 10, (0, 0, 255), -1)  # Red: Right shoulder
                cv2.circle(debug_img, tuple(map(int, final_ls)), 10, (0, 255, 0), -1)  # Green: Left shoulder
                cv2.circle(debug_img, tuple(map(int, final_rh)), 10, (255, 0, 0), -1)  # Blue: Right hip
                cv2.circle(debug_img, tuple(map(int, final_lh)), 10, (255, 255, 0), -1)  # Cyan: Left hip
                
                # Draw jersey boundary
                pts = np.array([final_rs, final_ls, final_lh, final_rh], dtype=np.int32)
                cv2.polylines(debug_img, [pts], True, (0, 255, 255), 3)
                
                # Draw landmarks
                cv2.circle(debug_img, tuple(ls), 8, (255, 0, 255), -1)  # Original shoulders
                cv2.circle(debug_img, tuple(rs), 8, (255, 0, 255), -1)
                
                # Save debug image
                cv2.imwrite('debug_jersey_position.jpg', debug_img)
                print(f"\n[DEBUG] Saved: debug_jersey_position.jpg")
                print(f"        Cek posisi 4 corner jersey (merah, hijau, biru, cyan)")
            
            # Cleanup
            pose_static.close()
            hands_detector.close()
            
            print(f"\n[RESULT] ✓ Processing completed!")
            print(f"         Jersey mask: {np.sum(combined_mask > 0)} pixels")
            print(f"         Skin preserved: {np.sum(skin_mask > 0)} pixels")
            print("="*70 + "\n
            # Step 4: FULL REPLACE - 100% jersey di area mask, 0% baju lama
            result = jersey_adapted * mask_alpha_3d + result * (1.0 - mask_alpha_3d)
            
            result = np.clip(result, 0, 255).astype(np.uint8)
            
            print(f"[PHOTO] REPLACE completed - Jersey area: {np.sum(combined_mask > 0)} pixels")
            
            # ================================================================
            # STEP 8: Post-processing - Minimal Sharpening
            # ================================================================
            # Sharpen sedikit saja untuk detail, jangan berlebihan
            kernel_sharpen = np.array([
                [0, -0.2, 0],
                [-0.2, 1.8, -0.2],
                [0, -0.2, 0]
            ])
            
            sharpened = cv2.filter2D(result, -1, kernel_sharpen)
            
            # Apply sharpening only to jersey area (very subtle)
            mask_for_sharpen = np.stack([mask_alpha] * 3, axis=-1)
            result = (sharpened * mask_for_sharpen * 0.3 + 
                     result * (1.0 - mask_for_sharpen * 0.3)).astype(np.uint8)
            
            # Cleanup
            pose_static.close()
            hands_detector.close()
            
            print(f"[PHOTO] ✓ REPLACE processing completed!")
            print(f"[PHOTO] Jersey area: {np.sum(combined_mask > 0)} pixels")
            print(f"[PHOTO] Skin preserved: {np.sum(skin_mask > 0)} pixels")
            
            return result
            
        except Exception as e:
            print(f"[PHOTO ERROR]: {e}")
            import traceback
            traceback.print_exc()
            return user_image
    
    def _get_torso_width_smart(self, body_mask, shoulder_y, hip_y, center_x, w):
        """Deteksi lebar torso dari segmentation mask"""
        y1 = max(0, int(shoulder_y))
        y2 = min(body_mask.shape[0], int(hip_y))
        
        torso_mask_slice = body_mask[y1:y2, :]
        
        if torso_mask_slice.size == 0:
            fallback_width = int(w * 0.25)
            return center_x - fallback_width, center_x + fallback_width
        
        torso_binary = (torso_mask_slice > 0.5).astype(np.uint8)
        horizontal_sum = np.sum(torso_binary, axis=0)
        
        if np.max(horizontal_sum) == 0:
            fallback_width = int(w * 0.25)
            return center_x - fallback_width, center_x + fallback_width
        
        threshold = np.max(horizontal_sum) * 0.3
        body_x_indices = np.where(horizontal_sum > threshold)[0]
        
        if len(body_x_indices) == 0:
            fallback_width = int(w * 0.25)
            return center_x - fallback_width, center_x + fallback_width
        
        body_left = int(np.min(body_x_indices))
        body_right = int(np.max(body_x_indices))
        
        return body_left, body_right
