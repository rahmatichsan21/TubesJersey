"""
Live Stream Processor
Untuk video streaming real-time (fokus pada kecepatan)
"""

import cv2
import numpy as np
import mediapipe as mp

class LiveStreamProcessor:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def process_frame(self, user_frame, jersey_img, meta=None, scales=(0.0, 0.2, 0.1)):
        """
        Process single video frame with jersey overlay (FAST)
        
        Args:
            user_frame: Frame dari webcam (BGR)
            jersey_img: Jersey dengan alpha (BGRA, pre-warped)
            meta: Metadata anchor points untuk collar handling
            scales: (scale_x, scale_neck_up, scale_length_down)
        
        Returns:
            Processed frame dengan jersey overlay
        """
        try:
            _, scale_y_up, scale_y_down = scales
            h, w, _ = user_frame.shape
            
            if jersey_img is None:
                return user_frame
            
            jh, jw, _ = jersey_img.shape
            
            # MediaPipe Processing
            user_img_rgb = cv2.cvtColor(user_frame, cv2.COLOR_BGR2RGB)
            user_img_rgb.flags.writeable = False
            results = self.pose.process(user_img_rgb)
            user_img_rgb.flags.writeable = True
            
            if not results.pose_landmarks or results.segmentation_mask is None:
                return user_frame
            
            lm = results.pose_landmarks.landmark
            body_mask = results.segmentation_mask
            
            # Get landmarks
            ls_pt = lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            rs_pt = lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            lh_pt = lm[self.mp_pose.PoseLandmark.LEFT_HIP]
            rh_pt = lm[self.mp_pose.PoseLandmark.RIGHT_HIP]
            
            if ls_pt.visibility < 0.5 or rs_pt.visibility < 0.5:
                return user_frame
            
            ls = np.array([int(ls_pt.x * w), int(ls_pt.y * h)])
            rs = np.array([int(rs_pt.x * w), int(rs_pt.y * h)])
            
            # Hip reference
            if lh_pt.visibility > 0.5 and rh_pt.visibility > 0.5:
                hip_y_ref = int((lh_pt.y * h + rh_pt.y * h) / 2)
            else:
                hip_y_ref = int(ls[1] + (w * 0.5))
            
            # Smart torso width detection
            center_x = int((ls[0] + rs[0]) / 2)
            body_left_x, body_right_x = self._get_torso_width_smart(body_mask, ls[1], hip_y_ref, center_x, w)
            body_width_px = body_right_x - body_left_x
            
            padding_fit = int(body_width_px * 0.06)
            final_left_x = body_left_x - padding_fit
            final_right_x = body_right_x + padding_fit
            final_jersey_width = final_right_x - final_left_x
            
            pad_y_up = int(final_jersey_width * scale_y_up)
            
            # Shoulder coordinates
            final_ls = [final_right_x, ls[1] - pad_y_up]
            final_rs = [final_left_x, rs[1] - pad_y_up]
            
            # Bottom calculation
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
            
            # Warp jersey
            src_pts = np.float32([[0, 0], [jw, 0], [jw, jh], [0, jh]])
            dst_pts = np.float32([final_rs, final_ls, final_lh, final_rh])
            matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
            
            if meta is not None:
                jersey_working = jersey_img.copy()
                
                # Collar handling
                p1 = meta["neck_left_top"]
                p2 = meta["neck_bottom"]
                p3 = meta["neck_right_top"]
                
                pt1 = (int(p1[0] * jw), int(p1[1] * jh))
                pt2 = (int(p2[0] * jw), int(p2[1] * jh))
                pt3 = (int(p3[0] * jw), int(p3[1] * jh))
                
                # Back collar shadow
                back_collar_center = (int((pt1[0] + pt3[0])/2), pt1[1])
                back_collar_axes = (int(abs(pt1[0]-pt3[0])/2), int(abs(pt1[1]-pt2[1])/2))
                cv2.ellipse(jersey_working, back_collar_center, back_collar_axes, 0, 180, 360, (30, 30, 30, 255), -1)
                
                # Cut front collar
                top_y_limit = -50
                pts_neck = np.array([pt1, pt2, pt3, (pt3[0], top_y_limit), (pt1[0], top_y_limit)], np.int32)
                cv2.fillPoly(jersey_working, [pts_neck], (0,0,0,0))
                
                warped_jersey = cv2.warpPerspective(jersey_working, matrix, (w, h))
            else:
                warped_jersey = cv2.warpPerspective(jersey_img, matrix, (w, h))
                
                # Manual collar
                neck_cx = int((final_ls[0] + final_rs[0]) / 2)
                neck_cy = int((final_ls[1] + final_rs[1]) / 2)
                nw = int(final_jersey_width * 0.18)
                nh = int(final_jersey_width * 0.12)
                
                cv2.ellipse(warped_jersey, (neck_cx, neck_cy - int(nh*0.5)), (nw, int(nh*0.8)), 0, 180, 360, (40,40,40,200), -1)
                cv2.ellipse(warped_jersey, (neck_cx, neck_cy), (nw, nh), 0, 0, 180, (0,0,0,0), -1)
            
            # Simple alpha blending (FAST)
            result = user_frame.copy()
            jersey_alpha = warped_jersey[:, :, 3] / 255.0
            
            for c in range(3):
                result[:, :, c] = (jersey_alpha * warped_jersey[:, :, c] + 
                                  (1 - jersey_alpha) * user_frame[:, :, c])
            
            return result
            
        except Exception as e:
            print(f"[LIVE ERROR]: {e}")
            return user_frame
    
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
    
    def close(self):
        """Release resources"""
        if self.pose:
            self.pose.close()
