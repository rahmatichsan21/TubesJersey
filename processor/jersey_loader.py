"""
Jersey Loader Utility
Helper untuk load dan pre-process jersey images
"""

import cv2
import numpy as np
import os
import json

class JerseyLoader:
    @staticmethod
    def apply_3d_cylindrical_warp(jersey_img, strength=0.15):
        """
        Apply cylindrical warp untuk efek 3D melengkung
        
        Args:
            jersey_img: Jersey image (BGRA)
            strength: Kekuatan lengkungan (0.0-0.5)
        
        Returns:
            Warped jersey image
        """
        h, w = jersey_img.shape[:2]
        result = np.zeros_like(jersey_img)
        
        center_x = w / 2
        max_offset = strength * w
        
        for y in range(h):
            for x in range(w):
                distance_from_center = abs(x - center_x)
                normalized_distance = distance_from_center / center_x
                
                # Parabolic curve
                horizontal_offset = int(max_offset * (normalized_distance ** 2))
                
                if x < center_x:
                    src_x = x - horizontal_offset
                else:
                    src_x = x + horizontal_offset
                
                src_x = max(0, min(w - 1, src_x))
                result[y, x] = jersey_img[y, src_x]
        
        return result
    
    @staticmethod
    def load_and_prepare_jersey(jersey_path, warp_strength=0.15):
        """
        Load jersey dan apply pre-processing
        
        Args:
            jersey_path: Path ke file jersey
            warp_strength: Cylindrical warp strength
        
        Returns:
            Prepared jersey (BGRA) or None if failed
        """
        img = cv2.imread(jersey_path, cv2.IMREAD_UNCHANGED)
        
        if img is None:
            print(f"[ERROR] Gagal load jersey: {jersey_path}")
            return None
        
        # Ensure BGRA
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            img[:, :, 3] = 255
        
        # Apply 3D warp
        warped = JerseyLoader.apply_3d_cylindrical_warp(img, warp_strength)
        
        print(f"[INFO] Jersey prepared: {os.path.basename(jersey_path)} ({warped.shape[1]}x{warped.shape[0]})")
        
        return warped
    
    @staticmethod
    def load_metadata(metadata_file):
        """Load jersey metadata JSON"""
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r') as f:
                    data = json.load(f)
                print(f"[INFO] Metadata loaded: {len(data)} jerseys")
                return data
            except Exception as e:
                print(f"[WARN] Failed to load metadata: {e}")
                return {}
        else:
            print(f"[WARN] Metadata file not found: {metadata_file}")
            return {}
