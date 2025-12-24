"""
Jersey Loader Module
Untuk load dan cache jersey images
"""

import os
import cv2
import json

class JerseyLoader:
    """Utility class untuk loading jersey images dan metadata"""
    
    def __init__(self, jersey_folder='Assets/Jerseys/PremierLeague/Home_NOBG', 
                 metadata_file='jersey_metadata.json',
                 landmarks_file='jersey_landmarks.json'):
        self.jersey_folder = jersey_folder
        self.metadata_file = metadata_file
        self.landmarks_file = landmarks_file
        self.jersey_cache = {}
        
        # Load metadata dan landmarks
        self.metadata = self._load_json(metadata_file)
        self.landmarks = self._load_json(landmarks_file)
    
    def _load_json(self, filepath):
        """Load JSON file"""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"[JerseyLoader] Warning: Could not load {filepath}: {e}")
        return {}
    
    def get_jersey(self, jersey_name, use_cache=True):
        """
        Load jersey image (dengan caching)
        
        Args:
            jersey_name: Nama file jersey
            use_cache: Gunakan cache atau tidak
            
        Returns:
            Jersey image (BGRA format) atau None
        """
        # Check cache first
        if use_cache and jersey_name in self.jersey_cache:
            return self.jersey_cache[jersey_name].copy()
        
        # Load from disk
        jersey_path = os.path.join(self.jersey_folder, jersey_name)
        
        if not os.path.exists(jersey_path):
            print(f"[JerseyLoader] Jersey not found: {jersey_path}")
            return None
        
        # Read with alpha channel
        jersey_img = cv2.imread(jersey_path, cv2.IMREAD_UNCHANGED)
        
        if jersey_img is None:
            print(f"[JerseyLoader] Failed to read jersey: {jersey_path}")
            return None
        
        # Ensure BGRA format
        if len(jersey_img.shape) < 3 or jersey_img.shape[2] < 4:
            if len(jersey_img.shape) == 3 and jersey_img.shape[2] == 3:
                # Add alpha channel
                b, g, r = cv2.split(jersey_img)
                alpha = cv2.cvtColor(jersey_img, cv2.COLOR_BGR2GRAY)
                _, alpha = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)
                jersey_img = cv2.merge((b, g, r, alpha))
        
        # Cache it
        if use_cache:
            self.jersey_cache[jersey_name] = jersey_img.copy()
        
        return jersey_img
    
    def get_metadata(self, jersey_name):
        """Get metadata untuk jersey tertentu"""
        return self.metadata.get(jersey_name, {})
    
    def get_landmarks(self, jersey_name):
        """Get landmarks untuk jersey tertentu"""
        return self.landmarks.get(jersey_name, {})
    
    def list_jerseys(self):
        """List semua jersey yang tersedia"""
        jerseys = []
        if os.path.exists(self.jersey_folder):
            for f in os.listdir(self.jersey_folder):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    jerseys.append(f)
        return sorted(jerseys)
    
    def clear_cache(self):
        """Clear jersey cache"""
        self.jersey_cache.clear()
