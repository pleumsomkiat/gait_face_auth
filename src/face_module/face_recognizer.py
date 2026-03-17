import cv2
import numpy as np
from deepface import DeepFace

class FaceRecognizer:
    def __init__(self, model_name='VGG-Face'):
        self.model_name = model_name

    def _read_img(self, img_path):
        """Safely read images from paths with potential non-ASCII characters on Windows."""
        if isinstance(img_path, str):
            # Read via numpy to avoid OpenCV's non-ASCII path limitation on Windows
            stream = open(img_path, "rb")
            bytes = bytearray(stream.read())
            numpy_array = np.asarray(bytes, dtype=np.uint8)
            img = cv2.imdecode(numpy_array, cv2.IMREAD_UNCHANGED)
            stream.close()
            return img
        return img_path # Already an array

    def extract_features(self, face_image):
        try:
            img = self._read_img(face_image)
            # Pass numpy array instead of path
            embeddings = DeepFace.represent(img_path=img, model_name=self.model_name, enforce_detection=False)
            return embeddings
        except Exception as e:
            print(f"Face extraction error: {e}")
            return None
        
    def compare(self, img1, img2):
        try:
            i1 = self._read_img(img1)
            i2 = self._read_img(img2)
            result = DeepFace.verify(img1_path=i1, img2_path=i2, model_name=self.model_name, enforce_detection=False)
            return result
        except Exception as e:
             print(f"Face comparison error: {e}")
             return {"verified": False, "distance": float('inf')}


