import cv2
from deepface import DeepFace

class FaceDetector:
    def __init__(self, backend='opencv'):
        self.backend = backend

    def detect(self, image):
        try:
            # extract_faces returns a list of dictionaries, one for each face
            faces = DeepFace.extract_faces(img_path=image, detector_backend=self.backend, enforce_detection=False)
            return faces
        except ValueError:
            return []

