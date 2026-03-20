import cv2
from deepface import DeepFace


class FaceDetector:
    def __init__(self, backend='opencv'):
        self.backend = backend
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(cascade_path)

    def detect(self, image):
        try:
            # extract_faces returns a list of dictionaries, one for each face
            faces = DeepFace.extract_faces(img_path=image, detector_backend=self.backend, enforce_detection=False)
            return faces
        except ValueError:
            return []

    def extract_primary_face(self, image, min_size=(40, 40)):
        if image is None or image.size == 0:
            return None

        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        except Exception:
            return None

        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=min_size,
        )

        if len(faces) == 0:
            return None

        x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
        pad_x = int(w * 0.18)
        pad_y = int(h * 0.28)

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(image.shape[1], x + w + pad_x)
        y2 = min(image.shape[0], y + h + pad_y)

        face_crop = image[y1:y2, x1:x2]
        if face_crop.size == 0:
            return None

        return face_crop

