import os
import pickle
import numpy as np
import cv2

from face_module.face_recognizer import FaceRecognizer
from gait_module.gait_recognizer import GaitRecognizer
from gait_module.pose_estimator import PoseEstimator


class EnrollmentManager:
    def __init__(self, db_path='database/embeddings.pkl'):
        self.db_path = db_path
        self.face_rec = FaceRecognizer(model_name='VGG-Face')
        self.pose_est = PoseEstimator()
        self.gait_rec = GaitRecognizer()
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

    def is_valid_face_image(self, img_path):
        """
        pre-check แบบเข้ม:
        - โหลดภาพได้
        - detect face ได้จริง
        """
        img = cv2.imread(img_path)
        if img is None:
            return False

        try:
            emb = self.face_rec.extract_features(img_path)
            return emb is not None and len(emb) > 0
        except Exception:
            return False

    def enroll_users(self, dataset_root):
        database = {}

        if not os.path.exists(dataset_root):
            print(f"Dataset path not found: {dataset_root}")
            return

        users = os.listdir(dataset_root)

        for user_name in users:
            user_path = os.path.join(dataset_root, user_name)
            if not os.path.isdir(user_path):
                continue

            print(f"\nEnrolling user: {user_name}")

            database[user_name] = {
                'face_embeddings': [],
                'face_mean': None,
                'gait_features': None
            }

            # -----------------------
            # FACE
            # -----------------------
            img_dir = os.path.join(user_path, 'data1')
            if os.path.exists(img_dir):
                for file in os.listdir(img_dir):
                    if not file.lower().endswith(('.jpg', '.png', '.jpeg')):
                        continue

                    img_path = os.path.join(img_dir, file)

                    if not self.is_valid_face_image(img_path):
                        print(f"  Skipped bad face image: {file}")
                        continue

                    try:
                        emb = self.face_rec.extract_features(img_path)
                        if emb and len(emb) > 0:
                            database[user_name]['face_embeddings'].append(
                                emb[0]['embedding']
                            )
                    except Exception:
                        print(f"  Failed face embedding: {file}")

            if database[user_name]['face_embeddings']:
                database[user_name]['face_mean'] = np.mean(
                    database[user_name]['face_embeddings'], axis=0
                )

            # -----------------------
            # GAIT
            # -----------------------
            vid_dir = os.path.join(user_path, 'data2')
            gait_list = []

            if os.path.exists(vid_dir):
                for file in os.listdir(vid_dir):
                    if not file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                        continue

                    video_path = os.path.join(vid_dir, file)
                    print(f"  Processing gait video: {file}")

                    cap = cv2.VideoCapture(video_path)
                    kpts_seq = []

                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break

                        res = self.pose_est.estimate(frame)

                        if res and len(res) > 0:
                            kpts = res[0].keypoints
                            if kpts is not None and kpts.data is not None:
                                kpts_np = kpts.data.cpu().numpy()
                                if len(kpts_np) > 0:
                                    kpts_seq.append(kpts_np[0])

                    cap.release()

                    if kpts_seq:
                        feat = self.gait_rec.extract_features(np.array(kpts_seq))
                        if feat is not None:
                            gait_list.append(feat)

            if gait_list:
                database[user_name]['gait_features'] = np.mean(gait_list, axis=0)

        with open(self.db_path, 'wb') as f:
            pickle.dump(database, f)

        print(f"\nEnrollment complete -> {self.db_path}")


if __name__ == "__main__":
    manager = EnrollmentManager()
    manager.enroll_users('dataset')