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

    def enroll_users(self, data1_root):
        """
        data1_root/
            ├── user1/
            │   ├── data1 (images)
            │   └── data2 (videos)
        """
        database = {}

        users = os.listdir(data1_root)

        for user_name in users:
            user_path = os.path.join(data1_root, user_name)
            if not os.path.isdir(user_path):
                continue

            print(f"\nEnrolling user: {user_name}")

            database[user_name] = {
                'face_embeddings': [],
                'gait_features': None
            }

            # -----------------------
            # 1) FACE (data1)
            # -----------------------
            img_dir = os.path.join(user_path, 'data1')
            if os.path.exists(img_dir):

                for file in os.listdir(img_dir):
                    if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                        img_path = os.path.join(img_dir, file)

                        emb = self.face_rec.extract_features(img_path)

                        if emb and len(emb) > 0:
                            database[user_name]['face_embeddings'].append(
                                emb[0]['embedding']
                            )

            # เฉลี่ย face embeddings
            if database[user_name]['face_embeddings']:
                database[user_name]['face_mean'] = np.mean(
                    database[user_name]['face_embeddings'], axis=0
                )

            # -----------------------
            # 2) GAIT (data2)
            # -----------------------
            vid_dir = os.path.join(user_path, 'data2')
            gait_list = []

            if os.path.exists(vid_dir):
                for file in os.listdir(vid_dir):
                    if file.lower().endswith(('.mp4', '.avi')):
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
                                        kpts_seq.append(
                                            self.normalize_keypoints(kpts_np[0])
                                        )

                        cap.release()

                        if kpts_seq:
                            feat = self.gait_rec.extract_features(
                                np.array(kpts_seq)
                            )
                            gait_list.append(feat)

            # รวม gait หลายคลิป
            if gait_list:
                database[user_name]['gait_features'] = np.mean(
                    gait_list, axis=0
                )

        # -----------------------
        # SAVE DATABASE
        # -----------------------
        with open(self.db_path, 'wb') as f:
            pickle.dump(database, f)

        print(f"\nEnrollment complete → {self.db_path}")

    # -----------------------
    # NORMALIZE KEYPOINTS
    # -----------------------
    def normalize_keypoints(self, kpts):
        # center (hip)
        center = kpts[11]
        kpts = kpts - center

        # scale (height)
        height = np.linalg.norm(kpts[5] - kpts[15])
        if height > 0:
            kpts = kpts / height

        return kpts


if __name__ == "__main__":
    manager = EnrollmentManager()
    manager.enroll_users('data1')