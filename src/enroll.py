import argparse
import os
import pickle

import cv2
import numpy as np

from face_module.face_recognizer import FaceRecognizer
from gait_module.gait_recognizer import GaitRecognizer
from gait_module.pose_estimator import PoseEstimator


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
DEFAULT_DB_PATH = os.path.join(PROJECT_ROOT, "database", "embeddings.pkl")
DEFAULT_DATASET_ROOT = os.path.join(PROJECT_ROOT, "data1")


class EnrollmentManager:
    def __init__(self, db_path=DEFAULT_DB_PATH):
        self.db_path = db_path
        self.face_rec = FaceRecognizer(model_name="VGG-Face")
        self.pose_est = PoseEstimator()
        self.gait_rec = GaitRecognizer()
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

    def extract_face_embedding(self, img_path):
        try:
            emb = self.face_rec.extract_features(img_path)
            if emb and len(emb) > 0:
                return np.asarray(emb[0]["embedding"], dtype=np.float32)
        except Exception:
            return None
        return None

    def _list_subdirs(self, root_dir):
        if not root_dir or not os.path.exists(root_dir):
            return {}

        return {
            name: os.path.join(root_dir, name)
            for name in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, name))
        }

    def resolve_user_sources(self, dataset_root):
        image_dirs = self._list_subdirs(os.path.join(dataset_root, "data1"))
        video_dirs = self._list_subdirs(os.path.join(dataset_root, "data2"))

        if image_dirs or video_dirs:
            user_names = sorted(set(image_dirs) | set(video_dirs))
            return {
                user_name: {
                    "img_dir": image_dirs.get(user_name),
                    "vid_dir": video_dirs.get(user_name),
                }
                for user_name in user_names
            }

        user_sources = {}
        for user_name, user_path in self._list_subdirs(dataset_root).items():
            user_sources[user_name] = {
                "img_dir": os.path.join(user_path, "data1"),
                "vid_dir": os.path.join(user_path, "data2"),
            }
        return user_sources

    def enroll_users(self, dataset_root):
        database = {}

        if not os.path.exists(dataset_root):
            print(f"Dataset path not found: {dataset_root}")
            return

        user_sources = self.resolve_user_sources(dataset_root)
        if not user_sources:
            print(f"No users found in dataset path: {dataset_root}")
            return

        for user_name, paths in user_sources.items():
            print(f"\nEnrolling user: {user_name}")

            database[user_name] = {
                "display_name": user_name,
                "face_embeddings": [],
                "face_mean": None,
                "gait_features": None,
            }

            img_dir = paths.get("img_dir")
            if img_dir and os.path.exists(img_dir):
                for file in os.listdir(img_dir):
                    if not file.lower().endswith((".jpg", ".png", ".jpeg")):
                        continue

                    img_path = os.path.join(img_dir, file)
                    embedding = self.extract_face_embedding(img_path)

                    if embedding is None:
                        print(f"  Skipped bad face image: {file}")
                        continue

                    database[user_name]["face_embeddings"].append(embedding)

            if database[user_name]["face_embeddings"]:
                database[user_name]["face_mean"] = np.mean(
                    database[user_name]["face_embeddings"],
                    axis=0,
                )
            else:
                print("  Warning: no face embeddings enrolled")

            vid_dir = paths.get("vid_dir")
            gait_list = []

            if vid_dir and os.path.exists(vid_dir):
                for file in os.listdir(vid_dir):
                    if not file.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
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

                    if not kpts_seq:
                        print(f"  No pose sequence extracted: {file}")
                        continue

                    feat = self.gait_rec.extract_features(np.array(kpts_seq))
                    if feat is not None:
                        gait_list.append(feat)

            if gait_list:
                database[user_name]["gait_features"] = np.mean(gait_list, axis=0)
            else:
                print("  Warning: no gait features enrolled")

        with open(self.db_path, "wb") as f:
            pickle.dump(database, f)

        print(f"\nEnrollment complete -> {self.db_path}")
        print("Enrollment summary:")
        for user_name, enrolled in database.items():
            print(
                f"  {user_name}: "
                f"faces={len(enrolled['face_embeddings'])}, "
                f"face_mean={'yes' if enrolled['face_mean'] is not None else 'no'}, "
                f"gait={'yes' if enrolled['gait_features'] is not None else 'no'}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enroll face and gait features into embeddings database"
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=DEFAULT_DATASET_ROOT,
        help=(
            "Dataset root. Supports either "
            "<root>/data1/<user> + <root>/data2/<user> "
            "or <root>/<user>/data1 + <root>/<user>/data2"
        ),
    )
    parser.add_argument(
        "--db_path",
        type=str,
        default=DEFAULT_DB_PATH,
        help="Output pickle database path",
    )
    args = parser.parse_args()

    manager = EnrollmentManager(db_path=args.db_path)
    manager.enroll_users(args.dataset_root)
