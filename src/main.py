import cv2
import argparse
import os
import pickle
import numpy as np
from datetime import datetime

from utils.google_drive import PublicDriveManager
from utils.attendance_logger import AttendanceLogger
from face_module.face_recognizer import FaceRecognizer
from gait_module.pose_estimator import PoseEstimator
from gait_module.gait_recognizer import GaitRecognizer
from tracker.person_tracker import PersonTracker


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
DB_PATH = os.path.join(PROJECT_ROOT, "database", "embeddings.pkl")


def cosine_similarity(v1, v2):
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


def authenticate_person(data, user_db, face_rec, gait_rec):
    current_gait_feat = gait_rec.extract_features(data["keypoints"])

    best_match = "Unknown"
    best_total_score = 0.0
    best_face_score = 0.0
    best_gait_score = 0.0

    for user_id, enrolled_data in user_db.items():
        enrolled_gait = enrolled_data.get("gait_features", None)
        enrolled_faces = enrolled_data.get("face_embeddings", [])

        gait_score = gait_rec.compare(current_gait_feat, enrolled_gait)

        face_scores = []
        recent_faces = data["faces"][-5:]

        for idx, face_img in enumerate(recent_faces):
            temp_path = os.path.join(PROJECT_ROOT, f"temp_face_{user_id}_{idx}.jpg")
            try:
                cv2.imwrite(temp_path, face_img)
                test_emb = face_rec.extract_features(temp_path)

                if test_emb and len(test_emb) > 0:
                    v1 = test_emb[0]["embedding"]
                    for v2 in enrolled_faces:
                        sim = cosine_similarity(v1, v2)
                        face_scores.append(sim)

            except Exception:
                pass
            finally:
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception:
                        pass

        avg_face_score = float(np.mean(face_scores)) if face_scores else None

        if avg_face_score is None:
            total_score = gait_score
            used_face_score = 0.0
        else:
            total_score = (avg_face_score * 0.5) + (gait_score * 0.5)
            used_face_score = avg_face_score

        if total_score > 0.50 and total_score > best_total_score:
            best_total_score = total_score
            best_match = user_id
            best_face_score = used_face_score
            best_gait_score = gait_score

    result = {
        "identity": best_match,
        "total_score": float(best_total_score),
        "face_score": float(best_face_score),
        "gait_score": float(best_gait_score),
    }
    return result


def preprocess_frame(frame):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    frame = cv2.filter2D(frame, -1, kernel)
    return frame


def draw_person_info(frame, x1, y1, x2, y2, info, frame_count):
    identity = info.get("identity", "Scanning...")
    total_score = info.get("total_score", 0.0)
    face_score = info.get("face_score", 0.0)
    gait_score = info.get("gait_score", 0.0)

    if identity == "Unknown":
        color = (0, 0, 255)
        status = "Scanning/Unknown"
    elif total_score >= 0.70:
        color = (0, 255, 0)
        status = "Verified"
    else:
        color = (0, 255, 255)
        status = "Possible Match"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    info_lines = [
        f"ID: {identity}",
        f"Total: {total_score:.2f}",
        f"Face: {face_score:.2f}",
        f"Gait: {gait_score:.2f}",
        f"Frames: {frame_count}",
        f"Status: {status}",
    ]

    start_y = max(25, y1 - 110)
    for i, text in enumerate(info_lines):
        cv2.putText(
            frame,
            text,
            (x1, start_y + i * 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )


def main():
    parser = argparse.ArgumentParser(
        description="Multimodal Biometrics: Gait and Face Authentication"
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "data1", "แทน", "data2"),
        help="Directory containing testing videos",
    )
    parser.add_argument(
        "--sync_url",
        type=str,
        help="Public Google Drive link for data sync",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use live camera instead of video files",
    )
    args = parser.parse_args()

    print("==================================================")
    print("Starting Multimodal Gait & Face Authentication System")

    if args.sync_url:
        print("Syncing data from Cloud...")
        drive_manager = PublicDriveManager()
        drive_manager.sync_public_folder(args.sync_url, os.path.join(PROJECT_ROOT, "data1"))

    face_rec = FaceRecognizer(model_name="VGG-Face")
    pose_est = PoseEstimator()
    gait_rec = GaitRecognizer()
    tracker = PersonTracker(pose_est.model)
    att_logger = AttendanceLogger()

    if not os.path.exists(DB_PATH):
        print(f"Error: Enrollment database not found: {DB_PATH}")
        print("Run src/enroll.py first.")
        return

    with open(DB_PATH, "rb") as f:
        user_db = pickle.load(f)

    sources = []
    if args.live:
        sources = [0]
    else:
        if os.path.exists(args.video_dir):
            sources = [
                os.path.join(args.video_dir, f)
                for f in os.listdir(args.video_dir)
                if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
            ]

    if not sources:
        print("No input source found.")
        return

    for source in sources:
        print(f"Opening source: {source}")
        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            print(f"Cannot open source: {source}")
            continue

        person_data = {}

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = preprocess_frame(frame)
            results = tracker.track(frame)

            current_time = datetime.now().strftime("%H:%M:%S")
            cv2.putText(
                frame,
                f"Time: {current_time}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            if results and len(results) > 0 and results[0].boxes is not None:
                boxes_obj = results[0].boxes
                keypoints_obj = results[0].keypoints

                if (
                    boxes_obj.id is not None
                    and keypoints_obj is not None
                    and keypoints_obj.data is not None
                ):
                    boxes = boxes_obj.xyxy.cpu().numpy()
                    track_ids = boxes_obj.id.int().cpu().tolist()
                    keypoints = keypoints_obj.data.cpu().numpy()

                    for i, track_id in enumerate(track_ids):
                        if i >= len(boxes) or i >= len(keypoints):
                            continue

                        if track_id not in person_data:
                            person_data[track_id] = {
                                "keypoints": [],
                                "faces": [],
                                "result": {
                                    "identity": "Scanning...",
                                    "total_score": 0.0,
                                    "face_score": 0.0,
                                    "gait_score": 0.0,
                                }
                            }

                        person_data[track_id]["keypoints"].append(keypoints[i])
                        person_data[track_id]["keypoints"] = person_data[track_id]["keypoints"][-90:]

                        x1, y1, x2, y2 = map(int, boxes[i])

                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(frame.shape[1], x2)
                        y2 = min(frame.shape[0], y2)

                        if x2 > x1 and y2 > y1:
                            person_crop = frame[y1:y2, x1:x2]

                            if person_crop.size > 0:
                                h, w = person_crop.shape[:2]
                                top_h = int(h * 0.45)
                                face_crop = person_crop[0:top_h, :]

                                if face_crop.size > 0:
                                    person_data[track_id]["faces"].append(face_crop)
                                    person_data[track_id]["faces"] = person_data[track_id]["faces"][-10:]

                        num_kpts = len(person_data[track_id]["keypoints"])
                        if num_kpts >= 60 and num_kpts % 15 == 0:
                            result = authenticate_person(
                                person_data[track_id],
                                user_db,
                                face_rec,
                                gait_rec,
                            )
                            person_data[track_id]["result"] = result

                            if result["identity"] != "Unknown":
                                att_logger.log_attendance(
                                    result["identity"],
                                    confidence=result["total_score"]
                                )

                        draw_person_info(
                            frame,
                            x1, y1, x2, y2,
                            person_data[track_id]["result"],
                            num_kpts
                        )

            cv2.imshow("Multimodal Authentication", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

    print("System finished.")


if __name__ == "__main__":
    main()