import argparse
import os
import pickle

import cv2
import numpy as np
try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # pragma: no cover - optional dependency fallback
    Image = None
    ImageDraw = None
    ImageFont = None

from face_module.face_detector import FaceDetector
from face_module.face_recognizer import FaceRecognizer
from gait_module.gait_recognizer import GaitRecognizer
from gait_module.pose_estimator import PoseEstimator
from tracker.person_tracker import PersonTracker
from utils.attendance_logger import AttendanceLogger
from utils.google_drive import PublicDriveManager


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
DB_PATH = os.path.join(PROJECT_ROOT, "database", "embeddings.pkl")
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")
FONT_CANDIDATES = (
    r"C:\Windows\Fonts\tahoma.ttf",
    r"C:\Windows\Fonts\LeelawUI.ttf",
    r"C:\Windows\Fonts\arial.ttf",
)


def cosine_similarity(v1, v2):
    if v1 is None or v2 is None:
        return 0.0

    v1 = np.asarray(v1, dtype=np.float32)
    v2 = np.asarray(v2, dtype=np.float32)

    if v1.shape != v2.shape:
        return 0.0

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(np.dot(v1, v2) / (norm1 * norm2))


def contains_non_ascii(text):
    return any(ord(char) > 127 for char in str(text))


def load_ui_font(font_size):
    if ImageFont is None:
        return None

    for font_path in FONT_CANDIDATES:
        if not os.path.exists(font_path):
            continue
        try:
            return ImageFont.truetype(font_path, font_size)
        except Exception:
            continue
    return None


def draw_text_line(frame, text, origin, color, font_scale=0.5, thickness=1):
    if not contains_non_ascii(text) or Image is None or ImageDraw is None:
        cv2.putText(
            frame,
            str(text),
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        return frame

    font_size = max(16, int(18 * max(1.0, font_scale * 2)))
    font = load_ui_font(font_size)
    if font is None:
        cv2.putText(
            frame,
            str(text).encode("ascii", errors="replace").decode("ascii"),
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        return frame

    x, y = origin
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    draw.text((x, max(0, y - font_size)), str(text), font=font, fill=color[::-1])
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def stable_result(history, fallback_result=None, default="Scanning..."):
    if not history:
        return dict(fallback_result or {"identity": default})

    latest_entry = history[-1]
    latest_identity = latest_entry.get("identity", default)
    latest_total = float(latest_entry.get("total_score", 0.0))

    # Trust a strong latest multimodal match instead of letting stale guesses
    # from earlier frames override it through majority voting.
    if latest_identity not in {"Unknown", "Scanning..."} and latest_total >= 0.78:
        return dict(latest_entry)

    weighted_scores = {}
    for idx, entry in enumerate(history, start=1):
        identity = entry.get("identity", default)
        if identity in {"Unknown", "Scanning..."}:
            continue

        total_score = float(entry.get("total_score", 0.0))
        if total_score <= 0:
            continue

        recency_weight = 0.70 + (0.12 * idx)
        weighted_scores[identity] = (
            weighted_scores.get(identity, 0.0) + (total_score * recency_weight)
        )

    if not weighted_scores:
        return dict(latest_entry)

    stable_identity = max(weighted_scores, key=weighted_scores.get)
    for entry in reversed(history):
        if entry.get("identity") == stable_identity:
            return dict(entry)

    return dict(latest_entry)


def make_person_state():
    return {
        "keypoints": [],
        "faces": [],
        "result": {
            "identity": "Scanning...",
            "display_name": "Scanning...",
            "total_score": 0.0,
            "face_score": 0.0,
            "gait_score": 0.0,
            "status_hint": "Collecting face and gait",
        },
        "history": [],
        "frame_counter": 0,
        "last_auth_frame": 0,
        "pending_identity": None,
        "pending_count": 0,
        "confirmed_identity": None,
        "session_logged": False,
    }


def resolve_live_session_result(state, result):
    identity = result.get("identity", "Scanning...")

    if identity in {"Unknown", "Scanning..."}:
        state["pending_identity"] = None
        state["pending_count"] = 0
        if state.get("confirmed_identity"):
            return dict(state["result"])
        return {
            "identity": "Scanning...",
            "display_name": "Scanning...",
            "total_score": 0.0,
            "face_score": 0.0,
            "gait_score": 0.0,
            "status_hint": "Collecting face and gait",
        }

    confirmed_identity = state.get("confirmed_identity")
    if confirmed_identity:
        if identity == confirmed_identity:
            return dict(result)
        return dict(state["result"])

    if state.get("pending_identity") == identity:
        state["pending_count"] += 1
    else:
        state["pending_identity"] = identity
        state["pending_count"] = 1

    if state["pending_count"] >= 2:
        state["confirmed_identity"] = identity
        return dict(result)

    return {
        "identity": "Scanning...",
        "display_name": "Scanning...",
        "total_score": 0.0,
        "face_score": 0.0,
        "gait_score": 0.0,
        "status_hint": "Confirming identity",
    }


def face_quality_score(face_img):
    if face_img is None or face_img.size == 0:
        return 0.0

    try:
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    except Exception:
        return 0.0

    sharpness = cv2.Laplacian(gray, cv2.CV_32F).var()
    h, w = gray.shape[:2]
    return float(sharpness * np.sqrt(max(1, h * w)))


def pose_frame_is_reliable(keypoints):
    if keypoints is None or len(keypoints) < 17:
        return False

    try:
        conf = np.asarray(keypoints[:, 2], dtype=np.float32)
    except Exception:
        return False

    core_idx = [5, 6, 11, 12, 13, 14, 15, 16]
    arm_idx = [7, 8, 9, 10]
    lower_idx = [11, 12, 13, 14, 15, 16]

    core_mean = float(np.mean(conf[core_idx]))
    arm_mean = float(np.mean(conf[arm_idx]))
    lower_min = float(np.min(conf[lower_idx]))

    return core_mean >= 0.32 and arm_mean >= 0.15 and lower_min >= 0.10


def extract_best_face_crop(person_crop, face_detector):
    if person_crop is None or person_crop.size == 0:
        return None

    h, _ = person_crop.shape[:2]
    top_h = min(h, max(50, int(h * 0.75)))
    upper_body = person_crop[:top_h, :]

    detected_face = face_detector.extract_primary_face(upper_body)
    if detected_face is None or detected_face.size == 0:
        return None

    face_h, face_w = detected_face.shape[:2]
    aspect_ratio = face_w / max(1.0, float(face_h))
    if face_h < 48 or face_w < 48:
        return None
    if aspect_ratio < 0.55 or aspect_ratio > 1.8:
        return None

    return detected_face


def pick_primary_live_index(boxes, frame_shape):
    if boxes is None or len(boxes) == 0:
        return None

    frame_h, frame_w = frame_shape[:2]
    frame_center = np.asarray([frame_w / 2.0, frame_h / 2.0], dtype=np.float32)
    frame_scale = np.asarray([frame_w, frame_h], dtype=np.float32)

    best_idx = None
    best_score = -1.0

    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = box[:4]
        width = max(1.0, x2 - x1)
        height = max(1.0, y2 - y1)
        area = width * height
        center = np.asarray([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)
        center_distance = np.linalg.norm((center - frame_center) / frame_scale)
        center_weight = max(0.2, 1.35 - (center_distance * 1.8))
        score = area * center_weight

        if score > best_score:
            best_score = score
            best_idx = idx

    return best_idx


def extract_recent_face_embeddings(face_images, face_rec, max_faces=3):
    embedding_candidates = []
    scored_faces = []

    for face_img in face_images:
        if face_img is None or face_img.size == 0:
            continue
        scored_faces.append((face_quality_score(face_img), face_img))

    scored_faces.sort(key=lambda item: item[0], reverse=True)

    for quality, face_img in scored_faces[: max(6, max_faces * 2)]:
        if quality <= 0:
            continue

        try:
            # The crop has already passed an explicit face detector, so allowing
            # represent() to skip a second hard rejection keeps live scan usable
            # while the earlier crop filters still block obvious non-face inputs.
            face_result = face_rec.extract_features(face_img, enforce_detection=False)
            if face_result and len(face_result) > 0:
                embedding_candidates.append(
                    (
                        float(quality),
                        np.asarray(face_result[0]["embedding"], dtype=np.float32),
                    )
                )
        except Exception:
            pass

    if not embedding_candidates:
        return []

    if len(embedding_candidates) == 1:
        return [embedding_candidates[0][1]]

    ranked_candidates = []
    for idx, (quality, embedding) in enumerate(embedding_candidates):
        peer_scores = [
            cosine_similarity(embedding, other_embedding)
            for other_idx, (_, other_embedding) in enumerate(embedding_candidates)
            if other_idx != idx
        ]
        consensus = float(np.mean(peer_scores)) if peer_scores else 1.0
        ranked_candidates.append((consensus, quality, embedding))

    ranked_candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)

    consistent_embeddings = []
    for consensus, _, embedding in ranked_candidates:
        if consensus >= 0.45 or len(consistent_embeddings) < 2:
            consistent_embeddings.append(embedding)
        if len(consistent_embeddings) >= max_faces:
            break

    return consistent_embeddings


def authenticate_person(data, user_db, face_rec, gait_rec, live_mode=False):
    current_gait_feat = gait_rec.extract_features(data["keypoints"])
    recent_face_embeddings = extract_recent_face_embeddings(data["faces"], face_rec)
    has_face_signal = len(recent_face_embeddings) > 0
    has_gait_signal = current_gait_feat is not None

    best_match = "Unknown"
    best_total_score = 0.0
    best_face_score = 0.0
    best_gait_score = 0.0
    ambiguous_live_match = False
    live_candidates = []

    for user_id, enrolled_data in user_db.items():
        enrolled_gait = enrolled_data.get("gait_features")
        enrolled_face_mean = enrolled_data.get("face_mean")
        gait_score = gait_rec.compare(current_gait_feat, enrolled_gait)

        face_scores = []
        if enrolled_face_mean is not None:
            face_scores = [
                cosine_similarity(face_embedding, enrolled_face_mean)
                for face_embedding in recent_face_embeddings
            ]

        avg_face_score = float(np.mean(face_scores)) if face_scores else None

        if live_mode:
            accept = False
            if avg_face_score is None or current_gait_feat is None:
                total_score = 0.0
                used_face_score = float(avg_face_score or 0.0)
            else:
                face_weight = 0.65
                gait_weight = 0.35
                total_score = (avg_face_score * face_weight) + (gait_score * gait_weight)
                used_face_score = avg_face_score
                live_candidates.append(
                    {
                        "identity": user_id,
                        "display_name": enrolled_data.get("display_name", user_id),
                        "total_score": float(total_score),
                        "face_score": float(avg_face_score),
                        "gait_score": float(gait_score),
                    }
                )
        else:
            if avg_face_score is None:
                total_score = gait_score
                accept = gait_score > 0.80
                used_face_score = 0.0
            else:
                total_score = (avg_face_score * 0.70) + (gait_score * 0.30)
                accept = total_score > 0.60
                used_face_score = avg_face_score

        if accept and total_score > best_total_score:
            best_total_score = total_score
            best_match = user_id
            best_face_score = used_face_score
            best_gait_score = gait_score

    if live_mode and live_candidates:
        ranked_candidates = sorted(
            live_candidates,
            key=lambda item: item["total_score"],
            reverse=True,
        )
        best_candidate = ranked_candidates[0]
        second_candidate = ranked_candidates[1] if len(ranked_candidates) > 1 else None

        total_margin = (
            best_candidate["total_score"] - second_candidate["total_score"]
            if second_candidate
            else best_candidate["total_score"]
        )
        face_margin = (
            best_candidate["face_score"] - second_candidate["face_score"]
            if second_candidate
            else best_candidate["face_score"]
        )
        gait_margin = (
            best_candidate["gait_score"] - second_candidate["gait_score"]
            if second_candidate
            else best_candidate["gait_score"]
        )

        margin_is_clear = second_candidate is None or (
            total_margin >= 0.05
            and (face_margin >= 0.03 or gait_margin >= 0.03)
        )
        accept_live_candidate = (
            best_candidate["total_score"] > 0.69
            and best_candidate["face_score"] > 0.62
            and best_candidate["gait_score"] > 0.56
            and margin_is_clear
        )

        if accept_live_candidate:
            best_match = best_candidate["identity"]
            best_total_score = best_candidate["total_score"]
            best_face_score = best_candidate["face_score"]
            best_gait_score = best_candidate["gait_score"]
        elif best_candidate["total_score"] > 0.64:
            ambiguous_live_match = True
            best_face_score = best_candidate["face_score"]
            best_gait_score = best_candidate["gait_score"]

    if live_mode:
        if not has_face_signal and not has_gait_signal:
            status_hint = "Collecting face and gait"
        elif not has_face_signal:
            status_hint = "Need face"
        elif not has_gait_signal:
            status_hint = "Need gait"
        elif ambiguous_live_match:
            status_hint = "Ambiguous match"
        elif best_match == "Unknown":
            status_hint = "No multimodal match"
        else:
            status_hint = "Verified"
    else:
        status_hint = "Verified" if best_match != "Unknown" else "Unknown"

    return {
        "identity": best_match,
        "display_name": user_db.get(best_match, {}).get("display_name", best_match),
        "total_score": float(best_total_score),
        "face_score": float(best_face_score),
        "gait_score": float(best_gait_score),
        "status_hint": status_hint,
    }


def preprocess_frame(frame):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(frame, -1, kernel)


def draw_person_info(frame, x1, y1, x2, y2, info):
    display_name = info.get("display_name") or info.get("identity", "Scanning...")
    identity = info.get("identity", "Scanning...")
    total_score = info.get("total_score", 0.0)

    if identity == "Unknown":
        color = (0, 0, 255)
    elif identity == "Scanning...":
        color = (255, 200, 0)
    elif total_score >= 0.75:
        color = (0, 255, 0)
    else:
        color = (0, 255, 255)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    if identity not in {"Unknown", "Scanning..."}:
        label_text = f"Name: {display_name}"
        label_width = min(frame.shape[1] - x1, max(180, len(label_text) * 11))
        label_y2 = y1 if y1 >= 30 else min(frame.shape[0] - 1, y1 + 30)
        label_y1 = max(0, label_y2 - 30)
        cv2.rectangle(
            frame,
            (x1, label_y1),
            (min(frame.shape[1] - 1, x1 + label_width), label_y2),
            color,
            -1,
        )
        frame = draw_text_line(
            frame,
            label_text,
            (x1 + 8, max(22, label_y2 - 7)),
            (255, 255, 255),
            font_scale=0.7,
            thickness=2,
        )

    return frame


def resolve_sources(video_dir, live_mode):
    if live_mode:
        return [0]

    if not video_dir:
        return []

    if os.path.isfile(video_dir):
        return [video_dir] if video_dir.lower().endswith(VIDEO_EXTENSIONS) else []

    if not os.path.exists(video_dir):
        return []

    sources = []
    for root, _, files in os.walk(video_dir):
        for file_name in files:
            if file_name.lower().endswith(VIDEO_EXTENSIONS):
                sources.append(os.path.join(root, file_name))

    return sorted(sources)


def load_user_db():
    if not os.path.exists(DB_PATH):
        print(f"Error: Enrollment database not found: {DB_PATH}")
        print("Run src/enroll.py first.")
        return None

    with open(DB_PATH, "rb") as f:
        raw_user_db = pickle.load(f)

    valid_user_db = {}
    print("Loaded enrollment database:")
    for user_id, enrolled_data in raw_user_db.items():
        has_face = enrolled_data.get("face_mean") is not None
        has_gait = enrolled_data.get("gait_features") is not None
        print(
            f"  - {user_id}: "
            f"face={'yes' if has_face else 'no'}, "
            f"gait={'yes' if has_gait else 'no'}"
        )
        if has_face or has_gait:
            valid_user_db[user_id] = enrolled_data

    if not valid_user_db:
        print("No valid enrolled users found in database.")
        print("Run src/enroll.py to rebuild the database.")
        return None

    return valid_user_db


def main():
    parser = argparse.ArgumentParser(
        description="Multimodal Biometrics: Gait and Face Authentication"
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "data1", "data2"),
        help="Video file or directory containing videos",
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
        print("Syncing data from cloud...")
        drive_manager = PublicDriveManager()
        drive_manager.sync_public_folder(
            args.sync_url,
            os.path.join(PROJECT_ROOT, "data1"),
        )

    user_db = load_user_db()
    if user_db is None:
        return

    face_rec = FaceRecognizer(model_name="VGG-Face")
    face_detector = FaceDetector()
    pose_est = PoseEstimator()
    gait_rec = GaitRecognizer()
    tracker = None if args.live else PersonTracker(pose_est.model)
    att_logger = AttendanceLogger()

    sources = resolve_sources(args.video_dir, args.live)
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
        live_subject_key = "live_primary"
        live_missing_frames = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (960, 540))
            clean_frame = frame.copy()
            inference_frame = preprocess_frame(frame)
            results = (
                pose_est.estimate(inference_frame)
                if args.live
                else tracker.track(inference_frame)
            )
            display_frame = clean_frame.copy()

            if results and len(results) > 0:
                try:
                    display_frame = results[0].plot(
                        labels=False,
                        conf=False,
                        boxes=False,
                    )
                except Exception:
                    display_frame = clean_frame.copy()

            detections = []
            if results and len(results) > 0 and results[0].boxes is not None:
                boxes_obj = results[0].boxes
                keypoints_obj = results[0].keypoints

                if keypoints_obj is not None and keypoints_obj.data is not None:
                    boxes = boxes_obj.xyxy.cpu().numpy()
                    keypoints = keypoints_obj.data.cpu().numpy()

                    if args.live:
                        primary_idx = pick_primary_live_index(boxes, clean_frame.shape)
                        if primary_idx is not None and primary_idx < len(keypoints):
                            detections.append((live_subject_key, primary_idx, boxes, keypoints))
                    elif boxes_obj.id is not None:
                        track_ids = boxes_obj.id.int().cpu().tolist()
                        for idx, track_id in enumerate(track_ids):
                            if idx < len(boxes) and idx < len(keypoints):
                                detections.append((track_id, idx, boxes, keypoints))

            if args.live:
                if detections:
                    live_missing_frames = 0
                else:
                    live_missing_frames += 1
                    if live_missing_frames > 20 and live_subject_key in person_data:
                        person_data[live_subject_key] = make_person_state()

            for subject_key, idx, boxes, keypoints in detections:
                if subject_key not in person_data:
                    person_data[subject_key] = make_person_state()

                state = person_data[subject_key]
                state["frame_counter"] += 1
                if pose_frame_is_reliable(keypoints[idx]):
                    state["keypoints"].append(keypoints[idx])
                    state["keypoints"] = state["keypoints"][-90:]

                x1, y1, x2, y2 = map(int, boxes[idx][:4])
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(clean_frame.shape[1], x2)
                y2 = min(clean_frame.shape[0], y2)

                if x2 > x1 and y2 > y1:
                    person_crop = clean_frame[y1:y2, x1:x2]
                    face_crop = extract_best_face_crop(person_crop, face_detector)

                    if (
                        face_crop is not None
                        and face_crop.size > 0
                        and face_crop.shape[0] >= 40
                        and face_crop.shape[1] >= 40
                    ):
                        state["faces"].append(face_crop)
                        state["faces"] = sorted(
                            state["faces"],
                            key=face_quality_score,
                            reverse=True,
                        )[:8]

                frame_counter = state["frame_counter"]
                num_kpts = len(state["keypoints"])
                auth_interval = 12 if args.live else 45
                min_kpts = 20 if args.live else 60
                should_authenticate = (
                    num_kpts >= min_kpts
                    and (frame_counter - state["last_auth_frame"]) >= auth_interval
                )

                if should_authenticate:
                    result = authenticate_person(
                        state,
                        user_db,
                        face_rec,
                        gait_rec,
                        live_mode=args.live,
                    )
                    state["last_auth_frame"] = frame_counter
                    state["history"].append(
                        {
                            "identity": result["identity"],
                            "display_name": result.get("display_name", result["identity"]),
                            "total_score": result["total_score"],
                            "face_score": result["face_score"],
                            "gait_score": result["gait_score"],
                            "status_hint": result["status_hint"],
                        }
                    )
                    state["history"] = state["history"][-5:]

                    smoothed_result = stable_result(
                        state["history"],
                        fallback_result=result,
                        default="Scanning...",
                    )
                    smoothed_id = smoothed_result["identity"]
                    smoothed_result["display_name"] = user_db.get(
                        smoothed_id,
                        {},
                    ).get("display_name", smoothed_result.get("display_name", smoothed_id))
                    final_result = (
                        resolve_live_session_result(state, smoothed_result)
                        if args.live
                        else smoothed_result
                    )
                    state["result"] = final_result

                    if args.live:
                        if (
                            state.get("confirmed_identity")
                            and not state.get("session_logged")
                            and final_result["identity"] not in {"Unknown", "Scanning..."}
                        ):
                            att_logger.log_attendance(
                                final_result["identity"],
                                confidence=final_result["total_score"],
                                display_name=final_result.get("display_name"),
                            )
                            state["session_logged"] = True
                    elif final_result["identity"] not in {"Unknown", "Scanning..."}:
                        att_logger.log_attendance(
                            final_result["identity"],
                            confidence=final_result["total_score"],
                            display_name=final_result.get("display_name"),
                        )

                display_frame = draw_person_info(
                    display_frame,
                    x1,
                    y1,
                    x2,
                    y2,
                    state["result"],
                )

            cv2.imshow("Multimodal Authentication", display_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

    print("System finished.")


if __name__ == "__main__":
    main()
