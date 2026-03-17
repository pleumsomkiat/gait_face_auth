import numpy as np


class GaitRecognizer:
    def __init__(self):
        pass

    def _angle(self, a, b, c):
        ba = a - b
        bc = c - b
        denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-6
        cosang = np.dot(ba, bc) / denom
        cosang = np.clip(cosang, -1.0, 1.0)
        return np.arccos(cosang)

    def _normalize_frame(self, frame):
        """
        frame shape: (17, 3) -> x, y, conf
        ใช้เฉพาะ x, y
        """
        xy = frame[:, :2].astype(np.float32).copy()

        # center ด้วยกึ่งกลางสะโพก
        hip_center = (xy[11] + xy[12]) / 2.0
        xy = xy - hip_center

        # scale ด้วยความกว้างไหล่
        shoulder_width = np.linalg.norm(xy[5] - xy[6])

        # fallback เป็นความกว้างสะโพก
        if shoulder_width < 1e-6:
            shoulder_width = np.linalg.norm(xy[11] - xy[12])

        if shoulder_width > 1e-6:
            xy = xy / shoulder_width

        return xy

    def extract_features(self, sequence_of_keypoints):
        """
        Input: (N_frames, 17, 3)
        Return: gait feature vector
        """
        if sequence_of_keypoints is None or len(sequence_of_keypoints) == 0:
            return None

        seq = np.array(sequence_of_keypoints, dtype=np.float32)
        per_frame_features = []

        for frame in seq:
            if frame.shape[0] < 17:
                continue

            xy = self._normalize_frame(frame)

            l_shoulder = xy[5]
            r_shoulder = xy[6]
            l_elbow = xy[7]
            r_elbow = xy[8]
            l_wrist = xy[9]
            r_wrist = xy[10]
            l_hip = xy[11]
            r_hip = xy[12]
            l_knee = xy[13]
            r_knee = xy[14]
            l_ankle = xy[15]
            r_ankle = xy[16]

            left_knee_angle = self._angle(l_hip, l_knee, l_ankle)
            right_knee_angle = self._angle(r_hip, r_knee, r_ankle)
            left_arm_angle = self._angle(l_shoulder, l_elbow, l_wrist)
            right_arm_angle = self._angle(r_shoulder, r_elbow, r_wrist)

            ankle_distance = np.linalg.norm(l_ankle - r_ankle)
            knee_distance = np.linalg.norm(l_knee - r_knee)
            wrist_distance = np.linalg.norm(l_wrist - r_wrist)

            shoulder_center = (l_shoulder + r_shoulder) / 2.0
            hip_center = (l_hip + r_hip) / 2.0
            trunk_length = np.linalg.norm(shoulder_center - hip_center)

            feat = [
                left_knee_angle,
                right_knee_angle,
                left_arm_angle,
                right_arm_angle,
                ankle_distance,
                knee_distance,
                wrist_distance,
                trunk_length,
            ]
            per_frame_features.append(feat)

        if len(per_frame_features) == 0:
            return None

        per_frame_features = np.array(per_frame_features, dtype=np.float32)

        mean_feat = np.mean(per_frame_features, axis=0)
        std_feat = np.std(per_frame_features, axis=0)

        final_feat = np.concatenate([mean_feat, std_feat]).astype(np.float32)

        norm = np.linalg.norm(final_feat)
        if norm > 1e-6:
            final_feat = final_feat / norm

        return final_feat

    def compare(self, features1, features2):
        if features1 is None or features2 is None:
            return 0.0

        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = np.dot(features1, features2) / (norm1 * norm2)
        return float(similarity)