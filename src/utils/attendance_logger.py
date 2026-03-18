import os
import csv
import time
from datetime import datetime


class AttendanceLogger:
    def __init__(self, log_path="logs/attendance.csv", cooldown_seconds=60):
        self.log_path = log_path
        self.cooldown_seconds = cooldown_seconds
        self.last_logged = {}  # user_id -> last timestamp

        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "user_id", "confidence"])

    def log_attendance(self, user_id, confidence=0.0):
        now_ts = time.time()

        if user_id in self.last_logged:
            elapsed = now_ts - self.last_logged[user_id]
            if elapsed < self.cooldown_seconds:
                return False

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, user_id, f"{confidence:.4f}"])

        self.last_logged[user_id] = now_ts
        return True