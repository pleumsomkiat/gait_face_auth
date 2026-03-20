import os
import csv
import json
import time
import threading
from datetime import datetime
from urllib import error, request


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
DEFAULT_LOG_PATH = os.path.join(PROJECT_ROOT, "logs", "attendance.csv")
DEFAULT_REMOTE_API_URL = "https://new-data2.onrender.com/checkin"
DEFAULT_USER_ID_MAP_PATH = os.path.join(PROJECT_ROOT, "config", "user_id_map.json")


class AttendanceLogger:
    def __init__(
        self,
        log_path=DEFAULT_LOG_PATH,
        cooldown_seconds=60,
        remote_api_url=None,
        user_id_map_path=DEFAULT_USER_ID_MAP_PATH,
        remote_timeout_seconds=3.0,
    ):
        self.log_path = log_path
        self.cooldown_seconds = cooldown_seconds
        self.remote_api_url = (
            remote_api_url
            or os.getenv("ATTENDANCE_API_URL")
            or DEFAULT_REMOTE_API_URL
        )
        self.user_id_map_path = user_id_map_path
        self.remote_timeout_seconds = remote_timeout_seconds
        self.last_logged = {}  # subject key -> last timestamp
        self.user_id_map = self._load_user_id_map()

        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "user_id", "confidence"])

    def _load_user_id_map(self):
        if not self.user_id_map_path or not os.path.exists(self.user_id_map_path):
            return {}

        try:
            with open(self.user_id_map_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
        except Exception as exc:
            print(f"Failed to load user ID map: {exc}")
            return {}

        if not isinstance(raw_data, dict):
            return {}

        return {str(key): str(value) for key, value in raw_data.items()}

    def resolve_remote_user_id(self, user_id, display_name=None):
        candidates = [display_name, user_id]
        for candidate in candidates:
            if not candidate:
                continue
            mapped_value = self.user_id_map.get(str(candidate))
            if mapped_value:
                return mapped_value

        return str(display_name or user_id)

    def post_remote_checkin(self, remote_user_id):
        if not self.remote_api_url:
            return False, "remote disabled"

        payload = json.dumps({"user_id": remote_user_id}).encode("utf-8")
        req = request.Request(
            self.remote_api_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=self.remote_timeout_seconds) as resp:
                body = resp.read().decode("utf-8", errors="replace")
                try:
                    body_json = json.loads(body)
                    message = body_json.get("message") or body
                except Exception:
                    message = body
                return True, message
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            return False, f"HTTP {exc.code}: {body}"
        except Exception as exc:
            return False, str(exc)

    def sync_remote_checkin_async(self, remote_user_id):
        worker = threading.Thread(
            target=self._sync_remote_checkin_worker,
            args=(remote_user_id,),
            daemon=True,
        )
        worker.start()

    def _sync_remote_checkin_worker(self, remote_user_id):
        remote_ok, remote_message = self.post_remote_checkin(remote_user_id)
        if remote_ok:
            print(f"Remote check-in synced: {remote_user_id} ({remote_message})")
        else:
            print(f"Remote check-in failed: {remote_user_id} ({remote_message})")

    def log_attendance(self, user_id, confidence=0.0, display_name=None):
        now_ts = time.time()

        if user_id in self.last_logged:
            elapsed = now_ts - self.last_logged[user_id]
            if elapsed < self.cooldown_seconds:
                return False

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, user_id, f"{confidence:.4f}"])

        remote_user_id = self.resolve_remote_user_id(user_id, display_name=display_name)
        self.sync_remote_checkin_async(remote_user_id)

        self.last_logged[user_id] = now_ts
        return True
