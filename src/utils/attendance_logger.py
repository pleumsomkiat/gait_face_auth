import csv
import os
from datetime import datetime

class AttendanceLogger:
    def __init__(self, log_path='logs/attendance.csv'):
        self.log_path = log_path
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self._init_log_file()

    def _init_log_file(self):
        """Create the CSV file with headers if it doesn't exist."""
        if not os.path.exists(self.log_path):
            with open(self.log_path, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Timestamp', 'User ID', 'Method', 'Confidence'])

    def log_attendance(self, user_id, method='Face+Gait', confidence=1.0):
        """Record a successful authentication entry."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.log_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, user_id, method, f"{confidence:.2f}"])
        print(f"Attendance logged for user: {user_id} at {timestamp}")

if __name__ == "__main__":
    # Test
    logger = AttendanceLogger()
    logger.log_attendance('test_user', confidence=0.95)
