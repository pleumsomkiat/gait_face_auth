from ultralytics import YOLO

class PoseEstimator:
    def __init__(self, model_path='yolov8n-pose.pt'):
        # Initialize YOLOv8-Pose model
        self.model = YOLO(model_path)

    def estimate(self, image):
        # image can be a numpy array or path to image
        results = self.model(image, verbose=False)
        return results

