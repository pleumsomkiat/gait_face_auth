class PersonTracker:
    def __init__(self, model):
        # We pass the YOLO model to use its .track() method
        self.model = model

    def track(self, frame):
        # Using YOLOv8's built-in tracking
        results = self.model.track(frame, persist=True, verbose=False)
        return results

