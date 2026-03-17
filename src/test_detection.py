import cv2
import argparse
from ultralytics import YOLO
from deepface import DeepFace

def test_on_image(image_path, yolo_model_path='yolov8n-pose.pt'):
    print(f"Testing on image: {image_path}")
    
    # 1. Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return

    # 2. Run Pose Estimation (YOLOv8-Pose)
    print("Running YOLOv8-Pose...")
    pose_model = YOLO(yolo_model_path)
    # The results object contains bounding boxes and keypoints
    results = pose_model(img, verbose=False)
    
    # Draw Pose results on image (ultralytics has a built-in plot function)
    # This draws skeletons and bounding boxes for persons
    annotated_img = results[0].plot()

    # 3. Run Face Detection (DeepFace)
    print("Running DeepFace...")
    try:
        # We use opencv backend for basic detection, you can change to mtcnn or retinaface
        faces = DeepFace.extract_faces(img_path=image_path, detector_backend='opencv', enforce_detection=False)
        print(f"Detected {len(faces)} face(s).")
        
        # Draw face bounding boxes
        for face_obj in faces:
            # facial_area contains x, y, w, h
            facial_area = face_obj['facial_area']
            x = facial_area['x']
            y = facial_area['y']
            w = facial_area['w']
            h = facial_area['h']
            
            # Draw a green rectangle for the face
            cv2.rectangle(annotated_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(annotated_img, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
    except ValueError as e:
        print(f"DeepFace detect error: {e}")

    # 4. Show results
    window_name = "Detection Test (Pose & Face)"
    cv2.imshow(window_name, annotated_img)
    print("Press any key to close the window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Face and Pose Detection")
    parser.add_argument('--image', type=str, required=True, help='Path to the test image')
    args = parser.parse_args()
    
    test_on_image(args.image)
