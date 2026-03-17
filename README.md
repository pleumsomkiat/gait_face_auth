# Multimodal Biometrics: Gait and Face Authentication

This project implements a system for authenticating individuals using both long-distance face recognition and gait (walking posture) analysis.

## Project Structure
- `data1/`: Contains images of faces to be scanned/registered.
- `data2/`: Contains videos of people walking for testing and gait analysis.
- `src/`: Source code directory.
  - `face_module/`: Face detection and recognition.
  - `gait_module/`: Pose estimation and gait recognition.
  - `tracker/`: Person tracking in videos.
  - `utils/`: Helper functions.

## Setup
1. Use `pip install -r requirements.txt` to install dependencies.
2. Put face images in `data1/` and video files in `data2/`.
3. Run the system using `python src/main.py`.
