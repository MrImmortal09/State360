import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp

# Load YOLOv8 for object detection (weapon detection)
yolo_model = YOLO("yolov8n.pt")  

# MediaPipe Pose for motion detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Define function for violence detection
def detect_violence(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    violence_detected = False

    # Run YOLO for object detection (filtering weapons only)
    yolo_results = yolo_model(frame_rgb)[0]
    
    for box, conf, cls in zip(yolo_results.boxes.xyxy, yolo_results.boxes.conf, yolo_results.boxes.cls):
        if conf < 0.5:  # Confidence threshold (adjust if needed)
            continue
        if int(cls) in [2, 3, 5, 6, 44, 46, 49, 50]:  # More weapon-related classes
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, "Weapon Detected", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            violence_detected = True

    # Run MediaPipe Pose for aggressive movements
    results = pose.process(frame_rgb)
    if results.pose_landmarks:
        try:
            # Extract key landmarks
            left_wrist_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y
            right_wrist_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y
            left_elbow_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y
            right_elbow_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y
            left_shoulder_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y
            right_shoulder_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y

            # Check if arms are raised (aggressive movement)
            if ((left_wrist_y < left_elbow_y < left_shoulder_y) or 
                (right_wrist_y < right_elbow_y < right_shoulder_y)):
                cv2.putText(frame, "Aggressive Movement", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                violence_detected = True

        except AttributeError:
            pass  # Handle missing landmarks gracefully

    return frame, violence_detected

# Video source: 0 for webcam, or provide a file path
video_source = 0
cap = cv2.VideoCapture(video_source)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame, violence = detect_violence(frame)

    if violence:
        cv2.putText(frame, "VIOLENCE DETECTED!", (100, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow("Violence Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
