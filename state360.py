import cv2
import time
import concurrent.futures
from pymongo import MongoClient
from datetime import datetime
from ultralytics import YOLO

# MongoDB Configuration
MONGO_URI = "mongodb+srv://mrimmortaldrive:omswami2004@cluster0.kw1su.mongodb.net/"
client = MongoClient(MONGO_URI)
db = client['WeaponDetection']
collection = db['logs']

# Load YOLO Models
model_gun = YOLO('ModelWeights/gun_nano_best.pt')
model_knife = YOLO('ModelWeights/Kniife_best.pt')

# Class Names
classNames = ["Gun"]
classNames1 = ["Knife"]

# Weapon Detection Function
def video_detection(video_path, camera_id):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video source {camera_id}.")
        return

    last_logged_time = None

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            print("End of video stream or camera disconnected.")
            break

        results_gun = model_gun.predict(img, conf=0.5)
        results_knife = model_knife.predict(img, conf=0.5)

        for r in results_gun + results_knife:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = round(float(box.conf[0]), 2)
                cls = int(box.cls[0])
                class_name = classNames[cls] if r in results_gun else classNames1[cls]
                label = f'{class_name} {conf * 100:.1f}%'

                color = (0, 255, 0) if conf >= 0.5 else (0, 0, 255)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                if conf >= 0.85:
                    current_time = datetime.now()
                    if last_logged_time is None or (current_time - last_logged_time).seconds > 600:
                        log_data = {
                            "timestamp": current_time,
                            "camera_id": camera_id,
                            "detected_object": class_name
                        }
                        collection.insert_one(log_data)
                        last_logged_time = current_time

        cv2.imshow(f"Camera {camera_id} - Weapon Detection", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Multiple Camera Inputs
video_sources = [(0, "Camera 1"), ("video.mp4", "Camera 2")]

with concurrent.futures.ThreadPoolExecutor() as executor:
    for video_path, camera_id in video_sources:
        executor.submit(video_detection, video_path, camera_id)
