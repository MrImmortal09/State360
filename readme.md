# State360 - Weapon Detection System  

## Overview  
State360 is a weapon detection system using object detection models to identify weapons (guns and knives) in CCTV cameras. When a weapon is detected with high confidence, a log is stored in MongoDB with the timestamp and camera ID.  

## Installation  

### 1. Install Dependencies  
Ensure you have Python installed, then install the required dependencies:  

```bash
pip install ultralytics pymongo opencv-python numpy
```

### 2. Set Up MongoDB  

- Create a MongoDB Atlas account or set up a local MongoDB server.  
- Replace `<db_password>` in `MONGO_URI` inside the script with your actual database password.  

### Usage  

Run the script:  

```bash
python state360.py
```

