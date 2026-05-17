import cv2
import numpy as np
from ultralytics import YOLO
from sort import *
import os
import math

# --- 1. Configuration ---
VIDEO_PATH = "../assets/Videos/trafic_camera.mp4" 
MODEL_PATH = "../model/yolov8s.pt"
SAVE_DIR = "violations"

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

model = YOLO(MODEL_PATH)

# Tracker tuned to hold onto vehicles for a long time (max_age=90) to prevent ID swapping
tracker = Sort(max_age=90, min_hits=1, iou_threshold=0.2)

# --- 2. Line Placement ---
line_1 = [300, 350, 900, 350]  # Green Line (Start)
line_2 = [150, 650, 1050, 650] # Red Line (Finish)

# --- 3. Real-World Calibration ---
DISTANCE = 35  # Meters
SPEED_LIMIT = 60 # km/h 

# --- 4. Data Storage (The State Machine) ---
track_data = {}      # Stores the exact status and start frame of every car
vehicle_speeds = {}  
violation_count = 0
frame_count = 0      

# --- 5. Video Setup & FPS ---
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"\n🚨 ERROR: Could not open video file. 🚨\n")
    exit()

fps = 30 # Hardcoded for mathematical stability

# --- 6. Main Processing Loop ---
while True:
    success, img = cap.read()
    if not success: 
        print("Video ended.")
        break
    
    frame_count += 1 
    
    results = model(img, stream=True)
    detections = np.empty((0, 5))

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            
            # Vehicles only
            if cls in [2, 3, 5, 7] and conf > 0.30:
                current_array = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, current_array))

    tracks = tracker.update(detections)

    for trk in tracks:
        x1, y1, x2, y2, track_id = map(int, trk)
        
        cx = (x1 + x2) // 2
        cy = y2 - 5 # Tracking the bottom tires
        
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        # --- THE STATE MACHINE LOGIC ---
        
        # 1. Initialize the car's state if we haven't seen it before
        if track_id not in track_data:
            if cy < line_1[1]:
                # Car spawned BEFORE the green line. Perfect.
                track_data[track_id] = {'state': 'approaching'}
            else:
                # Car spawned in the middle of the lines or below the red line. Ignore it.
                track_data[track_id] = {'state': 'ignored'}

        # Get the car's current state
        current_state = track_data[track_id]['state']

        # 2. Transition: Approaching -> Crossed Green
        if current_state == 'approaching' and cy >= line_1[1]:
            track_data[track_id]['state'] = 'timer_started'
            track_data[track_id]['start_frame'] = frame_count

        # 3. Transition: Crossed Green -> Crossed Red (Calculate Speed!)
        elif current_state == 'timer_started' and cy >= line_2[1]:
            # Lock the state so it never calculates again
            track_data[track_id]['state'] = 'finished' 
            
            frames_elapsed = frame_count - track_data[track_id]['start_frame']
            
            if frames_elapsed > 0:
                time_taken_seconds = frames_elapsed / fps
                speed_ms = DISTANCE / time_taken_seconds
                speed_kmh = int(speed_ms * 3.6)

                vehicle_speeds[track_id] = speed_kmh

                # Save Snapshot if speeding
                if speed_kmh > SPEED_LIMIT:
                    violation_count += 1
                    filename = f"{SAVE_DIR}/id_{track_id}_speed_{speed_kmh}.jpg"
                    
                    h, w = img.shape[:2]
                    crop_y1, crop_y2 = max(0, y1), min(h, y2)
                    crop_x1, crop_x2 = max(0, x1), min(w, x2)
                    
                    if crop_y2 > crop_y1 and crop_x2 > crop_x1:
                        cv2.imwrite(filename, img[crop_y1:crop_y2, crop_x1:crop_x2]) 
                    
                    print(f"📸 VIOLATION! ID {track_id} at {speed_kmh} km/h")

        # --- Display Speed on Screen ---
        if track_id in vehicle_speeds:
            cv2.putText(img, f"{vehicle_speeds[track_id]} km/h", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # --- 7. Visualization Graphics ---
    cv2.line(img, (line_1[0], line_1[1]), (line_1[2], line_1[3]), (0, 255, 0), 3) 
    cv2.line(img, (line_2[0], line_2[1]), (line_2[2], line_2[3]), (0, 0, 255), 3) 
    
    cv2.putText(img, f"Violations: {violation_count}", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

    cv2.imshow("Speed Detection System", img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()