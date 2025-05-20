import cv2
from ultralytics import YOLO
import json
import os

VIDEO_PATH        = 'footage1.mp4'

KNOWN_HEIGHT_M    = 1.5 
KNOWN_DIST_M      = None  # e.g. 5.0 metres, or None to prompt

VEHICLE_CLASSES   = [1, 2, 3, 5, 7]  # bicycle, car, motorcycle, bus, truck

CALIBRATION_DIR = 'calibrations'


def save_focal_length(focal_px, name):
    os.makedirs(CALIBRATION_DIR, exist_ok=True)
    path = os.path.join(CALIBRATION_DIR, f'{name}.json')
    with open(path, "w") as f:
        json.dump({'focal_px': focal_px}, f)
    print(f'Saved - Calibration {name}')
        

def load_focal_length(name):
    path = os.path.join(CALIBRATION_DIR, f"{name}.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
            print(f'Loading Calibration {name}.json')
            return data.get('focal_px')
    else:
        print(f"[Load Error] Calibration '{name}' not found.")
        return None
    
def list_focal_lengths():
    if not os.path.exists(CALIBRATION_DIR):
        print('Calibration directory not found. Create a new camera.')
        return None
    
    files = [f for f in os.listdir(CALIBRATION_DIR) if f.endswith('.json')]
    if not files:
        print('No calibration files found. Create a new camera.')
        return None
        
    print('=== Available Cameras ===\n')
    for file in files:
        path = os.path.join(CALIBRATION_DIR, file)
        with open(path, 'r') as f:
            try:
                data = json.load(f)
                focal = data.get('focal_px', 'N/A')
                print(f" - {file.replace('.json', '')}: {focal:.1f}px")
            except Exception as e:
                print(f' - {file}: Error reading file >> {e}')
                return None
    print('- Or Create a New One (enter 0)\n')
    return True
    
    
def calibrate_focal_length(frame):
    # Draw ROI
    roi = cv2.selectROI("Calibration: Select Reference Object", frame, False, False)
    cv2.destroyWindow("Calibration: Select Reference Object")
    x, y, w, h = roi
    pix_h = h

    # Prompt for real-world measurements if not provided
    real_h = KNOWN_HEIGHT_M 
    real_d = KNOWN_DIST_M  or float(input("Enter distance to object in metres (e.g. 5.0): "))

    # Compute focal length (px)
    focal_px = (pix_h * real_d) / real_h
    print(f"[Calibration] Pixel height: {pix_h}px → Focal length = {focal_px:.1f}px")
    return focal_px

def estimate_distance_m(bbox_height_px, focal_px, real_h):
    return (real_h * focal_px) / max(bbox_height_px, 1)

def createCamera(calibration_frame):
    focal_px = calibrate_focal_length(calibration_frame)
    # Saving
    saveCamera = input('Would you like to save your camera? Y/N\n')
    match saveCamera:
        # Yes
        case 'Y': 
            cameraName = input('Name Your Camera\n')
            save_focal_length(focal_px=focal_px, name=cameraName)
            return focal_px
        # No
        case 'N':
            return focal_px
        # Default
        case _:
            print('Invalid Input. Restarting')
            return None
            
def main():
    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Error: cannot open video.")
        return

    # Grab a frame for calibration
    ret, calib_frame = cap.read()
    if not ret:
        print("Error: cannot read frame for calibration.")
        return

    # Focal Length Selection
    focal_px = 0
    
    #List all cameras
    
    while True:
        # Choose focal from list, or create new (0)
        listFocal = list_focal_lengths()
        if listFocal == None:
            print('Creating New Camera')
            focal_px = createCamera(calib_frame)
            if focal_px != None:
                break
        chosenFocal = input('Choose a Camera, or Create a New One (0)\n')
        # Create new 
        if chosenFocal == '0':
            focal_px = createCamera(calib_frame)
            if focal_px != None:
                break
        # Loading previous Camera
        focal_px = load_focal_length(chosenFocal)
        # If camera found
        if focal_px != None:
            break
        else:
            print('Camera not found. Name might be wrong. Restarting.')
    print('=== Loading Distance Estimation ===')
    real_h = KNOWN_HEIGHT_M if KNOWN_HEIGHT_M else float(input("Re-enter real object height (m): "))
    
    # Load YOLOv8 model
    yolo = YOLO('yolov8n.pt')

    # Process video from start
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect vehicles
        res = yolo(frame, conf=0.4, verbose=False)[0]

        # Draw boxes & estimate distance
        for box in res.boxes:
            cls_id = int(box.cls[0])
            if cls_id in VEHICLE_CLASSES:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                pix_h = y2 - y1
                dist_m = estimate_distance_m(pix_h, focal_px, real_h)

                color = (0,0,255) if dist_m < 2.0 else (0,255,0)
                label = f"{res.names[cls_id]} {dist_m:.1f}m"

                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                cv2.putText(frame, label, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Display
        cv2.imshow('Distance Estimation', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
