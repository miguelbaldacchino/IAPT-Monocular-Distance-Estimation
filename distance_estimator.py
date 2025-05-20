import cv2
from ultralytics import YOLO

# === USER PARAMETERS ===
VIDEO_PATH        = 'footage1.mp4'

# You can change these defaults or enter at runtime
KNOWN_HEIGHT_M    = 1.5  # e.g. 1.5 for a typical car, or None to prompt
KNOWN_DIST_M      = None  # e.g. 5.0 metres, or None to prompt

# COCO vehicle classes
VEHICLE_CLASSES   = [1, 2, 3, 5, 7]  # bicycle, car, motorcycle, bus, truck

def calibrate_focal_length(frame):
    # Draw ROI
    roi = cv2.selectROI("Calibration – Select Reference Object", frame, False, False)
    cv2.destroyWindow("Calibration – Select Reference Object")
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
    """ Pinhole formula: d = (H_real * f_px) / h_image """
    return (real_h * focal_px) / max(bbox_height_px, 1)

def main():
    # 1. Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Error: cannot open video.")
        return

    # 2. Grab a frame for calibration
    ret, calib_frame = cap.read()
    if not ret:
        print("Error: cannot read frame for calibration.")
        return

    # 3. Calibrate focal length
    focal_px = calibrate_focal_length(calib_frame)
    real_h = KNOWN_HEIGHT_M if KNOWN_HEIGHT_M else float(input("Re-enter real object height (m): "))
    
    # 4. Load YOLOv8 model
    yolo = YOLO('yolov8n.pt')

    # 5. Process video from start
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 5a. Detect vehicles
        res = yolo(frame, conf=0.4, verbose=False)[0]

        # 5b. Draw boxes & estimate distance
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

        # 5c. Display
        cv2.imshow('Distance Estimation', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
