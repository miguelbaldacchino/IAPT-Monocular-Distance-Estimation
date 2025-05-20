# main.py

import cv2
from ultralytics import YOLO

# === Constants (from your calibration) ===
KNOWN_HEIGHT_M   = 1.5      # meters (e.g. average car height)
FOCAL_LENGTH_PX  = 1333.3   # px (output from calibrate.py)

# COCO vehicle classes
VEHICLE_CLASSES = [1, 2, 3, 5, 7]  # bicycle, car, motorcycle, bus, truck

# Distance estimator
def estimate_distance_m(bbox_height_px):
    # d = (H_real * f) / h_image
    return (KNOWN_HEIGHT_M * FOCAL_LENGTH_PX) / max(bbox_height_px, 1)

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Open video
cap = cv2.VideoCapture('footage1.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1) Detect
    res = model(frame, conf=0.4, verbose=False)[0]

    # 2) Draw & compute distance
    for box in res.boxes:
        cls_id = int(box.cls[0])
        if cls_id in VEHICLE_CLASSES:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            pix_h = y2 - y1
            dist_m = estimate_distance_m(pix_h)

            # Color code: red if <2 m, green otherwise
            color = (0,0,255) if dist_m < 2.0 else (0,255,0)
            label = f"{res.names[cls_id]} {dist_m:.1f}m"

            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 3) Display
    cv2.imshow('Vehicles + Distance', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
