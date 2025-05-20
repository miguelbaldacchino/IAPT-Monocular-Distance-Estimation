# calibrate.py

import cv2

# === User inputs ===
KNOWN_HEIGHT_M   = 1.5   # real car height in meters
KNOWN_DIST_M     = 5.0   # distance at which you measured the car
CALIB_FRAME_PATH = 'calibration_frame.jpg'  # extract from your video

# === Load image & draw a box around the reference object ===
img = cv2.imread(CALIB_FRAME_PATH)
# (Here you’d detect the car or manually draw the box:
#  e.g. x1,y1,x2,y2 = cv2.selectROI(...))
x1, y1, x2, y2 = 100, 200, 500, 650  # replace with your ROI

pix_h = y2 - y1
focal_length_px = (pix_h * KNOWN_DIST_M) / KNOWN_HEIGHT_M
print(f"Measured pixel height: {pix_h}px")
print(f"Computed focal length: {focal_length_px:.1f}px")
