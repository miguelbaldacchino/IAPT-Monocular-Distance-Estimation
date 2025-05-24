import os
import json
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO

# Constants
CALIBRATION_DIR = 'calibrations'
VEHICLE_CLASSES = [1, 2, 3, 5, 7]  # bicycle, car, motorcycle, bus, truck
KNOWN_HEIGHT_M = 1.5
DEFAULT_VIDEO = 'footage1.mp4'
BASE_DELAY_MS = 33  # approx for 30 FPS
SPEED_OPTIONS = [0.25, 1.0, 4.0, 7.0]
CLOSE_THRESHOLD = 1.5  # metres
TOO_CLOSE_THRESHOLD = 0.8  # metres

# Utility functions
def save_focal_length(focal_px, name):
    os.makedirs(CALIBRATION_DIR, exist_ok=True)
    path = os.path.join(CALIBRATION_DIR, f"{name}.json")
    with open(path, 'w') as f:
        json.dump({'focal_px': focal_px}, f)
    messagebox.showinfo('Saved', f'Calibration "{name}" stored.')

def delete_focal(name):
    path = os.path.join(CALIBRATION_DIR, f"{name}.json")
    if os.path.exists(path):
        os.remove(path)
        messagebox.showinfo('Deleted', f'Calibration "{name}" removed.')
    else:
        messagebox.showerror('Error', f'Camera "{name}" not found.')

def load_focal_list():
    cams = []
    if not os.path.isdir(CALIBRATION_DIR):
        return cams
    for fn in os.listdir(CALIBRATION_DIR):
        if fn.endswith('.json'):
            name = fn[:-5]
            try:
                with open(os.path.join(CALIBRATION_DIR, fn)) as f:
                    data = json.load(f)
                cams.append((name, data.get('focal_px')))
            except:
                pass
    return cams

def load_focal(name):
    path = os.path.join(CALIBRATION_DIR, f"{name}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f).get('focal_px')
    return None

def estimate_distance(bbox_h, focal, real_h):
    return (real_h * focal) / max(bbox_h, 1)

def get_rotated_box(x, y, w, h, angle):
    cx, cy = x + w/2, y + h/2
    pts = np.array([[-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]])
    theta = np.deg2rad(angle)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    rot = pts.dot(R.T)
    box = rot + np.array([cx, cy])
    return box.astype(int)

class DistanceEstimatorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Vehicle Distance Estimator')
        self.geometry('800x600')
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        # State
        self.video_path = tk.StringVar(value=DEFAULT_VIDEO)
        self.focal = None
        self.real_h = KNOWN_HEIGHT_M
        self.cap = None
        self.running = False
        self.speed = tk.DoubleVar(value=1.0)
        self.model = YOLO('yolov8n.pt')

        # Video display
        self.img_label = tk.Label(self)
        self.img_label.grid(row=0, column=0, sticky='nsew')

        # Controls
        ctrl = tk.Frame(self)
        ctrl.grid(row=1, column=0, sticky='ew')
        ctrl.columnconfigure(1, weight=1)

        tk.Label(ctrl, text='Video File:').grid(row=0, column=0, padx=5, pady=5)
        tk.Entry(ctrl, textvariable=self.video_path).grid(row=0, column=1, padx=5, pady=5, sticky='ew')
        tk.Button(ctrl, text='Browse', command=self.browse_video).grid(row=0, column=2, padx=5, pady=5)

        tk.Button(ctrl, text='Calibrate New', command=self.calibrate_new).grid(row=1, column=0, columnspan=3, pady=5)

        tk.Label(ctrl, text='Saved Cameras:').grid(row=2, column=0, padx=5)
        self.cam_var = tk.StringVar()
        self.cam_menu = tk.OptionMenu(ctrl, self.cam_var, *[c[0] for c in load_focal_list()])
        self.cam_menu.grid(row=2, column=1, sticky='ew', padx=5)
        tk.Button(ctrl, text='Load', command=self.load_camera).grid(row=2, column=2, padx=5)
        tk.Button(ctrl, text='Delete', command=self.delete_camera).grid(row=2, column=3, padx=5)

        tk.Label(ctrl, text='Speed:').grid(row=3, column=0, padx=5, pady=5)
        speed_menu = tk.OptionMenu(ctrl, self.speed, *SPEED_OPTIONS)
        speed_menu.grid(row=3, column=1, padx=5, pady=5)

        # Playback Controls: Start, Stop, Resume, Exit
        tk.Button(ctrl, text='Start', command=self.start).grid(row=4, column=0, pady=5)
        tk.Button(ctrl, text='Stop', command=self.stop).grid(row=4, column=1)
        tk.Button(ctrl, text='Resume', command=self.resume).grid(row=4, column=2)
        tk.Button(ctrl, text='Exit', command=self.exit).grid(row=4, column=3)

        self.update_cam_list()

    def browse_video(self):
        path = filedialog.askopenfilename(filetypes=[('MP4 Files','*.mp4')])
        if path:
            self.video_path.set(path)

    def update_cam_list(self):
        menu = self.cam_menu['menu']
        menu.delete(0, 'end')
        for name, _ in load_focal_list():
            menu.add_command(label=name, command=lambda v=name: self.cam_var.set(v))
        self.cam_var.set('')

    def load_camera(self):
        fl = load_focal(self.cam_var.get())
        if fl:
            self.focal = fl
            messagebox.showinfo('Loaded', f'Focal length: {fl:.1f}px')

    def delete_camera(self):
        name = self.cam_var.get()
        if not name:
            messagebox.showerror('Error','No camera selected')
            return
        if messagebox.askyesno('Confirm', f'Delete camera "{name}"?'):
            delete_focal(name)
            self.update_cam_list()
            self.focal = None

    def calibrate_new(self):
        path = self.video_path.get()
        if not os.path.isfile(path):
            messagebox.showerror('Error','Video not found')
            return
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(path)
        ret, frame = self.cap.read()
        if not ret:
            messagebox.showerror('Error','Cannot read video')
            return
        # Axis-aligned selection
        x, y, w, h = cv2.selectROI('Select Reference Object', frame, False, False)
        cv2.destroyWindow('Select Reference Object')
        if w == 0 or h == 0:
            return
        # Compute pixel height\้ว
        pix_h = h
        real_d = simpledialog.askfloat('Distance','Enter estimated distance (m):')
        if real_d is None:
            return
        self.focal = (pix_h * real_d) / self.real_h
        messagebox.showinfo('Calibration', f'Focal length: {self.focal:.1f}px')
        name = simpledialog.askstring('Name','Enter camera name:')
        if name:
            save_focal_length(self.focal, name)
            self.update_cam_list()

    def start(self):
        if not self.focal:
            messagebox.showerror('Error','Load or calibrate first')
            return
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.video_path.get())
        self.running = True
        self.process_frame()

    def stop(self):
        self.running = False

    def resume(self):
        if not self.running and self.cap:
            self.running = True
            self.process_frame()

    def process_frame(self):
        if not self.running:
            return
        speed = self.speed.get()
        ret, frame = self.cap.read()
        if not ret:
            self.running = False
            return
        for _ in range(max(int(speed)-1, 0)):
            self.cap.read()
        close = False
        too_close = False
        res = self.model(frame, conf=0.4, verbose=False)[0]
        for box in res.boxes:
            cls_id = int(box.cls[0])
            if cls_id in VEHICLE_CLASSES:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                h = y2 - y1
                dist = estimate_distance(h, self.focal, self.real_h)
                if dist < TOO_CLOSE_THRESHOLD:
                    too_close = True
                    color = (0, 0, 255)
                elif dist < CLOSE_THRESHOLD:
                    close = True
                    color = (0, 0, 150)
                else:
                    color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{res.names[cls_id]} {dist:.1f}m", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        # Overlays
        if too_close:
            overlay = frame.copy()
            overlay[:] = (0, 0, 255)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            cv2.putText(frame, 'WARNING: Object Too Close', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 3)
        elif close:
            overlay = frame.copy()
            overlay[:] = (0, 0, 150)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            cv2.putText(frame, 'Caution: Object Close', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 3)
        # Display frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(img)
        self.img_label.imgtk = imgtk
        self.img_label.config(image=imgtk)
        delay = int(BASE_DELAY_MS / speed)
        self.after(delay, self.process_frame)

    def exit(self):
        if self.cap:
            self.cap.release()
        self.destroy()

if __name__ == '__main__':
    app = DistanceEstimatorApp()
    app.mainloop()
