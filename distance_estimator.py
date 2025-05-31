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
OBJECT_HEIGHTS = {
    'car': 1.5,
    'person': 1.6,
    'motorcycle': 1.2,
    'bicycle': 1.1,
    'truck': 2.2,
    'bus': 3.0
}
KNOWN_HEIGHT_M = 1.5
DEFAULT_VIDEO = 'footage1.mp4'
BASE_DELAY_MS = 33  # approx for 30 FPS
SPEED_OPTIONS = [1.0, 4.0, 7.0, 12.0, 15.0]
CLOSE_THRESHOLD = 1.5  # metres
TOO_CLOSE_THRESHOLD = 1.2  # metres

# Utility functions
def saveFocal(focal_px, name):
    os.makedirs(CALIBRATION_DIR, exist_ok=True)
    path = os.path.join(CALIBRATION_DIR, f"{name}.json")
    with open(path, 'w') as f:
        json.dump({'focal_px': focal_px}, f)
    messagebox.showinfo('Saved', f'Calibration "{name}" stored.')

def deleteFocal(name):
    path = os.path.join(CALIBRATION_DIR, f"{name}.json")
    if os.path.exists(path):
        os.remove(path)
        messagebox.showinfo('Deleted', f'Calibration "{name}" removed.')
    else:
        messagebox.showerror('Error', f'Camera "{name}" not found.')

def loadFocalsList():
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

def loadFocal(name):
    path = os.path.join(CALIBRATION_DIR, f"{name}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f).get('focal_px')
    return None

def estimateDistance(bbox_h, focal, real_h):
    return (real_h * focal) / max(bbox_h, 1)

class DistanceEstimatorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.ignore_low_persons = tk.BooleanVar(value=True)
        self.use_double_decker = tk.BooleanVar(value=False)

        self.title('Distance Estimator')
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
        tk.Button(ctrl, text='Browse', command=self.browseVideo).grid(row=0, column=2, padx=5, pady=5)

        tk.Button(ctrl, text='Calibrate New', command=self.calibrateNew).grid(row=1, column=0, columnspan=3, pady=5)

        tk.Label(ctrl, text='Saved Cameras:').grid(row=2, column=0, padx=5)
        self.cam_var = tk.StringVar()
        camera_names = [c[0] for c in loadFocalsList()]
        default_value = camera_names[0] if camera_names else 'Select Camera'
        self.cam_var.set(default_value)
        self.cam_menu = tk.OptionMenu(ctrl, self.cam_var, *camera_names) if camera_names else tk.OptionMenu(ctrl, self.cam_var, default_value)
        self.cam_menu.grid(row=2, column=1, sticky='ew', padx=5)
        tk.Button(ctrl, text='Load', command=self.loadCamera).grid(row=2, column=2, padx=5)
        tk.Button(ctrl, text='Delete', command=self.deleteCamera).grid(row=2, column=3, padx=5)

        tk.Label(ctrl, text='Speed:').grid(row=3, column=0, padx=5, pady=5)
        speed_menu = tk.OptionMenu(ctrl, self.speed, *SPEED_OPTIONS)
        speed_menu.grid(row=3, column=1, padx=5, pady=5)

        # Nicer centered Playback Controls
        button_row = tk.Frame(ctrl)
        button_row.grid(row=4, column=0, columnspan=4, pady=10)

        btn_style = {'padx': 10, 'pady': 5}

        tk.Button(button_row, text='Preview', command=self.start).pack(side='left', **btn_style)
        tk.Button(button_row, text='Stop', command=self.stop).pack(side='left', **btn_style)
        tk.Button(button_row, text='Resume', command=self.resume).pack(side='left', **btn_style)
        tk.Button(button_row, text='Exit', command=self.exit).pack(side='left', **btn_style)

        tk.Button(button_row, text='Export Full Video', command=self.exportFullVideo).pack(side='left', **btn_style)
        tk.Checkbutton(ctrl, text='Ignore close-up persons (bottom 1/3)', variable=self.ignore_low_persons).grid(row=5, column=0, columnspan=2, padx=5, pady=5, sticky='w')
        tk.Checkbutton(ctrl, text='Use Double-Decker Bus (4.0 m)', variable=self.use_double_decker).grid(row=6, column=0, columnspan=2, padx=5, pady=5, sticky='w')


        self.updateCameraList()

    def browseVideo(self):
        path = filedialog.askopenfilename(filetypes=[('MP4 Files','*.mp4')])
        if path:
            self.video_path.set(path)

    def exportFullVideo(self):
        if not self.focal:
            messagebox.showerror('Error', 'Load or calibrate first')
            return

        path = self.video_path.get()
        if not os.path.isfile(path):
            messagebox.showerror('Error', 'Video not found')
            return

        cap = cv2.VideoCapture(path)
        save_path = filedialog.asksaveasfilename(defaultextension=".mp4",
                                                filetypes=[("MP4 files", "*.mp4")],
                                                title="Export Full Processed Video As")
        if not save_path:
            return

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            current += 1

            close = False
            too_close = False
            res = self.model(frame, conf=0.4, verbose=False)[0]
            for box in res.boxes:
                cls_id = int(box.cls[0])
                label = res.names[cls_id]
                if cls_id in VEHICLE_CLASSES or label.lower() == 'person':
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    h = y2 - y1
                    if label in OBJECT_HEIGHTS:
                        real_h = OBJECT_HEIGHTS.get(label.lower(), self.real_h)
                        frame_h = frame.shape[0]
                        if label.lower() == 'bus':
                            real_h = 4.0 if self.use_double_decker.get() else 3.0
                        if label.lower() == 'truck':
                            # Heuristic: adjust height for large trucks
                            if h > 180:  # Adjust this threshold as needed
                                real_h = 2.8  # large truck
                            if h > 230:  # Adjust this threshold as needed
                                real_h = 3.2  # larger truck
                    if label == 'person' and self.ignore_low_persons.get() and y2 > frame_h * (2/3):
                        continue
                    dist = estimateDistance(h, self.focal, real_h)
                    if dist < TOO_CLOSE_THRESHOLD:
                        too_close = True
                        color = (0, 0, 255)  # Red for 'too close'
                    elif dist < CLOSE_THRESHOLD:
                        close = True
                        color = (0, 165, 255)  # Orange for 'close'
                    else:
                        color = (0, 255, 0)  # Green for safe
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{label} {dist:.1f}m", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if too_close:
                overlay = frame.copy()
                overlay[:] = (0, 0, 255)
                cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                cv2.putText(frame, 'WARNING: Object Too Close', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 3)
            elif close:
                overlay = frame.copy()
                cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                cv2.putText(frame, 'Caution: Object Close', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 3)

            writer.write(frame)

            # Optional: Show simple progress every 50 frames
            if current % 50 == 0:
                print(f"Processed frame {current}/{total}")

        cap.release()
        writer.release()
        messagebox.showinfo("Export Complete", f"Full video saved to:\n{save_path}")


    def updateCameraList(self):
        menu = self.cam_menu['menu']
        menu.delete(0, 'end')
        for name, _ in loadFocalsList():
            menu.add_command(label=name, command=lambda v=name: self.cam_var.set(v))
        self.cam_var.set('')

    def loadCamera(self):
        fl = loadFocal(self.cam_var.get())
        if fl:
            self.focal = fl
            messagebox.showinfo('Loaded', f'Focal length: {fl:.1f}px')

    def deleteCamera(self):
        name = self.cam_var.get()
        if not name:
            messagebox.showerror('Error','No camera selected')
            return
        if messagebox.askyesno('Confirm', f'Delete camera "{name}"?'):
            deleteFocal(name)
            self.updateCameraList()
            self.focal = None

    def calibrateNew(self):
        path = self.video_path.get()
        if not os.path.isfile(path):
            messagebox.showerror('Error', 'Video not found')
            return

        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(path)

        found = False
        frame = None
        label = None

        # Find first frame with a known object
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            res = self.model(frame, conf=0.4, verbose=False)[0]
            for box in res.boxes:
                cls_id = int(box.cls[0])
                label = res.names[cls_id].lower()
                if label in OBJECT_HEIGHTS:
                    found = True
                    break
            if found:
                break

        if not found:
            messagebox.showerror('Error', 'No known object detected in video.')
            return

        # Let user know what to select
        messagebox.showinfo('Select Object',
            f'A "{label}" was detected.\nPlease select this object in the frame.\nPress Enter/Space to confirm selection, c to cancel.')

        # Manual bounding box selection
        x, y, w, h = cv2.selectROI(f'Select the {label}', frame, False, False)
        cv2.destroyWindow(f'Select the {label}')
        if w == 0 or h == 0:
            return

        pix_h = h
        real_h = OBJECT_HEIGHTS.get(label, self.real_h)

        # Ask for distance
        real_d = simpledialog.askfloat('Distance', f'Enter estimated distance to the {label} (in meters):')
        if real_d is None:
            return

        self.focal = (pix_h * real_d) / real_h
        messagebox.showinfo('Calibration', f'Detected object: {label}\nFocal length: {self.focal:.1f}px')

        name = simpledialog.askstring('Name', 'Enter a name for this camera:')
        if name:
            saveFocal(self.focal, name)
            self.updateCameraList()





    def start(self):
        if not self.focal:
            messagebox.showerror('Error','Load or calibrate first')
            return
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.video_path.get())
        self.running = True
        self.processFrame()

    def stop(self):
        self.running = False

    def resume(self):
        if not self.running and self.cap:
            self.running = True
            self.processFrame()

    def processFrame(self):
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
            label = res.names[cls_id]
            if cls_id in VEHICLE_CLASSES or label.lower() == 'person':
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                h = y2 - y1
                if label in OBJECT_HEIGHTS:
                    real_h = OBJECT_HEIGHTS.get(label.lower(), self.real_h)
                    frame_h = frame.shape[0]
                    if label.lower() == 'bus':
                        real_h = 4.0 if self.use_double_decker.get() else 3.0
                    if label.lower() == 'truck':
                        # Heuristic: adjust height for large trucks
                        if h > 180:  # Adjust this threshold as needed
                            real_h = 2.8  # large truck
                        if h > 230:  # Adjust this threshold as needed
                                real_h = 3.2  # larger truck
                if label == 'person' and self.ignore_low_persons.get() and y2 > frame_h * (2/3):
                    continue
                dist = estimateDistance(h, self.focal, real_h)
                if dist < TOO_CLOSE_THRESHOLD:
                    too_close = True
                    color = (0, 0, 255)
                elif dist < CLOSE_THRESHOLD:
                    close = True
                    color = (0, 0, 150)
                else:
                    color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} {dist:.1f}m", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Overlays
        if too_close:
            overlay = frame.copy()
            overlay[:] = (0, 0, 255)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            cv2.putText(frame, 'WARNING: Object Too Close', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 3)

        # Display frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(img)
        self.img_label.imgtk = imgtk
        self.img_label.config(image=imgtk)
        delay = int(BASE_DELAY_MS / speed)
        self.after(delay, self.processFrame)

    def exit(self):
        if self.cap:
            self.cap.release()
        self.destroy()

if __name__ == '__main__':
    app = DistanceEstimatorApp()
    app.mainloop()
