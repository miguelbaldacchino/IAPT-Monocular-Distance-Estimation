# Monocular Distance Estimation System for Cyclist Safety

**Author**: Miguel Baldacchino  
**Student ID**: 0171205L  
**Institution**: University of Malta  
**Module**: ARI2201 – Individual Assigned Practical Task

---

## 🚨 Problem

Cyclists are often endangered by vehicles overtaking too closely. Most regions mandate a minimum safe passing distance (typically 1.5m), but enforcement is minimal. This project implements a real-time monocular vision system to detect and flag proximity violations using consumer-grade video footage — no depth sensors, no stereo vision.

---

## 💡 Solution

A Python-based desktop application that estimates real-world distances from monocular video using:

- ✅ YOLOv8n for real-time object detection  
- ✅ Triangle similarity for distance calculation  
- ✅ Calibration via user-selected known distances  
- ✅ Tkinter-based GUI with real-time preview, export, and speed control  

---

## 🎯 Features

- **Real-time object detection** of `car`, `bus`, `truck`, `bicycle`, `motorcycle`, `pedestrian`
- **Per-frame distance estimation** using calibrated focal length
- **Warning overlays**:
  - Orange: < 1.5m
  - Red: < 1.0m
- **Custom camera calibration** from any video using known object size & distance
- **Speed-adjustable preview mode** (1x–15x)
- **Video export with annotations** (.mp4)
- **False-positive filtering** (e.g. cyclist hands, low-frame clutter)
- **Truck/bus size differentiation** using bounding-box heuristics

---

## 🎥 Watch the Demo

See the system in action: real-time object detection, distance overlays, and cyclist safety alerts based on monocular vision.

📺 [▶️ Watch the demo on YouTube](https://www.youtube.com/watch?v=K90Wvh3lkXg&t=64s)

The video shows:

- Distance estimation from live bike-mounted footage  
- Real-time bounding boxes + color-coded proximity warnings  
- Calibration process and object selection  
- Annotated video export results  

## 🧠 How It Works

### Object Detection
Powered by [YOLOv8n](https://github.com/ultralytics/ultralytics), pretrained on COCO.

### Distance Estimation
Applies the pinhole camera formula:

\[
D = \frac{H \cdot f}{h}
\]

- \( D \): Distance to object  
- \( H \): Known object height  
- \( h \): Pixel height in image  
- \( f \): Calibrated focal length  

---

## 🧱 Tech Stack

- Python 3.10+
- OpenCV (image processing, video export)
- Tkinter (GUI)
- Ultralytics YOLOv8n (object detection)
- Pillow (GUI rendering)
- NumPy + JSON (data and config)

---

## 🧪 Evaluation

- **Consistent multi-object distance estimates**
- **Reliable size-based vehicle differentiation**
- **Accurate detection under camera shake**
- **Handled edge cases** (occlusion, shadows, false positives)
- **Tested across 5+ realistic YouTube videos**

---
