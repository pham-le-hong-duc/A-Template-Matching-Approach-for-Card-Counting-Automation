# Real-time Card Detection using Template Matching

## Demo
[Watch on YouTube](https://youtu.be/qIaZIEO_7W8?si=M3EYNI9F8Auyigml)

## Overview
A real-time computer vision application that detects and tracks 52 playing cards from screen capture using OpenCV template matching.

## Key Features
- Real-time detection using cv2.matchTemplate
- Parallel processing with ThreadPoolExecutor
- State tracking with memory mechanism
- Interactive Tkinter UI

## Tech Stack
- Python
- OpenCV
- NumPy
- MSS
- Tkinter

## Architecture
Screen Capture → Grayscale → Parallel Template Matching → NMS → State Memory → UI Update

## How to Run
```bash
pip install -r requirements.txt
python main.py
```
