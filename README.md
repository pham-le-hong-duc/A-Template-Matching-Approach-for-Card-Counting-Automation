# Real-time Card Detection using Template Matching

An application for real-time detection and tracking of 52 playing cards from computer screen using **Template Matching** technique with OpenCV.

## Overview

This is a Computer Vision project application that:
- **Real-time detection** of 52 playing cards (4 suits x 13 ranks) from desktop screen
- **State tracking** of detected cards with intelligent memory
- **Intuitive interface** displaying a 4x13 grid of cards with distinct colors
- **Parallel processing** with ThreadPoolExecutor for optimized performance

## Key Features

### 1. Real-time Detection
- Continuous screen capture and card detection
- Uses **Template Matching** (cv2.matchTemplate) with threshold 0.85
- Parallel processing of 13 ranks for increased speed

### 2. Intelligent Memory
- **Currently detecting**: Displayed bright with thick border (live detection)
- **Previously detected**: Dimmer with thin border (in memory)
- **Not yet appeared**: Bright with original suit color
- Auto-reset when all 52 cards are detected

### 3. Auto Reset
- **"End" Template**: Place `End.png` file in `templates/` folder to trigger reset when detected
- **Auto-reset**: Automatically resets when all 52 cards are found

### 4. User Interface
- Mini window 650x350 pixels
- Grid of 4 rows x 13 columns displaying all 52 cards
- Distinct colors: Spade Black | Heart Red | Diamond Orange-Red | Club Green
- Controls: START / STOP / RESET / CLEAR MEMORY
- Status bar showing number of detected cards (x/52)

## Project Structure

```
.
├── main.py                      # Main file - Run application
├── requirements.txt             # Dependencies
├── templates/                   # Folder containing 52 template images
│   ├── Spade-2.png ... Spade-A.png    (13 spades)
│   ├── Heart-2.png ... Heart-A.png    (13 hearts)
│   ├── Diamond-2.png ... Diamond-A.png (13 diamonds)
│   ├── Club-2.png ... Club-A.png      (13 clubs)
│   └── End.png                  # Special template (optional)
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

## Installation and Usage

### Step 1: Install Python
- Requirement: **Python 3.7+**
- Download at: https://www.python.org/downloads/

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

Dependencies include:
- `opencv-python>=4.5.0` - Computer Vision
- `numpy>=1.21.0` - Array processing
- `Pillow>=8.0.0` - Image processing for Tkinter
- `mss>=6.1.0` - Fast screenshot

### Step 3: Run the application
```bash
python main.py
```

### Step 4: Usage
1. The UI window will automatically start detection
2. Open game/video with 52 playing cards on screen
3. Application will automatically detect and mark cards that appear
4. Controls:
   - **START**: Start detection
   - **STOP**: Pause
   - **RESET**: Reset everything to initial state
   - **CLEAR MEM**: Clear memory but keep current detection

## Configuration

### Change detection threshold
Open `main.py` file, find the line:
```python
threshold = 0.85  # Lower to 0.7-0.8 for poor detection, raise to 0.9 for many false positives
```

### Add "End" Template
1. Create image file `End.png` (logo, game end symbol, etc.)
2. Place in `templates/` folder
3. When this template appears on screen → auto reset

## How It Works

### 1. Template Matching Algorithm
```
For each frame from screen:
├── Capture screen (mss)
├── Convert to grayscale
├── Parallel 13 threads, each thread processes 1 rank:
│   ├── Scan 4 templates (4 suits) for that rank
│   ├── Use cv2.matchTemplate with TM_CCOEFF_NORMED
│   └── Return candidates with confidence > threshold
├── Merge results from 13 threads
├── Resolve conflicts (simple NMS)
├── Update memory
└── Update UI
```

### 2. Parallel Processing
- **13 ThreadPoolExecutor workers** - each worker processes 1 rank
- Each rank has 4 templates (4 suits)
- Detection speed increased ~10-13x compared to sequential

### 3. Memory Management
```python
detected_cards          # Set: Cards being detected in current frame
detected_cards_memory   # Set: All cards ever detected
```

### 4. UI Update Strategy
- **Bright (raised border)**: Cards not yet detected
- **Lighter dim (sunken, thick border)**: Cards currently being detected
- **Darker dim (sunken, thin border)**: Cards in memory but not detected in current frame

## Performance

- **FPS**: ~10-30 FPS (depending on machine configuration)
- **CPU Usage**: Average 30-50% (4 cores)
- **RAM**: ~200-300 MB
- **Accuracy**: 85-95% (depending on template quality and threshold)

### Performance Optimization
1. Reduce number of templates in `templates/` if not all 52 cards needed
2. Increase `threshold` to reduce number of false positives to process
3. Reduce screen capture resolution (modify in code)

## System Requirements

- **OS**: Windows 7/8/10/11, macOS, Linux
- **Python**: 3.7 or higher
- **RAM**: Minimum 4GB
- **CPU**: Multi-core recommended (to leverage threading)
- **Display**: Minimum resolution 1280x720

## Troubleshooting

### Error: "Template directory not found"
- Ensure `templates/` folder is at same level as `main.py`
- Check that all 52 template files exist (Suit-Rank.png)

### Inaccurate detection
- **Many false positives**: Increase `threshold` to 0.9
- **Missing many cards**: Lower `threshold` to 0.7-0.8
- **Wrong cards**: Check template image quality

### Low performance
- Close other heavy applications
- Reduce number of templates in `templates/` folder
- Check Task Manager for CPU/RAM usage

### Window not displaying
- Check for errors in console
- Try running with Administrator privileges
- Reinstall Tkinter: `pip install --upgrade tk`

## Technical Details

### Template Matching
- **Algorithm**: Normalized Cross-Correlation (`TM_CCOEFF_NORMED`)
- **Pros**: Simple, fast, no training required
- **Cons**: Sensitive to scale, rotation, lighting

### Threading Strategy
- **ThreadPoolExecutor**: Divide work by ranks
- **Lock**: `threading.Lock()` to synchronize `detected_cards` access
- **UI Thread**: Tkinter runs on main thread, detection on background

### Screen Capture
- **Library**: `mss` - Faster than PIL/Pillow ~10x
- **Format**: BGRA → BGR conversion for OpenCV

## References

- **OpenCV Template Matching**: https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html
- **Python Threading**: https://docs.python.org/3/library/concurrent.futures.html
- **MSS Documentation**: https://python-mss.readthedocs.io/

## Educational Value

This project demonstrates:
- **Computer Vision**: Template Matching, Image Processing
- **Parallel Computing**: ThreadPoolExecutor, Multi-threading
- **GUI Programming**: Tkinter, Event-driven programming
- **Software Engineering**: Clean code, Modular design

## License

MIT License - Free to use for educational purposes.

## Author

Computer Vision Project - Template Matching for Card Detection

---

**Note**: This application is designed for educational and research purposes. Not recommended for use in game cheating.
