# Control music with gestures
A computer vision Python application that allows you to control audio (Play/Pause) using an Open Palm gesture (cool).
The script uses Google MediaPipe for real-time hand tracking and PyAutoGUI to simulate system media keys. 
The core logic detects an "Open Palm" gesture by analyzing finger angles and knuckle alignment to trigger the play/pause state.

## Features
* Real-time hand tracking: accurately detects 21 hand landmarks (through MediaPipe).
* Gesture recognition: Heuristic-based detection for "Open Palm" (fingers fully extended and upright).
* Audio control: Automatically toggles Play/Pause on your system when the gesture is detected.
* Visualization: Overlays joint connections and finger angles on the video feed.

## Prerequisites
* Python 3.10+
* A webcam (or input video file)

### Dependencies Install the required Python libraries:
```bash
pip install opencv-python mediapipe pyautogui numpy
```

## Usage
1) Clone the repository:
```bash
git clone https://github.com/bl4ze4447/computer-vision.git
cd computer-vision/control-music-with-gestures
```

2) Choose input source:
* Default: The python script looks for 'input.mp4' in the same directory.
* Webcam: You must replace the commented line below with the uncommented one:
```Python
# cap = cv2.VideoCapture("input.mp4")
cap = cv2.VideoCapture(0) 
```

3) Run and enjoy:
```Bash
python main.py
```
### Controls
* <b>Open Palm Gesture</b>: Pause audio.
* <b>Pressing 'q'</b>: Close script.
