# Control cursor with gestures
A computer vision Python application that allows you to control cursor movement and click using a pinching gesture (cool).
The script uses Google MediaPipe for real-time hand tracking and PyAutoGUI to simulate cursor movement and click. 
The core logic detects an "Open Palm" gesture by analyzing finger angles and knuckle alignment to trigger the play/pause state.

## Features
* Real-time hand tracking: accurately detects 21 hand landmarks (through MediaPipe).
* Gesture recognition: Heuristic-based detection for "Pinching".
* Cursor control: Moves cursor while pinching and clicks when the gesture is released.
* Visualization: Overlays joint connections..

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
* Default: The python script uses the webcam by default.
* Video input: You must replace the commented line below with the uncommented one:
```Python
# cap = cv2.VideoCapture(0) 
cap = cv2.VideoCapture("input.mp4")
```

3) Run and enjoy:
```Bash
python main.py
```
### Controls
* <b>Pinch and move hand</b>: Move cursor.
* <b>Release pinch</b>: Click.
* <b>Pressing 'q'</b>: Close script.
