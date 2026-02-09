import time
import math
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyautogui

pyautogui.PAUSE = 0

# Variables that help me do what I currently want to do :O
is_music_playing = True

# https://ai.google.dev/static/mediapipe/images/solutions/hand-landmarks.png
WRIST = 0
THUMB_1 = 1
THUMB_2 = 2
THUMB_3 = 3
THUMB_4 = 4
INDEX_1 = 5
INDEX_2 = 6
INDEX_3 = 7
INDEX_4 = 8
MIDDLE_1 = 9
MIDDLE_2 = 10
MIDDLE_3 = 11
MIDDLE_4 = 12
RING_1 = 13
RING_2 = 14
RING_3 = 15
RING_4 = 16
PINKY_1 = 17
PINKY_2 = 18
PINKY_3 = 19
PINKY_4 = 20
FINGERS_KEYPOINTS = [
    [THUMB_1, THUMB_2, THUMB_3, THUMB_4],
    [INDEX_1, INDEX_2, INDEX_3, INDEX_4],
    [MIDDLE_1, MIDDLE_2, MIDDLE_3, MIDDLE_4],
    [RING_1, RING_2, RING_3, RING_4],
    [PINKY_1, PINKY_2, PINKY_3, PINKY_4]
]
HAND_CONNECTIONS = [
    (WRIST, THUMB_1), (THUMB_1, THUMB_2), (THUMB_2, THUMB_3), (THUMB_3, THUMB_4),
    (WRIST, INDEX_1), (INDEX_1, INDEX_2), (INDEX_2, INDEX_3), (INDEX_3, INDEX_4),
    (INDEX_1, MIDDLE_1), (MIDDLE_1, MIDDLE_2), (MIDDLE_2, MIDDLE_3), (MIDDLE_3, MIDDLE_4),     
    (MIDDLE_1, RING_1), (RING_1, RING_2), (RING_2, RING_3), (RING_3, RING_4),  
    (RING_1, PINKY_1), (WRIST, PINKY_1), (PINKY_1, PINKY_2), (PINKY_2, PINKY_3), (PINKY_3, PINKY_4)
]

def draw_landmarks(image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    
    if not hand_landmarks_list:
        return image, None

    annotated_image = np.copy(image)
    height, width, _ = annotated_image.shape

    pixel_landmarks = []
    for hand_landmarks in hand_landmarks_list:        
        # Convert normalized coordinates (0-1) to pixel coordinates
        for landmark in hand_landmarks:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            pixel_landmarks.append((x, y))

        # Draw joints
        for connection in HAND_CONNECTIONS:
            start_idx = connection[0]
            end_idx = connection[1]
            
            start_point = pixel_landmarks[start_idx]
            end_point = pixel_landmarks[end_idx]
            
            cv2.line(annotated_image, start_point, end_point, (0, 255, 0), 2)

        # Draw keypoints
        for point in pixel_landmarks:
            cv2.circle(annotated_image, point, 4, (255, 0, 0), -1)

    return annotated_image, pixel_landmarks

def distance(point1, point2):
    return math.sqrt((point1[1] - point2[1])**2 + (point1[0] - point2[0])**2)

def middle_point(point1, point2):
    return ((point1[0] + point2[0]) // 2, (point1[1] + point2[1]) // 2)

def in_interval(interval, value):
    return interval[0] <= value and value <= interval[1]

cursor_last_pos = None
cursor_origin = None
gesture_origin = None
cursor_move_amount = None
awaiting_click = False
screen_size = pyautogui.size()
def process_detection_results(pixel_landmarks, frame):
    global cursor_last_pos, cursor_origin, gesture_origin, screen_size, awaiting_click        
    if not pixel_landmarks:
        return
    
    # Check if we are holding the cursor
    # (by having the tip of the thumb and index finger touching)
    thumb_point = pixel_landmarks[THUMB_4]
    index_point = pixel_landmarks[INDEX_4]
    fingers_distance = distance(thumb_point, index_point)

    # Margin of Error for "cursor move" gesture
    gesture_margin_error = 25

    # We are holding the cursor
    #
    # what do I need to do?
    # first of all, I need an 'origin' point to start from. Why?
    # because I do not want the cursor to move to a random position
    # everytime I start the cursor hold
    #
    # okay, so I start from that origin. how do I move?
    # well, I also need an origin for the start of the gesture.
    # Why? so that I keep track of how much I moved (10px on X axis, 25px on Y axis)
    #
    # Until now I need:
    # cursor_last_position -> Everytime we start the gesture, this gets set to the current position of the cursor
    # gesture_origin -> Tracks the position of the gesture inside of the webcam
    # > We calculate the position of the cursor based on the distance traveled on the xOy axis
    # > with the gesture_origin as the origin
    if fingers_distance <= gesture_margin_error:
        awaiting_click = True
        if gesture_origin and cursor_origin:
            current_gesture_pos = middle_point(thumb_point, index_point)
            dx = gesture_origin[0] - current_gesture_pos[0]
            dy = gesture_origin[1] - current_gesture_pos[1]
            multiplier_x = 4
            multiplier_y = 4
            cursor_x = cursor_origin[0] + (dx * multiplier_x) 
            cursor_y = cursor_origin[1] + -(dy * multiplier_y) # invert the Y axis

            # Bounds checks 
            cursor_x = min(max(cursor_x, 0), screen_size[0])
            cursor_y = min(max(cursor_y, 0), screen_size[1])
            smoothing_factor = 0.2
            
            if cursor_last_pos:
                cursor_x = cursor_last_pos[0] + (cursor_x - cursor_last_pos[0]) * smoothing_factor
                cursor_y = cursor_last_pos[1] + (cursor_y - cursor_last_pos[1]) * smoothing_factor

            
            pyautogui.moveTo(cursor_x, cursor_y, _pause=False)
            cursor_last_pos = (cursor_x, cursor_y)
            
            return

        if not gesture_origin:
            # Middle point between the fingers
            gesture_origin = middle_point(thumb_point, index_point)
        if not cursor_origin:
            cursor_origin = pyautogui.position()

        return
    
    if awaiting_click:
        pyautogui.click()
        awaiting_click = False
        
    cursor_last_pos = None
    cursor_origin = None
    gesture_origin = None
    cursor_move_amount = None

detection_result = None
def update_detection_result(result, *args):
    global detection_result
    detection_result = result
    
# Setup mediapipe hand detection
model_path = "hand_landmarker.task"
options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
    running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
    num_hands=1,
    result_callback=update_detection_result)

landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[Error] Cannot open video capture.")
    exit()

while True:
    success, frame = cap.read()
    if not success:
        print("[Info] Video ended.")
        break

    # OpenCV default color scheme is BGR
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Mediapipe accepts RGB
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    landmarker.detect_async(mp_image, frame_timestamp_ms)

    if detection_result:
        frame, pixel_landmarks = draw_landmarks(frame, detection_result)
        process_detection_results(pixel_landmarks, frame)

    # Convert back to OpenCV default color scheme
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame = cv2.flip(frame, 1)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
landmarker.close()
