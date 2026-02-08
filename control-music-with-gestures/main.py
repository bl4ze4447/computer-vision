import time
import math
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyautogui

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

def process_detection_results(pixel_landmarks, frame):
    # Check if the user is making an open palm gesture
    #
    # For each finger, make sure the 'knuckles' are in order on the Y axis
    # and the angle between the start and end of a finger is in a specific
    # range based on the finger
    global is_music_playing
    if not pixel_landmarks:
        return
    
    is_open_palm = True
    for (finger_idx, finger_keypoints) in enumerate(FINGERS_KEYPOINTS):
        # Ignore the thumb
        if finger_idx == 0:
            continue 
        if not is_open_palm:
            break

        last_kp = finger_keypoints[len(finger_keypoints) - 1]
        first_kp = finger_keypoints[0]
        
        dy = pixel_landmarks[first_kp][1] - pixel_landmarks[last_kp][1]
        dx = pixel_landmarks[first_kp][0] - pixel_landmarks[last_kp][0]
        angle = abs(math.ceil(math.atan2(dy, dx) * 180 / math.pi))

        frame = cv2.putText(frame, f"Finger: {finger_idx} | Angle: {angle}", (0, (40)*finger_idx), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)

        # Check if the angle is ok for the finger
        max_angle = 130
        min_angle = 60
        match finger_idx:
            # Index finger
            case 1:
                max_angle = 120
                min_angle = 50
            case 2:
                max_angle = 120
                min_angle = 60
            case 3:
                max_angle = 110
                min_angle = 70
            case 4:
                max_angle = 120
                min_angle = 60
                
        if angle > max_angle or angle < min_angle:
            is_open_palm = False
        
        for i in range(len(finger_keypoints) - 1):
            if pixel_landmarks[finger_keypoints[i]][1] < pixel_landmarks[finger_keypoints[i+1]][1]:
                is_open_palm = False
                break

    if is_music_playing and is_open_palm:
        is_music_playing = False
        pyautogui.press('playpause')
    elif not is_music_playing and not is_open_palm:
        is_music_playing = True
        pyautogui.press('playpause')

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
cap = cv2.VideoCapture("input.mp4")
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
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
landmarker.close()
