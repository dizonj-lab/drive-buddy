import dlib
import cv2
from scipy.spatial import distance as dist
import numpy as np
import time
import matplotlib.pyplot as plt
from collections import deque

# Initialize the face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Function to calculate EAR (Eye Aspect Ratio)
def eye_aspect_ratio(eye):
    eye = [(point.x, point.y) for point in eye]
    eye = np.array(eye, dtype=np.float32)
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Function to calculate MAR (Mouth Aspect Ratio)
def mouth_aspect_ratio(mouth):
    mouth = [(point.x, point.y) for point in mouth]
    mouth = np.array(mouth, dtype=np.float32)
    A = dist.euclidean(mouth[12], mouth[16])
    B = dist.euclidean(mouth[14], mouth[18])
    C = dist.euclidean(mouth[2], mouth[10])
    mar = (A + B) / (2.0 * C)
    return mar

# Open a video stream from your webcam
cap = cv2.VideoCapture(0)

# Initialize variables for eye state tracking
left_eye_open = True
right_eye_open = True

# Initialize variables for measuring eye closed duration
left_eye_closed_time = 0
right_eye_closed_time = 0

# Initialize variables for mouth state tracking (yawning detection)
mouth_open = True
yawn_start_time = 0
YAWN_THRESHOLD = 0.5  # Adjust this threshold as needed

# Threshold for determining if the eye is closed (adjust as needed)
EAR_THRESHOLD = 0.2

# Initialize variables to collect blink and yawn data
blink_times = []
blink_durations = []
yawn_times = []

# Initialize the rolling window for blink durations (3 minutes)
window_size = 180  # 3 minutes * 60 seconds
rolling_window = deque([0] * window_size, maxlen=window_size)

# Create a figure for the time chart
plt.figure(figsize=(8, 4))
plt.title('Rolling Blink Durations (3 minutes)')
plt.xlabel('Time (seconds)')
plt.ylabel('Blink Duration (seconds)')
line, = plt.plot(list(range(window_size)), rolling_window)
plt.ylim(0, 10)  # Set the y-axis limit to a maximum of 10 seconds
plt.show(block=False)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        shape = predictor(gray, face)

        left_eye = [shape.part(i) for i in range(42, 48)]
        right_eye = [shape.part(i) for i in range(36, 42)]
        mouth = [shape.part(i) for i in range(48, 68)]

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        mar = mouth_aspect_ratio(mouth)

        if left_ear < EAR_THRESHOLD:
            if left_eye_open:
                left_eye_closed_time = time.time()
            left_eye_open = False
        else:
            left_eye_open = True

        if right_ear < EAR_THRESHOLD:
            if right_eye_open:
                right_eye_closed_time = time.time()
            right_eye_open = False
        else:
            right_eye_open = True

        left_eye_closed_duration = time.time() - left_eye_closed_time
        right_eye_closed_duration = time.time() - right_eye_closed_time

        if not left_eye_open and not right_eye_open:
            blink_times.append(time.time())
            blink_durations.append(left_eye_closed_duration + right_eye_closed_duration)
            rolling_window.append(left_eye_closed_duration + right_eye_closed_duration)
            line.set_ydata(rolling_window)
            plt.draw()

        if mar < YAWN_THRESHOLD:
            if mouth_open:
                yawn_start_time = time.time()
            mouth_open = False
        else:
            mouth_open = True

        if not mouth_open:
            yawn_duration = time.time() - yawn_start_time
            if yawn_duration > 1.0:  # Ignore short false positives
                yawn_times.append(time.time())
                print("Yawn detected")
                # Draw a dot marker on the mouth
                mouth_center = (int((mouth[48].x + mouth[54].x) / 2), int((mouth[48].y + mouth[54].y) / 2))
                cv2.circle(frame, mouth_center, 2, (0, 0, 255), -1)

        # Draw dots around the eyes
        for point in left_eye:
            x, y = point.x, point.y
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        for point in right_eye:
            x, y = point.x, point.y
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    cv2.imshow('Video Feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
