import dlib
import cv2
from scipy.spatial import distance as dist
import numpy as np
import time

# Initialize the face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Function to calculate EAR
def eye_aspect_ratio(eye):
    # Extract (x, y) coordinates from the eye landmarks
    eye = [(point.x, point.y) for point in eye]

    # Convert the eye landmarks to NumPy array
    eye = np.array(eye, dtype=np.float32)

    # Calculate the Euclidean distances between the two sets of vertical eye landmarks
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # Calculate the Euclidean distance between the horizontal eye landmarks
    C = dist.euclidean(eye[0], eye[3])

    # Calculate the EAR
    ear = (A + B) / (2.0 * C)
    return ear

# Open a video stream from your webcam
cap = cv2.VideoCapture(0)

# Initialize variables to track the state of each eye
left_eye_open = True
right_eye_open = True

# Initialize variables to measure the duration each eye is closed
left_eye_closed_time = 0
right_eye_closed_time = 0

# Define a threshold for determining if the eye is closed
EAR_THRESHOLD = 0.2  # Adjust this threshold as needed

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to grayscale for face and landmark detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = detector(gray)

    for face in faces:
        shape = predictor(gray, face)

        # Extract landmarks for both the left and right eyes
        left_eye = [shape.part(i) for i in range(42, 48)]
        right_eye = [shape.part(i) for i in range(36, 42)]

        # Calculate the EAR for both eyes
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        # Check if the left eye is closed
        if left_ear < EAR_THRESHOLD:
            if left_eye_open:
                left_eye_closed_time = time.time()
            left_eye_open = False
        else:
            left_eye_open = True

        # Check if the right eye is closed
        if right_ear < EAR_THRESHOLD:
            if right_eye_open:
                right_eye_closed_time = time.time()
            right_eye_open = False
        else:
            right_eye_open = True

        # You can print the EAR values for both eyes for debugging
        print(f"Left Eye EAR: {left_ear:.2f}")
        print(f"Right Eye EAR: {right_ear:.2f}")

        # Calculate the duration each eye is closed
        left_eye_closed_duration = time.time() - left_eye_closed_time
        right_eye_closed_duration = time.time() - right_eye_closed_time

        print(f"Left Eye Closed for {left_eye_closed_duration:.2f} seconds")
        print(f"Right Eye Closed for {right_eye_closed_duration:.2f} seconds")

        # Draw the landmarks on the frame for visualization
        for point in left_eye + right_eye:
            x, y = point.x, point.y
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

    cv2.imshow('Eye Blink Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
