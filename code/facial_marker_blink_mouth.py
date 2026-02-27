import cv2
import dlib
from scipy.spatial import distance as dist
import numpy as np

# Load the face detector and shape predictor from dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Define a function to calculate the Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(np.array(eye[1]), np.array(eye[5]))
    B = dist.euclidean(np.array(eye[2]), np.array(eye[4]))
    C = dist.euclidean(np.array(eye[0]), np.array(eye[3]))
    ear = (A + B) / (2.0 * C)
    return ear

# Define a function to calculate the Mouth Aspect Ratio (MAR)
def mouth_aspect_ratio(mouth):
    A = dist.euclidean(np.array(mouth[3]), np.array(mouth[9]))
    B = dist.euclidean(np.array(mouth[2]), np.array(mouth[10]))
    C = dist.euclidean(np.array(mouth[4]), np.array(mouth[8]))
    mar = (A + B + C) / (3.0 * A)
    return mar

# Initialize variables for blink detection
EAR_THRESHOLD = 0.2  # Adjust this threshold as needed
eye_closed = False

# Initialize variables for mouth open detection
MAR_THRESHOLD = 0.5  # Adjust this threshold as needed
mouth_open = False

while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        left_eye = [landmarks.part(i) for i in range(36, 42)]
        right_eye = [landmarks.part(i) for i in range(42, 48)]
        mouth = [landmarks.part(i) for i in range(48, 68)]

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        mar = mouth_aspect_ratio(mouth)

        # Check for eye blink
        if left_ear < EAR_THRESHOLD and right_ear < EAR_THRESHOLD:
            if not eye_closed:
                print("Eye Blink Detected")
                eye_closed = True
        else:
            eye_closed = False

        # Check for mouth open
        if mar > MAR_THRESHOLD:
            if not mouth_open:
                print("Mouth Open Detected")
                mouth_open = True
        else:
            mouth_open = False

    cv2.imshow("Facial Landmarks", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
