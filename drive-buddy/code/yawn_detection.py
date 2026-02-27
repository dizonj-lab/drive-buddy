import cv2
import dlib
from scipy.spatial import distance as dist
import time

# Load the face detector and shape predictor from dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Define a function to calculate the mouth aspect ratio (MAR)
def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[3], mouth[9])
    B = dist.euclidean(mouth[2], mouth[10])
    C = dist.euclidean(mouth[4], mouth[8])
    mar = (A + B + C) / (3.0 * A)
    return mar

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Initialize variables for yawn detection
yawn_count = 0
YAWN_THRESHOLD = 0.5  # Adjust this threshold as needed
yawn_start_time = None
YAWN_DURATION_THRESHOLD = 1.5  # Adjust this threshold for yawn duration (in seconds)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        mouth = []
        for i in range(48, 68):
            x, y = landmarks.part(i).x, landmarks.part(i).y
            mouth.append((x, y))

        mar = mouth_aspect_ratio(mouth)

        if mar < YAWN_THRESHOLD:
            if yawn_start_time is None:
                yawn_start_time = time.time()
            elif time.time() - yawn_start_time >= YAWN_DURATION_THRESHOLD:
                yawn_count += 1
                yawn_start_time = None
                print("Yawn detected - Yawn Count: {}".format(yawn_count))
        else:
            yawn_start_time = None

    cv2.imshow("Yawn Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
