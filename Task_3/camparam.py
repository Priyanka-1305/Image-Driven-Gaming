import cv2
import mediapipe as mp
import numpy as np

# Load intrinsic and distortion parameters
mtx = np.array([
    [926.077, 0.0, 356.089],
    [0.0, 925.775, 355.249],
    [0.0, 0.0, 1.0]
], dtype=np.float32)

dist = np.array([-0.0139, 1.4192, -0.0035, -0.0035, -6.3943], dtype=np.float32)

# Validate camera matrices
def validate_camera_matrices(mtx, dist):
    """ Validate the intrinsic and distortion matrices """
    if mtx.shape != (3, 3):
        raise ValueError("Intrinsic matrix must be 3x3!")
    if dist.shape != (5,):
        raise ValueError("Distortion coefficients must have 5 values!")

    # Check if intrinsic matrix is upper triangular
    if not np.allclose(mtx, np.triu(mtx)):
        raise ValueError("Intrinsic matrix is not upper triangular!")

    print("Camera parameters validated successfully!")

# Run validation
try:
    validate_camera_matrices(mtx, dist)
except ValueError as e:
    print(e)
    exit(1)

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Open video files
cap1 = cv2.VideoCapture('cam0_test.mp4')
cap2 = cv2.VideoCapture('cam1_test.mp4')

# Process video frames one by one
while True:
    ret0, frame0 = cap1.read()
    ret1, frame1 = cap2.read()

    # Stop when video ends
    if not ret0 or not ret1:
        print("Video processing completed. Exiting...")
        break

    def process_landmarks(results, frame):
        if not results.multi_hand_landmarks:
            return

        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

    # Convert frames to RGB for MediaPipe
    rgb_frame0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
    rgb_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)

    # Detect hand keypoints
    results0 = hands.process(rgb_frame0)
    results1 = hands.process(rgb_frame1)

    # Process landmarks
    process_landmarks(results0, frame0)
    process_landmarks(results1, frame1)

    # Display frames
    cv2.imshow('Frame1', frame0)
    cv2.imshow('Frame2', frame1)

    # Press 'q' to exit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap1.release()
cap2.release()
cv2.destroyAllWindows()
