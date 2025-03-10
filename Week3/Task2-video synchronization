import cv2
import mediapipe as mp
import time
import threading
from collections import deque

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Open video streams (0 and 1 for two cameras)
cap1 = cv2.VideoCapture(0)  # First camera (Laptop Webcam)
cap2 = cv2.VideoCapture(1)  # Second camera (Phone via Iriun)

# Buffers to store frames and timestamps
buffer1 = deque()
buffer2 = deque()
threshold = 30  # Maximum timestamp difference in milliseconds
lock = threading.Lock()

def capture_frames(cap, buffer):
    """Continuously captures frames with timestamps and stores them in a buffer."""
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
        if timestamp == 0:  # Fallback if timestamp isn't provided
            timestamp = time.time() * 1000  # Convert seconds to milliseconds
        with lock:
            buffer.append((timestamp, frame))

# Start capture threads
thread1 = threading.Thread(target=capture_frames, args=(cap1, buffer1))
thread2 = threading.Thread(target=capture_frames, args=(cap2, buffer2))
thread1.start()
thread2.start()

def synchronize_and_display():
    """Synchronizes frames from both streams and displays them side by side."""
    while True:
        with lock:
            if not buffer1 or not buffer2:
                continue  # Wait for frames
            
            t1, f1 = buffer1[0]
            t2, f2 = buffer2[0]
            
            if abs(t1 - t2) <= threshold:
                buffer1.popleft()
                buffer2.popleft()
                
                # Convert frames to RGB for MediaPipe
                f1_rgb = cv2.cvtColor(f1, cv2.COLOR_BGR2RGB)
                f2_rgb = cv2.cvtColor(f2, cv2.COLOR_BGR2RGB)
                
                # Process with MediaPipe Hands
                results1 = hands.process(f1_rgb)
                results2 = hands.process(f2_rgb)
                
                # Draw hand landmarks on first frame
                if results1.multi_hand_landmarks:
                    for hand_landmarks in results1.multi_hand_landmarks:
                        mp_draw.draw_landmarks(f1, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Draw hand landmarks on second frame
                if results2.multi_hand_landmarks:
                    for hand_landmarks in results2.multi_hand_landmarks:
                        mp_draw.draw_landmarks(f2, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Concatenate and display both frames side by side
                combined_frame = cv2.hconcat([f1, f2])
                cv2.imshow('Synchronized Hand Tracking', combined_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            elif t1 < t2:
                buffer1.popleft()
            else:
                buffer2.popleft()

# Start synchronization and display loop
synchronize_and_display()

# Release resources
cap1.release()
cap2.release()
cv2.destroyAllWindows()
