import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # Read frame
    success, frame = cap.read()
    if not success:
        continue
        
    # YOUR CODE HERE
    # 1. Convert BGR to RGB
    rgbimage=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    cv2.imshow('Frame converted to RBG',rgbimage)
    
    # 2. Process the frame for hand detection
    res=hands.process(rgbimage)

    # 3. Draw landmarks and index fingertip circle
    if res.multi_hand_landmarks:
        for point in res.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame,point,mp_hands.HAND_CONNECTIONS)

        #index fingertip circle:
        index=point.landmark[8]
        h,w,c=frame.shape
        cx,cy=int(index.x*w),int(index.y*h)
        cv2.circle(frame,(cx,cy),radius=10,color=(0,0,255),thickness=2)
        cv2.imshow('Landmarks identified',frame)

    # 4. Add pinch detection (for bonus)

        thumb=point.landmark[4]
        c1x,c1y=int(thumb.x*w),int(thumb.y*h)
        cv2.circle(frame,(c1x,c1y),radius=10,color=(0,0,0),thickness=2)
      
        if np.sqrt((cx-c1x)**2+(cy-c1y)**2)<40:
            print('Pinching')
        else:
            print('Not pinching')

    # 5. Display the frame
    cv2.imshow('Video frame',frame)
    
    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

