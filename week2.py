import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.2, min_tracking_confidence=0.5, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)
totfrm=0
detfrm=0
start=time.time()

while cap.isOpened():
    # Read frame
    success, frame = cap.read()
    totfrm+=1
    if not success:
        continue

    # Convert BGR to RGB
    rgbimage=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    
    # Process the frame for hand detection
    res=hands.process(rgbimage)

    # Draw landmarks and print all 21 hand keypoints
    if res.multi_hand_landmarks:
        detfrm+=1
        for point in res.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame,point,mp_hands.HAND_CONNECTIONS)

        h,w,c=frame.shape
        keypts=[]
        for id,lndmrk in enumerate(point.landmark):
            cx, cy = int(lndmrk.x*w),int(lndmrk.y*h)
            keypts.append((id, cx, cy))
            cv2.circle(frame,(cx,cy),5,(0,0,0),-1)
            cv2.putText(frame,str(id),(cx+10,cy-10),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,0),1)

        #print the hand keypoints for every frame
        print("Hand Keypoints:",keypts)
        

    # Display the frame
    cv2.imshow('Video frame',frame)
    
    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

tm=time.time()-start
print(tm,detfrm,detfrm*100/totfrm)
cap.release()
cv2.destroyAllWindows()

