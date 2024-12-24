import cv2

# initializing video capture from webcam using OpenCV
video=cv2.VideoCapture(0)
width=video.get(3)
height=video.get(4)
saved_vid=cv2.VideoWriter('Saved_video.mp4',cv2.VideoWriter_fourcc(*'mp4v'),10,(int(width),int(height)))
while True:
    isTrue,frame=video.read()
    #displaying live video stream from webcam.
    cv2.imshow('Video',frame)

    #converting each frame of video to grayscale and displaying it
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    cv2.imshow('Grayscale video',gray)

    #experimebting brightness and comtrast
    adj=cv2.convertScaleAbs(frame,alpha=1.7,beta=0.3)
    cv2.imshow('Brightness adjusted',adj)

    #saving a snippet of the processed video aftetr adjusting contrast and brightness to a file
    saved_vid.write(frame)
    
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

video.release()
saved_vid.release()
cv2.destroyAllWindows()
