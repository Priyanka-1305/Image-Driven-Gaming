import cv2

#reading image using OpenCV library
img=cv2.imread('image.jpg')

#displaying original image:
cv2.imshow("Image",img)

#reading the image in grayscale
img2=cv2.imread('image.jpg',0)
print(img.shape)
#converting original image to grayscale and then displaying it
gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow('Converted to grayscale',gray_img)
cv2.waitKey(0)

#tried blurring original img
blr=cv2.blur(img,(3,7))
cv2.imshow('',blr)

#resizing image after taking dimensions from user
height=int(input('Specify height of the frame:'))
width=int(input('Specify width of the frame:'))
img=cv2.resize(img,(width,height))
cv2.imshow("Image after resizing",img)

#writing grayscale image to a new file
cv2.imwrite('Grayscale_Image_Saved.png',gray_img)
print('Grayscale image successfully saved')

cv2.waitKey(0)
cv2.destroyAllWindows()

