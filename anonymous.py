import cv2
import numpy as np

# initialize camera, classifier and load the new image
cap=cv2.VideoCapture(0)
img=cv2.imread("D:\\Downloads\\anonymous_face_mask.jpg")
classifier= cv2.CascadeClassifier('D:\\Downloads\\Computer-Vision-Tutorial-master\\Computer-Vision-Tutorial-master\\Haarcascades\\haarcascade_frontalface_default.xml')

# create masks to be used later
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,original_mask=cv2.threshold(img_gray,100,255,cv2.THRESH_BINARY_INV)
original_mask_inv=cv2.bitwise_not(original_mask)

while(cap.isOpened()):
    # get image and find the face(s)
    _,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    face = classifier.detectMultiScale(gray,1.8,3)
    
    for x,y,w,h in face:
        # resize images and mask to size of the face
        newFace = cv2.resize(img,(w,h),cv2.INTER_AREA)
        mask = cv2.resize(original_mask,(w,h),cv2.INTER_AREA)
        mask_inv = cv2.resize(original_mask_inv,(w,h),cv2.INTER_AREA)

        # obtain the foreground of the image and the background of the camera frame
        roi=frame[y:y+h,x:x+w]
        frame_bg=cv2.bitwise_and(roi,roi,mask=mask)
        img_fg=cv2.bitwise_and(newFace,newFace,mask=mask_inv)

        # replace the face with the image data and draw a rectangle
        frame[y:y+h,x:x+w]= frame_bg + img_fg
        #cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
    
    # show image and wait parse key    
    cv2.imshow("framee",frame)
    k = cv2.waitKey(1) & 0xff
    if k == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()