import numpy as np
import cv2 as cv
cap1 = cv.VideoCapture(0)
cap2 = cv.VideoCapture(1)
while(True):
    # Capture frame-by-frame
    ret1, frame1 = cap1.read()
    # Our operations on the frame come here
    gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    # Display the resulting frame
    cv.imshow('frame1',gray1)
    # Capture frame-by-frame
    ret2, frame2 = cap2.read()
    # Our operations on the frame come here
    gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    # Display the resulting frame
    cv.imshow('frame2',gray2)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
# When everything done, release the capture
cap1.release()
cap2.release()
cv.destroyAllWindows()