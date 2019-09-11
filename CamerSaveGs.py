import numpy as np
import cv2
import time

def gstreamer_pipeline (capture_width=640, capture_height=360, display_width=1360, display_height=768, framerate=60, flip_method=0) :   
    return ('nvarguscamerasrc ! ' 
    'video/x-raw(memory:NVMM), '
    'width=(int)%d, height=(int)%d, '
    'format=(string)NV12, framerate=(fraction)%d/1 ! '
    'nvvidconv flip-method=%d ! '
    'video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! '
    'videoconvert ! '
    'video/x-raw, format=(string)BGR ! appsink'  % (capture_width,capture_height,framerate,flip_method,display_width,display_height))

cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
num = 0

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    cv2.imshow('frame',frame)

    c = cv2.waitKey(1)
    if c & 0xFF == ord('q'):
        break
    if c & 0xFF == ord('s'):
        cv2.imwrite('/Users/benzandonati/Desktop/code/opencv/CalibImages/image{0}.png'.format(num), frame)
        num += 1
        print(num)
        time.sleep(1)

         


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()