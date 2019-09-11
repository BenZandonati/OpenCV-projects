from __future__ import print_function
import pyzbar.pyzbar as pyzbar
from pyzbar.pyzbar import decode
import numpy as np
import cv2
from imutils.video import WebcamVideoStream
from imutils.video import FPS
import imutils
import glob
import sys, time, math

calib_path  = "/Users/benzandonati/Desktop/code/opencv/CalibImages/"
mtx   = np.loadtxt(calib_path+'cameraMatrix.txt', delimiter=',')
dist   = np.loadtxt(calib_path+'cameraDistortion.txt', delimiter=',')

dimension = 70 #- mm

objp = np.zeros((2*2,3), np.float32)
objp[:,:2] = np.mgrid[0:2,0:2].T.reshape(-1,2)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, dimension, 0.001)

axis = np.float32([[1,0,0], [0,1,0], [0,0,1]]).reshape(-1,3)

def gstreamer_pipeline (capture_width=640, capture_height=360, display_width=1360, display_height=768, framerate=60, flip_method=0) :   
    return ('nvarguscamerasrc ! ' 
    'video/x-raw(memory:NVMM), '
    'width=(int)%d, height=(int)%d, '
    'format=(string)NV12, framerate=(fraction)%d/1 ! '
    'nvvidconv flip-method=%d ! '
    'video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! '
    'videoconvert ! '
    'video/x-raw, format=(string)BGR ! appsink'  % (capture_width,capture_height,framerate,flip_method,display_width,display_height))

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 3)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 3)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 3)
    return img
 
def decode(im) : 
  # Find barcodes and QR codes
  decodedObjects = pyzbar.decode(im)
 
  # Print results
  for obj in decodedObjects:
    print('Type : ', obj.type)
    print('Data : ', obj.data)
    print('Points : ', obj.polygon,'\n')
 
  return decodedObjects 

# Display barcode and QR code location  
def display(im, decodedObjects):
 
  # Loop over all decoded objects
  for decodedObject in decodedObjects: 
    points = decodedObject.polygon
 
    # If the points do not form a quad, find convex hull
    if len(points) > 4 : 
      hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
      hull = list(map(tuple, np.squeeze(hull)))
    else : 
      hull = points;
     
    # Number of points in the convex hull
    n = len(hull)
 
    # Draw the convext hull
    for j in range(0,n):
      cv2.line(im, hull[j], hull[ (j+1) % n], (0,0,0), 3)
 
  # Display results 

  #cv2.imshow("Results", im);
  #cv2.waitKey(0);
 
   
# Main 
if __name__ == '__main__':
  #cap = cv2.VideoCapture(0)

  # construct the argument parse and parse the arguments

  font = cv2.FONT_HERSHEY_SIMPLEX
  print("[INFO] sampling frames from webcam...")
  cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
  fps = FPS().start()
  
  while True: 
    
    frame = cap.read()
    #frame = imutils.resize(frame, width=800)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
    decodedObjects = decode(frame)
    
    #if args["display"] > 0:
    #print(len(decodedObjects))
    if len(decodedObjects) > 0:

      if str(decodedObjects[0][0]) == "b'51.645349/-0.632181'" :
        break      

      cv2.putText(frame,str(decodedObjects[0][0]),(10,50), font, 0.5,(0,0,0),1,cv2.LINE_AA)

      points = decodedObjects[0].polygon
      corners = np.array([point for point in points], dtype=np.float32)

      #print(corners)

      temp0 = corners[0]
      temp1 = corners[1]
      temp2 = corners[2]
      temp3 = corners[3]

      corners = np.array([temp0, temp1, temp3, temp2])

      #print(corners)

      corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

      ret,rvecs, tvecs = cv2.solvePnP(objp, corners2, mtx, dist)

      imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)

      print(tvecs*dimension)
      if len(corners2) > 3 :
        frame = draw(frame,corners2,imgpts)

      
    display(frame, decodedObjects)

    #im = imutils.resize(frame, width=300)
    #cv2.imshow("Results", frame);
    fps.update()
    key = cv2.waitKey(1)
    if key == 27:
      break


  fps.stop()

  print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
  print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

  cap.release()
  cv2.destroyAllWindows()

