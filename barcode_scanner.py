from __future__ import print_function
import pyzbar.pyzbar as pyzbar
from pyzbar.pyzbar import decode
import numpy as np
import cv2
from imutils.video import WebcamVideoStream
from imutils.video import FPS
import imutils


 
def decode(im) : 
  # Find barcodes and QR codes
  decodedObjects = pyzbar.decode(im, symbols=[64])
 
  # Print results
  for obj in decodedObjects:
    print('Type : ', obj.type)
    print('Data : ', obj.data,'\n')
     
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
      cv2.line(im, hull[j], hull[ (j+1) % n], (255,0,0), 3)
 
  # Display results 
  cv2.imshow("Results", im);
  #cv2.waitKey(0);
 
   
# Main 
if __name__ == '__main__':
  #cap = cv2.VideoCapture(0)

  # construct the argument parse and parse the arguments

  font = cv2.FONT_HERSHEY_SIMPLEX
  print("[INFO] sampling frames from webcam...")
  cap = WebcamVideoStream(src=0).start()
  fps = FPS().start()
  
  while True: 
    
    frame = cap.read()
    frame = imutils.resize(frame, width=800)
    im = frame
 
    decodedObjects = decode(im)
    
    #if args["display"] > 0:
    #print(len(decodedObjects))
    if len(decodedObjects) > 0:
      cv2.putText(im,str(decodedObjects[0][0]),(10,50), font, 0.5,(255,0,255),1,cv2.LINE_AA)
    
    display(im, decodedObjects)
    fps.update()
    key = cv2.waitKey(1)
    if key == 27:
      break

  fps.stop()

  print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
  print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

  cap.stop()
  cv2.destroyAllWindows()

