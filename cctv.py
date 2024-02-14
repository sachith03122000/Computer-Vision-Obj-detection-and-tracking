import cv2
import numpy as np
from tracker import *



# Create tracker object
tracker = EuclideanDistTracker()

cap = cv2.VideoCapture("cctv.mp4")

# Object detection from Stable camera
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows = True)


while True:

    ret, frame = cap.read()
    height, width, _ = frame.shape
    #print(height, width)

    # Extract Region of interest 
    roi = frame[0: 180,960: 1100]

    # 1. Object Detection
    mask = fgbg.apply(roi) #object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    

    
            #accuracy = 0.001 * cv2.arcLength(cnt , True) 
    
 
    
    
    detections = [] 
    for cnt in contours:
        # Calculate area and remove small elements
        area = cv2.contourArea(cnt)
        if area > 100:
            #cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(cnt)
            detections.append([x, y, w, h])
            


    


    # 2. Object Tracking
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
         x, y, w, h, id = box_id
         cv2.putText(roi, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
         cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 1)




    cv2.imshow("roi", roi)
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()