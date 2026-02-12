import cv2
from ultralytics import YOLO
import numpy as np

model = YOLO('yolov8n-pose.pt')             #model used might change
cap = cv2.VideoCapture("IMG_6356.MOV")                   #access webcam. 0 tries for the webcam
current = False


def angle(a, b, c) :                         #checks the angle of the legs and returns it
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc))
    return np.degrees(np.arccos(cos_angle))

while True:
    ret, frame = cap.read()                 #read the camera

    if not ret:                             #if the camera is not accessed try again
        cap = cv2.VideoCapture(0)           
        continue

    #frame = cv2.resize(frame, (640,720))   #resizing code, uncomment to use

    width, height = frame.shape[:2]         #copy width and height of the camera and create a blank space to place the keypoints on
    blank = np.zeros((width, height, 3), dtype=np.uint8)

    res = model(frame)                      #run the current frame through the model

    for keypoints in res[0].keypoints.data: #plot each keypoint 
        keypoints = keypoints.cpu().numpy()

        for i, keypoint in enumerate(keypoints):
            x,y,confidence = keypoint
            if confidence > 0.7:
                cv2.circle(blank, (int(x), int(y)), radius=5, 
                           color=(255,0,0), thickness=1) #keypoint marked with circle
                
                cv2.putText(blank, f'{i}', (int(x), int(y)-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1,cv2.LINE_AA) #keypoints named with a number


    connections = [             #lines between the numbered keypoints
        (3,1), (1,0), (0,2), 
        (2,4), (1,2), (4,6), 
        (3,5), (5,6), (5,7),
        (7,9), (6,8), (8,10), 
        (11,12),(11,13),(13,15),
        (12,14), (14,16), (5,11),
        (6,12)
    ]

    for part_a, part_b in connections:  #ploting the connections
        x1, y1, conf1 = keypoints[part_a]
        x2, y2, conf2 = keypoints[part_b]

        if conf1 > 0.5 and conf2 > 0.5: #drawing th lines onto the blank page
            cv2.line(blank, (int(x1), int(y1)), (int(x2),int(y2)), (255,0,255),thickness=2)
    
    hip = keypoints[11]     #keypoints for legs
    knee = keypoints[13]
    ankle = keypoints[15]

    knee_angle = angle(hip, knee, ankle)    #check angle of the legs

    
    if knee_angle < 140:                    #should the angle be less than 140 they are sitting
        standing = "Sitting"
    else:
        standing = "Standing"


    cv2.putText(blank, standing , (int(10), int(10)), 
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1,cv2.LINE_AA) #keypoints named with a number
    

    cv2.imshow('frame',frame)           #display the camera and the keypoints
    cv2.imshow('Skeleton', blank)
    if cv2.waitKey(1) & 0xFF == ord('q'):   #'q' pressed to end the loop
        break

cap.release()
cv2.destroyAllWindows()