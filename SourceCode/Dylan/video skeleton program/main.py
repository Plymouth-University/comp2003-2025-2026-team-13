import cv2
from ultralytics import YOLO
import numpy as np

cap = cv2.VideoCapture('IMG_6057.mp4')  #read the video
model = YOLO('yolov8n-pose.pt')         #YOLO model for reading footage


while True:
    ret, frame = cap.read()             #reads the video, ret is bool for if the video is read

    if not ret:                         #if the video is not read try again
        cap = cv2.VideoCapture('IMG_6057.mp4')  
        continue                        #this will loop the video forever, remove if needed

    frame = cv2.resize(frame, (640,720)) #size of video
    width, height = frame.shape[:2]
    blank = np.zeros((width, height, 3), dtype=np.uint8) #create blank image to make skeleton easier to read

    res = model(frame)                  #skeleton of the current frame
    #frame = res[0].plot()               #plot the skeleton on the original video

    for keypoints in res[0].keypoints.data: #get the keypoints of the 
        keypoints = keypoints.cpu().numpy()

        for i, keypoint in enumerate(keypoints):    #for each keypoint
            x,y,confidence = keypoint
            if confidence > 0.7:                    #plot the keypoints
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

    for part_a, part_b in connections:  #ploting the connections``
        x1, y1, conf1 = keypoints[part_a]
        x2, y2, conf2 = keypoints[part_b]

        if conf1 > 0.5 and conf2 > 0.5: #drawing th lines onto the blank page
            cv2.line(blank, (int(x1), int(y1)), (int(x2),int(y2)), (255,0,255),thickness=2)


    cv2.imshow('frame', frame)  #show video
    cv2.imshow('Skeleton', blank)
    if cv2.waitKey(1) & 0xFF == ord('q'):   #'q' pressed to end the loop
        break

cap.release()                           #release system resources
cv2.destroyAllWindows()                 #close the video