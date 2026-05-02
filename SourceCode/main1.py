import cv2
from ultralytics import YOLO
import numpy as np
import time
import csv
from datetime import datetime
model = YOLO('yolov8n-pose.pt')             #model used might change
cap = cv2.VideoCapture(0)                   #access webcam. 0 tries for the webcam
current_activity = None
fall_count = 0
fall_detect = False
velocity = 0
prev_hip_y = None
prev_time = 0
fall_frames = 0

def process_frame(image):

    def angle(a, b, c) :                         #checks the angle of the legs and returns it
        ba = a - b
        bc = c - b
        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc))
        return np.degrees(np.arccos(cos_angle))

    global prev_hip_y, prev_time, fall_frames, current_activity, velocity
    frame = image
    status = "Unknown"
    res = model(frame)

    for keypoints in res[0].keypoints.data:
        keypoints = keypoints.cpu().numpy()

        #SITTING STANDING DETECTION
        hip = keypoints[11]
        knee = keypoints[13]
        ankle = keypoints[15]

        knee_angle = angle(hip, knee, ankle)
        print("knee angle: " , knee_angle)
        if knee_angle < 140:
            status = "Sitting"
        else:
            status = "Standing"


        #FALL DETECTION
        #body angle
        shoulder_mid = (keypoints[5][:2] + keypoints[6][:2])/2  #find the middle keypoint between shoulders
        hip_mid = (keypoints[11][:2] + keypoints[12][:2])/2     #find the middle keypoint between hips

        dx = hip_mid[0] - shoulder_mid[0]                       #x and y for both
        dy = hip_mid[1] - shoulder_mid[1]

        body_angle = np.degrees(np.arctan2(dy,dx))              #find the angle of this middle line created from dy and dx
        horizontal = abs(body_angle) < 30  or abs(body_angle) > 150                      #body angle q
        #print("Angle: " , body_angle)
        #print("Horizontal: " , horizontal)

        if horizontal == True:
            status = "Laying"

        #speed
        max_dt = 0.5
        current_time = time.time()                              #get the current time of the frame
        if prev_hip_y != None:
            dy = hip_mid[1] - prev_hip_y                        #find hip midpoint minus the last detected midpoint
            dt = current_time - prev_time                       #find time between frames
            
            #if dt > 0 and dt < max_dt:
            velocity = dy/dt                                    #find velocity
            #else:
            #    velocity = 0
        else:
            velocity = 0

        #print("Velocity: " , velocity)

        fall_speed = abs(velocity) > 200                        #if velocity exceeds 200 trigger fall_speed
        prev_hip_y = hip_mid[1]
        prev_time = current_time    
        #print("Speed: " , fall_speed)

        #final fall check
        if horizontal and fall_speed:
            fall_frames += 1
        else:
            fall_frames = 0
        #print("Fall frames: ", fall_frames)
        if fall_frames >= 1:
            status = "FALL DETECTED"

        #WRITE TO CSV
        if current_activity != status:
            with open("activity.csv", mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([datetime.now(), status])
            current_activity = status

    return {"activity": status}
