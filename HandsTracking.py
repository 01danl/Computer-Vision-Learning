#libraries import

import cv2
import time
import mediapipe as mp

"""
Basic code (bare minimum for CV handtracking)
#create our video object firstly
cap = cv2.VideoCapture(1) #Use our webcamera #1 (default)

while True:
    succes, img = cap.read()
    cv2.imshow("Image", img)
    cv2.waitKey(1)

"""

#create our video object firstly
cap = cv2.VideoCapture(0) #Use our webcamera #1 (default)

mpHands = mp.solutions.hands
hands = mpHands.Hands() #should involve some parameters static_image_mode, max_num_hands, model_complexity, min_detection_confidence, min_tracking_confidence
mpDraw = mp.solutions.drawing_utils

pTime = 0
current_Time = 0


while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # rgb - color
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark): # id - index number, lm from handlms.landmark take hand
                h,w,c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h) # id -> fingers
                if id == 4:
                    cv2.circle(img, (cx, cy), 25, (255, 0, 0), cv2.FILLED)
            mpDraw.draw_landmarks(
                img, handLms, mpHands.HAND_CONNECTIONS,
                mpDraw.DrawingSpec(color=(0, 0, 255), thickness=3, circle_radius=4),
                mpDraw.DrawingSpec(color=(0, 0, 255), thickness=3)
            )
    current_Time = time.time()
    fps = 1/(current_Time - pTime) # to calculate fps current - previous (time)
    pTime = current_Time

    cv2.putText(img, f"FPS: {int(fps)}", (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255),3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

"""
In this code we run a webcamera 
"""
