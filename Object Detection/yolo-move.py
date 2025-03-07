from ultralytics import YOLO
import cv2
import cvzone
import math
import time


cap = cv2.VideoCapture("Your route")


classNames = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase",
    "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "TV", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

model = YOLO('../Yolo Weights/yolov8n.pt')

while True:
    success, img = cap.read()
    results = model(img, stream=True) #stream -> use generators stream
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # first method with cv2 -> x1,y1,x2,y2
           # x1, y1, x2, y2 = box.xyxy[0]
           # x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            #print(x1,y2,w,h)
           # cv2.rectangle(img, (x1,y1), (x2,y2) , (0,0,255) ,2)

            #second method
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w,h = x2-x1, y2-y1
            cvzone.cornerRect(img, (x1, y1, w, h))

            cls = int(box.cls[0])
            conf = math.ceil((box.conf[0] * 100))/100
            cvzone.putTextRect(img, f"{classNames[cls]} {conf}%", (x1,y1-20))


    cv2.imshow("Webcamera", img)
    cv2.waitKey(1)
