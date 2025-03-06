import cv2
import mediapipe as mp
import time

class FaceDetection:
    def __init__(self, minDetectionConf = 0.5):
        self.MinDetectionConf = minDetectionConf

        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.FaceDetection = self.mpFaceDetection.FaceDetection(self.MinDetectionConf)

    def FindFaces(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.FaceDetection.process(imgRGB)
        res = [] # list to store box info, id#, score
        if self.results.detections:
            for id, det in enumerate(self.results.detections):
                # mpDraw.draw_detection(img, det) #rectangle face
                # print(det.score)  -> probability of face precise
                bboxC = det.location_data.relative_bounding_box
                ih, iw, ic = img.shape  # image height, image weight, image channels
                bbox = int((bboxC.xmin * iw)), int((bboxC.ymin * ih)), \
                    int((bboxC.width * iw)), int((bboxC.height * ih))
                self.draw_angle(img, bbox)
                score = int((det.score[0]) * 100)
                # print(det.score) -> list [0] index
                res.append([id, bbox, det.score])
                cv2.putText(img, f"prediction_of_face: {score}%", ((int(bboxC.xmin * iw)-20), (int(bboxC.ymin * ih)-20)),
                            cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
        return img, res

    def draw_angle(self, img, bbox, length=30, thickness=5):
       x, y, w, h = bbox
       x1 = x + w #diagonals
       y1 = y + h #diagonals

       cv2.rectangle(img, bbox, (0, 255, 255), 2)
       #Top Left
       cv2.line(img, (x,y), (x+length, y), (0, 0, 255), thickness)
       cv2.line(img, (x,y), (x, y+length), (0, 0, 255), thickness)
       #Top right
       cv2.line(img, (x1,y), (x1-length, y), (0, 0, 255), thickness)
       cv2.line(img, (x1,y), (x1, y+length), (0, 0, 255), thickness)
       #Bottom left
       cv2.line(img, (x,y1), (x+length, y1), (0, 0, 255), thickness)
       cv2.line(img, (x,y1), (x, y1-length), (0, 0, 255), thickness)
       #Bottom right
       cv2.line(img, (x1,y1), (x1-length, y1), (0, 0, 255), thickness)
       cv2.line(img, (x1,y1), (x1, y1-length), (0, 0, 255), thickness)
       return img

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = FaceDetection()

    while True:
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        success, img = cap.read()
        img, res = detector.FindFaces(img)
        cv2.putText(img, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
        cv2.imshow('Face Detection', img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
