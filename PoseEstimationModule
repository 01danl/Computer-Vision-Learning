import cv2
import mediapipe as mp
import time

class PoseDetector():
    def __init__(self, mode=False, complexity=1, sm_lm = True, en_seg=True, sm_seg=True, min_det_conf=0.5, min_track_conf=0.5):
        self.mode = mode
        self.complexity = complexity
        self.sm_lm = sm_lm
        self.en_seg = en_seg
        self.sm_seg = sm_seg
        self.min_det_conf = min_det_conf
        self.min_track_conf = min_track_conf

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.complexity, self.sm_lm, self.en_seg, self.sm_seg)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                for poseLMS in self.results.pose_landmarks.landmark:
                    h, w, c = img.shape
                    cx, cy = int(poseLMS.x * w), int(poseLMS.y * h)
                    size = 15
                    cv2.rectangle(img, (cx - size, cy - size), (cx + size, cy + size), (0, 255, 0), 2)

                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS,
                                      self.mpDraw.DrawingSpec(color=(20, 220, 53), thickness=3),
                                      self.mpDraw.DrawingSpec(color=(255, 0, 255), thickness=5)
                                      )


       # for id, lm in enumerate(results.pose_landmarks.landmark):
           # a,b,c = img.shape
           # c1, c2 = int(lm.x * a), int(lm.y * b)
           # cv2.circle(img, (c1, c2), 25, (255, 0, 0), cv2.FILLED)
        return img
    def getPosition(self, img, draw=True):
        res = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h,w,c = img.shape

                cx, cy = int(lm.x * w), int(lm.y * h)
                res.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return res
def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = PoseDetector()

    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        res = detector.getPosition(img)
        print(res)

        cv2.imshow("Pose tracking", img)
        cv2.waitKey(1)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)


if __name__ == "__main__":
    main()
