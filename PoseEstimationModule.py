import cv2
import time
import mediapipe as mp

#create our video object firstly

class HandDetector():
    def __init__(self, mode=False, maxHands = 2, detectionConf = 0.5, trackConf = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionConf = detectionConf
        self.trackConf = trackConf

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionConf,
            min_tracking_confidence=self.trackConf
        )  # should involve some parameters static_image_mode, max_num_hands, model_complexity, min_detection_confidence, min_tracking_confidence
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # rgb - color
        results = self.hands.process(imgRGB)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img, handLms, self.mpHands.HAND_CONNECTIONS,
                        self.mpDraw.DrawingSpec(color=(0, 0, 255), thickness=3, circle_radius=4),
                        self.mpDraw.DrawingSpec(color=(0, 0, 255), thickness=3)
                    )
        return img


def main():
    pTime = 0
    current_Time = 0
    cap = cv2.VideoCapture(0, cv2.CAP_ANY)
    detector = HandDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)

        current_Time = time.time()
        fps = 1 / (current_Time - pTime)  # to calculate fps current - previous (time)
        pTime = current_Time

        cv2.putText(img, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__== "__main__": # if we running this script
    main()
