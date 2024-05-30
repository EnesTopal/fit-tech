import cv2
import numpy as np
import PosEstimationModule as pm

# cap = cv2.VideoCapture(0)
path = "videos/workOutMyself.mp4"
cap = cv2.VideoCapture(0)

detector = pm.poseDetector()
count = 0
dir = 0


while True:
    _, img = cap.read()
    #img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    img = cv2.resize(img, (940, 580))
    img = detector.findPose(img, draw=False)
    lmList = detector.getPosition(img)
    if len(lmList) != 0:
        angle = detector.findAngle(img, 11, 13, 15)
        per = np.interp(angle, (210, 280), (0, 100))
        bar = np.interp(angle, (220, 280), (430, 60))
        # print(angle, per)

        # Check for the dumbbell curls
        color = (255, 0, 255)
        if per == 100:
            color = (0, 255, 0)
            if dir == 0:
                count += 0.5
                dir = 1
        if per == 0:
            color = (0, 255, 0)
            if dir == 1:
                count += 0.5
                dir = 0
        # print(count)

        # Draw Bar
        print(bar)
        cv2.rectangle(img, (760, 60), (810, 430), color, 3)
        cv2.rectangle(img, (760, int(bar)), (810, 430), color, cv2.FILLED)
        cv2.putText(img, f'{int(per)}%', (763, 25), cv2.FONT_HERSHEY_PLAIN, 2,
                    color, 2)

        cv2.rectangle(img, (29, 16), (118, 122), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(int(count)), (50, 100), cv2.FONT_HERSHEY_PLAIN, 5,
                    (255, 0, 0), 5)

        cx, cy = lmList[13][1], lmList[13][2]  # Get coordinates of the elbow (point 13)
        cv2.putText(img, str(int(angle)), (cx - 50, cy - 20), cv2.FONT_HERSHEY_PLAIN, 2,
                    (255, 0, 0), 2)

        # Calculate the slope between p1 and p2
        x1, y1 = lmList[11][1], lmList[11][2]  # p1 coordinates (shoulder)
        x2, y2 = lmList[13][1], lmList[13][2]  # p2 coordinates (elbow)
        slope = np.degrees(np.arctan2(y2 - y1, x2 - x1))

        # Draw line between p1 and p2 with color change condition
        line_color = (0, 255, 0)  # Default line color (green)
        if abs(slope) < 65 or abs(slope) > 95:
            line_color = (0, 0, 255)  # Change to red if slope > 30 degrees

        cv2.line(img, (x1, y1), (x2, y2), line_color, 3)

        cx, cy = lmList[11][1], lmList[11][2]  # Get coordinates of the elbow (point 13)
        cv2.putText(img, str(int(slope)), (cx - 50, cy - 20), cv2.FONT_HERSHEY_PLAIN, 2,
                    (255, 0, 0), 2)

    cv2.imshow('Image', img)
    cv2.waitKey(1)