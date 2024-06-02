import cv2
import numpy as np
import PosEstimationModule as pm

# cap = cv2.VideoCapture(0)
path = "../videos/squat_example.mp4"
cap = cv2.VideoCapture(path)

detector = pm.poseDetector()
count = 0
dir = 0
valid_back_angle = True  # Flag to check if the back angle is valid

while True:
    _, img = cap.read()
    img = cv2.resize(img, (940, 580))
    img = detector.findPose(img, draw=False)
    lmList = detector.getPosition(img)
    if len(lmList) != 0:
        # Calculate back angle (shoulder-hip-knee)
        back_angle = detector.findAngle(img, 11, 23, 25)
        if back_angle < 170 or back_angle > 325:  # Adjust this threshold as needed
            valid_back_angle = False
            back_color = (0, 0, 255)  # Red color for invalid back angle
        else:
            valid_back_angle = True
            back_color = (0, 255, 0)  # Green color for valid back angle

        # Draw line to indicate back angle
        x1, y1 = lmList[11][1], lmList[11][2]  # shoulder
        x2, y2 = lmList[23][1], lmList[23][2]  # hip
        cv2.line(img, (x1, y1), (x2, y2), back_color, 3)

        # Calculate leg angle (hip-knee-ankle)
        leg_angle = detector.findAngle(img, 23, 25, 27)
        per = np.interp(leg_angle, (55,155), (0, 100))
        bar = np.interp(leg_angle, (55,155), (430, 60))

        # Check for the squats only if the back angle is valid
        color = (255, 0, 255)
        if valid_back_angle:
            if per == 0:
                color = (0, 255, 0)
                if dir == 0:
                    count += 0.5
                    dir = 1
            if per == 100:
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

        # Display back and leg angles
        cx, cy = lmList[25][1], lmList[25][2]  # Get coordinates of the knee (point 25)
        cv2.putText(img, str(int(leg_angle)), (cx - 50, cy - 20), cv2.FONT_HERSHEY_PLAIN, 2,
                    (255, 0, 0), 2)

        cx, cy = lmList[23][1], lmList[23][2]  # Get coordinates of the hip (point 23)
        cv2.putText(img, str(int(back_angle)), (cx - 50, cy - 20), cv2.FONT_HERSHEY_PLAIN, 2,
                    (255, 0, 0), 2)

    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
