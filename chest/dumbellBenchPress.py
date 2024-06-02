import cv2
import numpy as np
import PosEstimationModule as pm

# Video kaynağını başlat
path = "../videos/workOutMyself.mp4"
cap = cv2.VideoCapture(0)

detector = pm.poseDetector()
count = 0
dir = 0

while True:
    _, img = cap.read()
    img = cv2.resize(img, (940, 580))
    img = detector.findPose(img, draw=False)
    lmList = detector.getPosition(img)

    if len(lmList) != 0:
        # Omuz (11), dirsek (13) ve bilek (15) noktaları arasındaki açıyı bul
        angle = detector.findAngle(img, 11, 13, 15)
        per = np.interp(angle, (200,300), (0, 100))  # Açı aralığını bench press için ayarla
        bar = np.interp(angle, (200, 300), (430, 60))

        # p1 ve p2 arasındaki eğimi hesapla (omuz ve dirsek)
        x1, y1 = lmList[11][1], lmList[11][2]  # Omuz koordinatları (p1)
        x2, y2 = lmList[13][1], lmList[13][2]  # Dirsek koordinatları (p2)

        # p1 ve p2 arasına çizgi çiz
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Dumbbell bench press sayısını kontrol et
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

        # Bar çiz
        cv2.rectangle(img, (760, 60), (810, 430), color, 3)
        cv2.rectangle(img, (760, int(bar)), (810, 430), color, cv2.FILLED)
        cv2.putText(img, f'{int(per)}%', (763, 25), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

        # Sayım göster
        cv2.rectangle(img, (29, 16), (118, 122), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(int(count)), (50, 100), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)

        # Açı ve eğim değerlerini ekranda göster
        cx, cy = lmList[13][1], lmList[13][2]  # Dirsek koordinatları
        cv2.putText(img, str(int(angle)), (cx - 50, cy - 20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    cv2.imshow('Image', img)
    cv2.waitKey(1)
