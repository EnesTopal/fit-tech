import cv2
import numpy as np
import PosEstimationModule as pm

# Video kaynağını başlat
path = "../videos/lat_pulldown_example.mp4"
cap = cv2.VideoCapture(0)

detector = pm.poseDetector()
count = 0
dir = 0
valid_back_angle = True  # Bel-omuz açısını kontrol etmek için bayrak

while True:
    _, img = cap.read()
    img = cv2.resize(img, (940, 580))
    img = detector.findPose(img, draw=False)
    lmList = detector.getPosition(img)
    if len(lmList) != 0:
        # Bel-omuz açısını hesapla (kalça-omuz-dirsek)
        back_angle = detector.findAngle(img, 23, 11, 13)
        if back_angle < 120 or back_angle > 330:  # Bu eşiği ihtiyaca göre ayarlayın
            valid_back_angle = False
            back_color = (0, 0, 255)  # Geçersiz açı için kırmızı renk
        else:
            valid_back_angle = True
            back_color = (0, 255, 0)  # Geçerli açı için yeşil renk

        # Bel-omuz açısını göstermek için çizgi çiz
        x1, y1 = lmList[23][1], lmList[23][2]  # Kalça
        x2, y2 = lmList[11][1], lmList[11][2]  # Omuz
        cv2.line(img, (x1, y1), (x2, y2), back_color, 3)

        # Dirsek açısını hesapla (omuz-dirsek-bilek)
        arm_angle = detector.findAngle(img, 11, 13, 15)
        per = np.interp(arm_angle, (25, 170), (0, 100))
        bar = np.interp(arm_angle, (25, 170), (430, 60))

        # Bel-omuz açısı geçerli ise lat pulldown sayısını kontrol et
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

        # Bar çiz
        cv2.rectangle(img, (760, 60), (810, 430), color, 3)
        cv2.rectangle(img, (760, int(bar)), (810, 430), color, cv2.FILLED)
        cv2.putText(img, f'{int(per)}%', (763, 25), cv2.FONT_HERSHEY_PLAIN, 2,
                    color, 2)

        cv2.rectangle(img, (29, 16), (118, 122), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(int(count)), (50, 100), cv2.FONT_HERSHEY_PLAIN, 5,
                    (255, 0, 0), 5)

        # Bel-omuz ve dirsek açılarını ekranda göster
        cx, cy = lmList[13][1], lmList[13][2]  # Dirsek koordinatları
        cv2.putText(img, str(int(arm_angle)), (cx - 50, cy - 20), cv2.FONT_HERSHEY_PLAIN, 2,
                    (255, 0, 0), 2)

        cx, cy = lmList[11][1], lmList[11][2]  # Omuz koordinatları
        cv2.putText(img, str(int(back_angle)), (cx - 50, cy - 20), cv2.FONT_HERSHEY_PLAIN, 2,
                    (255, 0, 0), 2)

    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()