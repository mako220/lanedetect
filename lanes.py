import cv2
import numpy as np


cap = cv2.VideoCapture('lanedetect.mp4')

cv2.namedWindow("Frame")

while(cap.isOpened()):
    _, frame = cap.read()

    img = cv2.resize(frame , (1920//2,1080//2))

    gray = np.mean(img, axis=2).astype(np.uint8)
    #print(img.shape)
    #print(img)
    cv2.imshow("Frame", img)
    key = cv2.waitKey(1)
    if key == 27:
         break

cap.release()
cv2.destroyAllWindows()
