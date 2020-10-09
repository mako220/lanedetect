import cv2
import numpy as np


cap = cv2.VideoCapture('lanedetect.mp4')
width = int(cap.get(3))
height = int(cap.get(4))
fps = cap.get(5)
print(fps)

cv2.namedWindow("Frame")

while(cap.isOpened()):
    _, frame = cap.read()

    img = cv2.resize(frame , (width//2,height//2))

    grey = np.mean(img, axis=2).astype(np.uint8)
    #print(img.shape)
    #print(img)
    cv2.imshow("Frame", grey)
    key = cv2.waitKey(1)
    if key == 27:
         break

cap.release()
cv2.destroyAllWindows()
