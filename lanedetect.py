import cv2

img = cv2.imread('lanedetect.png')

cv2.imshow('result', img)
cv2.waitKey(0)
