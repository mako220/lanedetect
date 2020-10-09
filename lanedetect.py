import cv2
import numpy as np
#import matplotlib.pyplot as plt

img = cv2.imread('lanedetect.png')

height, width = img.shape[:2]

img = cv2.resize(img,(width//2, height//2))

grey = np.mean(img, axis=2).astype(np.uint8)

height1, width1 = grey.shape[:2]

print("height : ", height1, " width : ", width1)

blur = cv2.GaussianBlur(grey, (7, 7), 0)

canny = cv2.Canny(blur, 50, 150)

roi_poly = np.array([
[(349, 131), (638, 131), (728, 261), (292, 261)]
], 'int32')

#mask2 = np.zeros_like(grey)

#print(mask2.dtype)

mask = np.zeros((height1, width1), dtype = np.uint8)

#print(mask.dtype)

cv2.fillPoly(mask, roi_poly, (255, 255, 255))

#cv2.imshow('mask', mask)
#cv2.waitKey(0)

roi = cv2.bitwise_and(canny, mask)

#cv2.imshow('masked_image', roi)
#cv2.waitKey(0)

lines = cv2.HoughLinesP(roi, 2, np.pi/180, 50, np.array([]), minLineLength = 40, maxLineGap = 5)

line_image = np.zeros_like(img)

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)

final_image = cv2.addWeighted(img, 0.7, line_image, 1, 1)
#cv2.imshow('original', img)
cv2.imshow('result', final_image)
cv2.waitKey(0)
#plt.imshow(final_image)
#plt.show( )
