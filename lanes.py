import cv2
import numpy as np


cap = cv2.VideoCapture('lanedetect.mp4')
width = int(cap.get(3))
height = int(cap.get(4))
fps = cap.get(5)
print(fps)


roi_poly = np.array([
[(349, 131), (638, 131), (728, 261), (292, 261)]
], 'int32')

#mask2 = np.zeros_like(grey)

#print(mask2.dtype)

mask = np.zeros((540, 960), dtype = np.uint8)

#print(mask.dtype)

cv2.fillPoly(mask, roi_poly, (255, 255, 255))


cv2.namedWindow("Frame")

while(cap.isOpened()):
    _, frame = cap.read()

    img = cv2.resize(frame , (width//2,height//2))

    grey = np.mean(img, axis=2).astype(np.uint8)

    print(grey.shape)

    #grey = np.mean(img, axis=2).astype(np.uint8)

    blur = cv2.GaussianBlur(grey, (7, 7), 0)

    canny = cv2.Canny(blur, 50, 150)

    roi = cv2.bitwise_and(canny, mask)

    lines = cv2.HoughLinesP(roi, 2, np.pi/180, 50, np.array([]), minLineLength = 40, maxLineGap = 5)

    line_image = np.zeros_like(img)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)

    final_image = cv2.addWeighted(img, 0.7, line_image, 1, 1)

    cv2.imshow("Frame", final_image)
    key = cv2.waitKey(1)
    if key == 27:
         break

cap.release()
cv2.destroyAllWindows()
