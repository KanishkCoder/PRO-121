import cv2
import time
import numpy as np

fourcc = cv2.VideoWriter_fourcc(*"XVID")
output_file = cv2.VideoWriter("Output.avi",fourcc,20.0,(640,480))

cap = cv2.VideoCapture(0)

time.sleep(2)
bg = 0

for i in range(60):
    ret, bg = cap.read()

bg = np.flip(bg, axis = 1)

while(cap.isOpened()):
    ret, image = cap.read()
    if not ret:
        break

    image = np.flip(image, axis = 1)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    image = cv2.resize(image, (640, 480))
    frame = cv2.resize(frame, (640, 480))
    
    lower_red = np.array([104,153,70])
    upper_red = np.array([30,30,0])
    mask_1 = cv2.inRange(hsv,lower_red,upper_red)

    mask = cv2.morphologyEx(mask_1, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))

    res = cv2.bitwise_and(frame, frame, mask = mask)
    
    final_output = cv2.addWeighted(res, 1, res, 1, 0)
    output_file.write(final_output)
    cv2.imshow('magic',final_output)
    cv2.waitKey(1)

cap.release()
out.release()
cv2.destroyAllWindows()