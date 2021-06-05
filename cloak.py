import time
import cv2
import numpy as np

# To save the output in a file (avi extension)
fourCC = cv2.VideoWriter_fourcc(*"XVID")

output_file = cv2.VideoWriter("output.avi", fourCC, 20, (640, 480))

camera = cv2.VideoCapture(0)

time.sleep(2)

background = 0

for i in range(60):
    ret, background = camera.read()

background = np.flip(background, axis=1)

while camera.isOpened():
    ret, img = camera.read()

    if not ret:
        break

    img = np.flip(img, axis=1)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0, 120, 50])
    upper_red = np.array([10, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    lower_red = np.array([170, 120, 70])
    upper_red = np.array([180, 255, 255])

    mask2 = cv2.inRange(hsv, lower_red, upper_red)

    mask1 = mask1 + mask2

    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.once((3, 3)), np.uint8)
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_DILUTE, np.once((3, 3)), np.uint8)

    # Except the mask1 colour, all the image will be saved.
    mask2 = cv2.bitwise_not(mask1)

    result1 = cv2.bitwise_and(img, img, mask=mask2)

    result2 = cv2.bitwise_and(background, background, mask=mask1)

    final_result = cv2.AddWeighted(result1, 1, result2, 1, 0)

    output_file.write(final_result)

    cv2.imshow("magic", final_result)
    cv2.waitKey(1)

camera.release()
cv2.destroyAllWindows()
