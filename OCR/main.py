import cv2
import time
from preProcess import *
from textExtract import *

cam = cv2.VideoCapture(0)

cv2.namedWindow("audiofy")

img_counter = 0

org_time = time.time()

while True:
    cur_time = time.time()
    ret, frame = cam.read()
    cv2.imshow("audiofy", frame)
    if not ret:
        break
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif cur_time-org_time >= 5:
        org_time = cur_time
        img_name = "images/pic_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))

        # Pre-processing the captured image
        preProcess("images/image2.jpg")
        print("Pre-processing done!")

        # Text extraction
        extract("processedImage/resultImage3.jpg")
        print("Text extracted successfully!")
        
        img_counter += 1

cam.release()

cv2.destroyAllWindows()