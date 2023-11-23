import cv2 as cv
import numpy as np
import os
from time import time
from windowcapture import WindowCapture
from vision import Vision

os.chdir(os.path.dirname(os.path.abspath(__file__)))


wincap = WindowCapture('Nazwa Okna')

cascade_diamond = cv.CascadeClassifier('diamond_model.xml')

vision_diamond = Vision(None)

loop_time = time()
while(True):
    # gray_diamonds_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    screenshot = wincap.get_screenshot()

    rectangles = cascade_diamond.detectMultiScale(screenshot)

    detection_image = vision_diamond.draw_rectangles(screenshot, rectangles)

    cv.imshow('Diamonds', detection_image)

    print('FPS {}'.format(1 / (time() - loop_time)))
    loop_time = time()

    # press 'q' with the output window focused to exit.
    # press 'f' to save screenshot as a positive image, press 'd' to 
    # save as a negative image.
    # waits 1 ms every loop to process key presses
    key = cv.waitKey(1)
    if key == ord('q'):
        cv.destroyAllWindows()
        break
    elif key == ord('p'):
        cv.imwrite('positive/{}.jpg'.format(loop_time), screenshot)
    elif key == ord('n'):
        cv.imwrite('negative/{}.jpg'.format(loop_time), screenshot)


print('Done.')
