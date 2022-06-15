#export OPENBLAS_CORETYPE=ARMV8
'''
(C) Dott. Daniel Rossi - Università degli Studi di Modena e Reggio Emilia
    Computer Vision Project - Artificial Intelligence Engineering

    Real Time Camera script for road analysis


    Camera model: See3Cam_CU27 REV X1
'''

import cv2
import time
import numpy as np

from RTCamera import RTCamera
from pynput.keyboard import Key, Listener

from RTCamera import RTCamera

CAMERA_DEVICE = 0
TRESH_MODE = "ADAPTIVE_GAUSSIAN" # OTSU ADAPTIVE_GAUSSIAN ADAPTIVE_MEAN

PRESSED_KEY = ''

def processing(img: np.ndarray):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        th = None

        if TRESH_MODE == "OTSU":
            ret, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif TRESH_MODE == "ADAPTIVE_GAUSSIAN":
            th = cv2.adaptqiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 19, 4)
        elif TRESH_MODE == "ADAPTIVE_MEAN":
            th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

        canny = cv2.Canny(blur, 50, 5)

        rgb_th = cv2.cvtColor(th ,cv2.COLOR_GRAY2RGB)
        rgb_canny = cv2.cvtColor(canny ,cv2.COLOR_GRAY2RGB)
        rgb_blur = cv2.cvtColor(blur, cv2.COLOR_GRAY2RGB)

        H_stack = np.hstack((rgb_blur, rgb_th, rgb_canny))

        cv2.imshow("images", H_stack)


def on_press(key):
    global PRESSED_KEY
    if hasattr(key, 'char'):
        if key.char == 'q':
            PRESSED_KEY = 'q'

        if key.char == 'r':
            PRESSED_KEY = 'r'

        if key.char == 'g':
            PRESSED_KEY = 'g'

        if key.char == 'e':
            PRESSED_KEY = 'e'

listener = Listener(on_press=on_press)

if __name__ == "__main__":
    camera = RTCamera(CAMERA_DEVICE, resolution=(640, 480))
    camera.start()

    start_fps = time.time()
    fps = 0
    listener.start()

    while True:
        frame = camera.get_frame()
        if camera.available():
            cv2.putText(frame, str(fps) + " fps", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2, cv2.LINE_AA)
            frame = cv2.resize(frame, (640, 480))
            cv2.imshow("frame", frame)

            if time.time() - start_fps > 2:
                fps = camera.get_fps()
                start_fps = time.time()

            if PRESSED_KEY == 'q':
                listener.stop()
                listener.join()
                print("closing!")
                break

            if PRESSED_KEY == 'r':
                print("recording started...")
                camera.register("out.mp4")
                PRESSED_KEY = ''

            if PRESSED_KEY == 'g':
                gain = int(input("please insert the gain: "))
                camera.set_gain(gain)
                PRESSED_KEY = ''

            if PRESSED_KEY == 'e':
                exp = int(input("please insert the exposure: "))
                camera.set_exposure(exp)
                PRESSED_KEY = ''

    camera.stop()
    cv2.destroyAllWindows()
    print("closed")
