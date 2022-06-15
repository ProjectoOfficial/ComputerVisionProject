#export OPENBLAS_CORETYPE=ARMV8
'''
(C) Dott. Daniel Rossi - Università degli Studi di Modena e Reggio Emilia
    Computer Vision Project - Artificial Intelligence Engineering

    Real Time Camera script for road analysis


    Camera model: See3Cam_CU27 REV X1
'''
import os
import sys
  
current = os.path.dirname(os.path.realpath(__file__))  
parent = os.path.dirname(current)
sys.path.append(parent)

import cv2
import time
import numpy as np

from RTCamera import RTCamera
from pynput.keyboard import Listener

from RTCamera import RTCamera
import Geometry

CAMERA_DEVICE = 1
PRESSED_KEY = ''

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
    camera = RTCamera(CAMERA_DEVICE, fps=100, resolution=(640, 480))
    camera.start()

    start_fps = time.time()
    fps = 0
    listener.start()

    while True:
        frame = camera.get_frame() 
        if camera.available():
            cv2.putText(frame, str(fps) + " fps", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2, cv2.LINE_AA)
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