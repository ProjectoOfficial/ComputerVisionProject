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
from datetime import date, datetime

from RTCamera import RTCamera
from pynput.keyboard import Listener

from RTCamera import RTCamera
from Geometry import Geometry

CAMERA_DEVICE = 0
PRESSED_KEY = ''
CALIBRATE = False

def on_press(key):
    global PRESSED_KEY
    if hasattr(key, 'char'):
        if key.char is not None:
            if key.char in "qrgesci":
                PRESSED_KEY = key.char


listener = Listener(on_press=on_press)

if __name__ == "__main__":
    camera = RTCamera(CAMERA_DEVICE, fps=60, resolution=(640, 480))
    camera.start()

    start_fps = time.time()
    fps = 0
    listener.start()

    if CALIBRATE:
        geometry = Geometry(r"{}/Camera/Calibration/".format(os.getcwd()))
        calibrated, mtx, dist, rvecs, tvecs = geometry.get_calibration()
        camera.calibrate(calibrated, mtx, dist, rvecs, tvecs)

    while True:
        frame = camera.get_frame() 
        if camera.available():
            edges = cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 220, 230)

            cv2.putText(frame, str(fps) + " fps", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2, cv2.LINE_AA)
            H_stack = np.hstack((frame, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)))
            cv2.imshow("frame", H_stack)

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

            if PRESSED_KEY == 'g':
                gain = int(input("please insert the gain: "))
                camera.set_gain(gain)

            if PRESSED_KEY == 'e':
                exp = int(input("please insert the exposure: "))
                camera.set_exposure(exp)


            if PRESSED_KEY == 's':                  
                now = datetime.now()
                path = r"{}/Camera/Calibration/frame_{}.jpg".format(os.getcwd(), now.strftime("%d_%m_%Y__%H_%M_%S"))
                camera.save_frame(path)

                print("saved frame {} ".format(path))


            if PRESSED_KEY == 'c':
                print("Calibration in process, please wait...\n")
                cv2.destroyAllWindows()
                geometry = Geometry(r"{}/Camera/Calibration/".format(os.getcwd()))
                calibrated, mtx, dist, rvecs, tvecs = geometry.get_calibration()
                camera.calibrate(calibrated, mtx, dist, rvecs, tvecs)

            if PRESSED_KEY == 'i':
                print("Frame AVG value: {}".format(frame.mean(axis=(0,1))))

            if PRESSED_KEY != '':
                PRESSED_KEY = ''

    camera.stop()
    cv2.destroyAllWindows()
    print("closed")
