#export OPENBLAS_CORETYPE=ARMV8
__author__ = "Daniel Rossi, Riccardo Salami, Filippo Ferrari"
__copyright__ = "Copyright 2022"
__credits__ = ["Daniel Rossi", "Riccardo Salami", "Filippo Ferrari"]
__license__ = "GPL-3.0"
__version__ = "1.0.0"
__maintainer__ = "Daniel Rossi"
__email__ = "miniprojectsofficial@gmail.com"
__status__ = "Computer Vision Exam"

import os
import sys

current = os.path.dirname(os.path.realpath(__file__))  
parent = os.path.dirname(current)
sys.path.append(parent)

import cv2
import time
import numpy as np
from datetime import datetime

from pynput.keyboard import Listener

from RTCamera import RTCamera
from Geometry import Geometry
from Preprocessing import Preprocessing

'''
INSTRUCTION:
    1) CAMERA DEVICE is the ID of the camera that is connected to your PC (if you only have one camera set i to 0). With this param, passed to the RTCamera constructor, you can 
        choose the camera to work with
    2) CALIBRATE allows to calculate camera distortion and calibrate the camera automatically before catching the first frame
    3) FILENAME is the name that this script will use to store the video recording 

'''

CAMERA_DEVICE = 0
PRESSED_KEY = ''
CALIBRATE = False
BLUR = False
TRANSFORMS = False
FILENAME = "out"

def on_press(key):
    global PRESSED_KEY
    if hasattr(key, 'char'):
        if key.char is not None:
            if key.char in "qrgescibt": # add here a letter if you want to insert a new command
                PRESSED_KEY = key.char


listener = Listener(on_press=on_press)

if __name__ == "__main__":
    camera = RTCamera(CAMERA_DEVICE, fps=60, resolution=(640, 480), cuda=True)
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
        if frame is None:
            continue

        if camera.available():
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
                camera.register("{}.mp4".format(FILENAME))

            if PRESSED_KEY == 'g':
                gain = int(input("please insert the gain: "))
                camera.set_gain(gain)

            if PRESSED_KEY == 'e':
                exp = int(input("please insert the exposure: "))
                camera.set_exposure(exp)

            if PRESSED_KEY == 's':                  
                now = datetime.now()
                if not os.path.isdir(current + "/Calibration"):
                    os.makedirs(current + "/Calibration")
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
                print("Frame AVG value: {}".format(frame.mean(axis=(0, 1, 2))))

            if PRESSED_KEY == 'b':
                BLUR = not BLUR
                print("blur: {}".format(BLUR))

            if PRESSED_KEY == 't':
                TRANSFORMS = not TRANSFORMS
                print("transform: {}".format(TRANSFORMS))
            
            if BLUR:
                frame = Preprocessing.GaussianBlur(frame, 1)

            if TRANSFORMS:
                frame = Preprocessing.Transforms(frame)

            if PRESSED_KEY != '':
                PRESSED_KEY = ''

                
            edges = cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 220, 230)

            frame = frame.copy()

            cv2.putText(frame, str(fps) + " fps", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2, cv2.LINE_AA)
            H_stack = np.hstack((frame, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)))
            cv2.imshow("frame", H_stack)

    camera.stop()
    cv2.destroyAllWindows()
    print("closed")
