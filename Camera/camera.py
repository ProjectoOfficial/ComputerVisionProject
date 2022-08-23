#export OPENBLAS_CORETYPE=ARMV8
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
import logging

import PIL

from RTCamera import RTCamera
from Geometry import Geometry
from Preprocessing import Preprocessing
from Distance import Distance
from traffic.traffic_video import Sign_Detector, Annotator, draw_circles

'''
INSTRUCTION:
    1) CAMERA DEVICE is the ID of the camera that is connected to your PC (if you only have one camera set i to 0). With this param, passed to the RTCamera constructor, you can 
        choose the camera to work with
    2) CALIBRATE allows to calculate camera distortion and calibrate the camera automatically before catching the first frame
    3) CHESSBOARD activates chessboard identification in a frame
    4) FILENAME is the name that this script will use to store the video recording 

'''

CAMERA_DEVICE = 0
PRESSED_KEY = ''
CALIBRATE = False
RECORDING = False
BLUR = False
TRANSFORMS = False
CHESSBOARD = False
FILENAME = "out"
RESOLUTION = (1280, 720)
SAVE_SIGN = False

# Colors
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# defining the fonts
fonts = cv2.FONT_HERSHEY_COMPLEX

def on_press(key):
    global PRESSED_KEY
    if hasattr(key, 'char'):
        if key.char is not None:
            if key.char in "qrgescibtf": # add here a letter if you want to insert a new command
                PRESSED_KEY = key.char

logging.getLogger("imported_module").setLevel(logging.ERROR)
listener = Listener(on_press=on_press)

if __name__ == "__main__":
    if not os.path.isdir(os.path.join(current, "signs")):
        os.makedirs(os.path.join(current, "signs"))

    if not os.path.isdir(os.path.join(current, "Calibration")):
        os.makedirs(os.path.join(current, "Calibration"))

    if not os.path.isdir(os.path.join(current, "Recordings")):
        os.makedirs(os.path.join(current, "Recordings"))

    camera = RTCamera(CAMERA_DEVICE, fps=30, resolution=RESOLUTION, cuda=True, auto_exposure=False, rotation=cv2.ROTATE_90_COUNTERCLOCKWISE)
    camera.start()

    start_fps = time.time()
    fps = 0
    listener.start()
    
    sd = Sign_Detector()
    an = Annotator(*RESOLUTION)
    an.org = (20, 50)
    circles = None
    speed = 0
    updates = 0

    if CALIBRATE:
        geometry = Geometry(os.path.join(current, 'Calibration'))
        calibrated, mtx, dist, rvecs, tvecs = geometry.get_calibration()
        camera.calibrate(calibrated, mtx, dist, rvecs, tvecs)

    while True:
        frame = camera.get_frame() 
        if frame is None:
            continue

        if frame.size == 0:
            continue

        if camera.available():
            if time.time() - start_fps > 1:
                fps = camera.get_fps()
                start_fps = time.time()

            if PRESSED_KEY == 'q': # QUIT
                listener.stop()
                listener.join()
                print("closing!")
                break

            if PRESSED_KEY == 'r': # REGISTER/STOP RECORDING
                if not RECORDING:
                    print("recording started...")
                    camera.register(os.path.join(current, "Recordings", "{}__{}.mp4".format(FILENAME, datetime.now().strftime("%d_%m_%Y__%H_%M_%S"))))
                    RECORDING = True
                else:
                    camera.stop_recording()
                    print("recording stopped!")
                    RECORDING = False

            if PRESSED_KEY == 'g' and not RECORDING: # CHANGE GAIN
                gain = int(input("please insert the gain: "))
                camera.set_gain(gain)

            if PRESSED_KEY == 'e' and not RECORDING: # CHANGE EXPOSURE
                exp = int(input("please insert the exposure: "))
                camera.set_exposure(exp)

            if PRESSED_KEY == 's' and not RECORDING: # SAVE CURRENT FRAME
                path = os.path.join(current, 'Calibration', 'frame_{}.jpg'.format(datetime.now().strftime("%d_%m_%Y__%H_%M_%S")))
                camera.save_frame(path)

                print("saved frame {} ".format(path))

            if PRESSED_KEY == 'c' and not RECORDING: # CALIBRATE CAMERA
                print("Calibration in process, please wait...\n")
                cv2.destroyAllWindows()
                geometry = Geometry(os.path.join(current, 'Calibration'))
                calibrated, mtx, dist, rvecs, tvecs = geometry.get_calibration()
                camera.calibrate(calibrated, mtx, dist, rvecs, tvecs)

            if PRESSED_KEY == 'i': # SHOW MEAN VALUE OF CURRENT FRAME
                print("Frame AVG value: {}".format(frame.mean(axis=(0, 1, 2))))

            if PRESSED_KEY == 'b': # BLUR FRAME
                BLUR = not BLUR
                print("blur: {}".format(BLUR))

            if PRESSED_KEY == 't' and not RECORDING: # APPLY TRANSFORMS TO FRAME
                TRANSFORMS = not TRANSFORMS
                print("transform: {}".format(TRANSFORMS))

            if PRESSED_KEY == 'f' and not RECORDING: # SHOW CHESSBOARD
                CHESSBOARD = not CHESSBOARD
                print("Chessboard: {}".format(CHESSBOARD))
                cv2.destroyAllWindows()
            
            if BLUR:
                frame = Preprocessing.GaussianBlur(frame, 1)

            if TRANSFORMS:
                frame = Preprocessing.Transform_base(frame)

            if CHESSBOARD:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(gray, (7,9), cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK)
                if ret:
                    cv2.drawChessboardCorners(frame, (7,9), corners, ret)

            if PRESSED_KEY != '':
                PRESSED_KEY = ''

            # Object detector (using face detector while waiting for Object detection to be ready)
            '''
            face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            bounding_boxes = face_detector.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1.3, 5)
            distances = Distance().get_Distances(bounding_boxes)
            
            for idx, (x, y, h, w) in enumerate(bounding_boxes):
                if idx == len(distances):
                    break
                cv2.rectangle(frame, (x, y), (x + w, y + h), GREEN, 2)
                cv2.putText(frame,"Distance: {:.2f}".format(distances[idx]), (x + 5, y + 20), fonts, 0.6, GREEN, 2)
            '''

            im = PIL.Image.fromarray(frame)
            converter = PIL.ImageEnhance.Color(im)
            im = converter.enhance(1.5)

            frame = np.asarray(im)

            height, width, _ = frame.shape
            h = height // 4
            w = width // 3
            found, c, s, u = sd.detect(frame, h, w)
            if found and s != 0:
                circles, speed, updates = c, s, u
                frame = draw_circles(frame, circles, (height, width), (height, width), (h, w))
                an.write(frame, speed, updates)
                if SAVE_SIGN:    
                    path = os.path.join(current, 'signs', 'sign_{}.jpg'.format(current, datetime.now().strftime("%d_%m_%Y__%H_%M_%S")))

                    cv2.imwrite(path, frame)
                
            an.write(frame, speed, updates)

            cv2.putText(frame, str(fps) + " fps", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow("frame", frame)
    camera.stop()
    cv2.destroyAllWindows()
    print("closed")
