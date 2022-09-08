#export OPENBLAS_CORETYPE=ARMV8
import os
import sys
from pathlib import Path

current = os.path.dirname(os.path.realpath(__file__))  
parent = os.path.dirname(current)
sys.path.append(parent)

import torch
import numpy as np
import cv2
import time
from datetime import datetime

from pynput.keyboard import Listener
import logging

from RTCamera import RTCamera
from Geometry import Geometry
from Preprocessing import Preprocessing
from Distance import Distance
from traffic.traffic_video import Sign_Detector, Annotator
from Tracking import Tracking

from Models.YOLOv7.yolo_test import Test
from Models.YOLOv7.utils.general import increment_path, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy

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
SAVE_SIGN = True

# Yolo parameters
BATCH_SIZE = 32
CONF_THRES= 0.001
DEVICE = '0'
IOU_THRES= 0.65  # for NMS
NAME = 'camera'
PROJECT = os.path.join(parent, 'Models', 'YOLOv7', 'runs', 'test') # save dir
SAVE_HYBRID = False
SAVE_TXT = False | SAVE_HYBRID
WEIGHTS = os.path.join(parent, 'Models', 'YOLOv7', 'last.pt')

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

preprocessor = Preprocessing((640, 640))

if __name__ == "__main__":
    save_dir = Path(increment_path(Path(PROJECT) / NAME, exist_ok=False))  # increment run
    (save_dir / 'labels' if SAVE_TXT else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

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

    tester = Test(WEIGHTS, BATCH_SIZE, DEVICE, save_dir)
    names = tester.model.names

    tracker = Tracking()

    # Main infinite loop
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
                frame = preprocessor.GaussianBlur(frame, 1)

            if TRANSFORMS:
                (frame, _) = preprocessor.Transform_base(frame)

            if CHESSBOARD:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(gray, (7,9), cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK)
                if ret:
                    cv2.drawChessboardCorners(frame, (7,9), corners, ret)

            if PRESSED_KEY != '':
                PRESSED_KEY = ''

            # Object Recognition
            img, _ = preprocessor.Transform_base(frame)
            img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(dim=0)
            out, train_out = tester.predict(img)
            out = non_max_suppression(out, conf_thres=CONF_THRES, iou_thres=IOU_THRES, multi_label=True)

            detections = []
            for si, pred in enumerate(out):
                predn = pred.clone()
                ratio = ((1, 1), (0, 0))
                scale_coords(img.shape[1:], predn[:, :4], (640, 640), ratio)  # native-space pred

                for *xyxy, conf, cls in predn.tolist():
                    if conf > 0.7:
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1) # xywh
                        xywh = [ int(x) for x in xywh ]
                        x, y, w, h = xywh

                        detections.append((cls, xywh))
                        distance = Distance().get_Distance(xywh)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        cv2.putText(frame, "{:.2f} {} {:.2f}".format(conf, names[int(cls)], distance), (x + 5, y + 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 0, 255), 1)

            # Tracking
            hsvframe =  cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
            tracker.zero_objects()
            for cls, box in detections:
                x, y, w, h = box
                box[0] = int(box[0] + box[2]/2)
                box[1] = int(box[1] + box[3]/2)
                id = tracker.update_obj(cls, box)

                prediction, pts = tracker.track(hsvframe, box)
                cv2.putText(frame, "ID: {}".format(id), (x - 60, y + 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 0, 255), 1)
                cv2.putText(frame, "ID: {}".format(id), (int(prediction[0] - (0.5 * w)) + 5, int(prediction[1] - (0.5 * h)) + 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 0, 255), 1)
                cv2.rectangle(frame, (int(prediction[0] - (0.5 * w)), int(prediction[1] - (0.5 * h))), (int(prediction[0] + (0.5 * w)), int(prediction[1] + (0.5 * h))), (0, 255, 0), 2)
            tracker.clear_objects()

            # traffic sign detection
            height, width, _ = frame.shape
            h = height // 4
            w = width // 3
            found, c, s, u = sd.detect(frame, h, w)
            if found and s != 0:
                circles, speed, updates = c, s, u

                if circles is not None:
                    frame = an.draw_circles(frame, circles, (height, width), (height, width), (h, w))

                an.write(frame, speed, updates)
                if SAVE_SIGN:    
                    path = os.path.join(current, 'signs', 'sign_{}.jpg'.format(datetime.now().strftime("%d_%m_%Y__%H_%M_%S")))
                    cv2.imwrite(path, frame)
                
            an.write(frame, speed, updates)

            cv2.putText(frame, str(fps) + " fps", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow("frame", frame)
    camera.stop()
    cv2.destroyAllWindows()
    print("closed")
