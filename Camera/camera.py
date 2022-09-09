#export OPENBLAS_CORETYPE=ARMV8
import os
import sys
from pathlib import Path
import argparse

current = os.path.dirname(os.path.realpath(__file__))  
parent = os.path.dirname(current)
sys.path.append(parent)

import torch
import cv2
import time
from datetime import datetime
import csv

from pynput.keyboard import Listener

from RTCamera import RTCamera
from Geometry import Geometry
from Preprocessing import Preprocessing
from Distance import Distance
from traffic.traffic_video import Sign_Detector, Annotator
from Tracking import Tracking


from Models.YOLOv7.yolo_test import Test
from Models.YOLOv7.utils.general import increment_path, non_max_suppression, scale_coords, xyxy2xywh

PRESSED_KEY = ''
RECORDING = False
BLUR = False
TRANSFORMS = False
CHESSBOARD = False
ROTATION = None

def on_press(key):
    global PRESSED_KEY
    if hasattr(key, 'char'):
        if key.char is not None:
            if key.char in "qrgescibtf": # add here a letter if you want to insert a new command
                PRESSED_KEY = key.char

listener = Listener(on_press=on_press)
preprocessor = Preprocessing((640, 640))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-b', '--batch-size', type=int, default=1, help='YOLOv7 batch-size')
    parser.add_argument('-c', '--calibrate', action='store_true', default=False, help='true if you want to calibrate the camera')
    parser.add_argument('-cd', '--camera-device', type=int, default=0, help='Camera device ID')
    parser.add_argument('-ct', '--conf-thres', type=float, default=0.001, help='YOLOv7 conf threshold')
    parser.add_argument('-d', '--device', type=str, default='0', help='cuda device(s)')
    parser.add_argument('-it', '--iou-thres', type=float, default=0.65, help='YOLOv7 iou threshold')
    parser.add_argument('-f', '--filename', type=str, default='out', help='filename for recordings')
    parser.add_argument('-j', '--jetson', action='store_true', default=False, help='true if you are using the Nvidia Jetson Nano')
    parser.add_argument('-l', '--label', action='store_true', default=False, help='true if you want to save labelled signs')
    parser.add_argument('-n', '--name', type=str, default='camera', help='YOLOv7 result test directory name')
    parser.add_argument('-p', '--project', type=str, default=os.path.join(parent, 'Models', 'YOLOv7', 'runs', 'test') , help='YOLOv7 project save directory')
    parser.add_argument('-r', '--resolution', type=tuple, default=(1280, 720), help='camera resolution')
    parser.add_argument('-rt', '--rotate', action='store_true', default=False, help='rotate frame for e-con camera')
    parser.add_argument('-s', '--save-sign', action='store_true', default=False, help='save frames which contain signs')
    parser.add_argument('-sh', '--save-hybrid', action='store_true', default=False, help='YOLOv7 save hybrid')
    parser.add_argument('-st', '--save-txt', action='store_true', default=False, help='YOLOv7 save txt')
    parser.add_argument('-w', '--weights', type=str, default=os.path.join(parent, 'Models', 'YOLOv7', 'last.pt') , help='YOLOv7 weights')
    opt = parser.parse_args()

    opt.save_txt |= opt.save_hybrid

    if opt.rotate:
        ROTATION = cv2.ROTATE_90_COUNTERCLOCKWISE

    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=False))  # increment run
    (save_dir / 'labels' if opt.save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    if not os.path.isdir(os.path.join(current, "signs")):
        os.makedirs(os.path.join(current, "signs"))

    if not os.path.isdir(os.path.join(current, "Calibration")):
        os.makedirs(os.path.join(current, "Calibration"))

    if not os.path.isdir(os.path.join(current, "Recordings")):
        os.makedirs(os.path.join(current, "Recordings"))

    if not os.path.isdir(os.path.join(current, "ItalianSigns")):
        os.makedirs(os.path.join(current, "ItalianSigns"))

    if not os.path.isdir(os.path.join(current, "ItalianSigns" , 'images')):
        os.makedirs(os.path.join(current, "ItalianSigns", 'images'))

    if not os.path.isdir(os.path.join(current, "ItalianSigns", 'labels')):
        os.makedirs(os.path.join(current, "ItalianSigns", 'labels'))

    if not os.path.isfile(os.path.join(current, "ItalianSigns", 'labels', 'ItalianSigns.csv')):
        f = open(os.path.join(current, "ItalianSigns", 'labels', 'ItalianSigns.csv'), 'w')
        writer = csv.writer(f)
        writer.writerow(["filename", "x top left", "y top left", "x bottom right",  "y bottom right", "speed limit", "valid"])
        f.close()

    camera = RTCamera(opt.camera_device, fps=30, resolution=opt.resolution, cuda=True, auto_exposure=False, rotation=ROTATION)
    camera.start()

    start_fps = time.monotonic()
    fps = 0
    listener.start()
    
    sd = Sign_Detector()
    an = Annotator(*opt.resolution)
    an.org = (20, 50)
    circles = None
    speed = 0
    updates = 0

    if opt.calibrate:
        geometry = Geometry(os.path.join(current, 'Calibration'))
        calibrated, mtx, dist, rvecs, tvecs = geometry.get_calibration()
        camera.calibrate(calibrated, mtx, dist, rvecs, tvecs)

    tester = None
    names = None
    tracker = None

    if not opt.jetson:
        tester = Test(opt.weights, opt.batch_size, opt.device, save_dir)
        names = tester.model.names
        tracker = Tracking()

    label_file = None
    label_writer = None
    if opt.label:
        label_file = open(os.path.join(current, "ItalianSigns", 'labels', 'ItalianSigns.csv'), 'a')
        label_writer = csv.writer(label_file)

    # Main infinite loop
    while True:
        frame = camera.get_frame() 

        if camera.available():
            original = frame.copy()

            if time.monotonic() - start_fps > 1:
                fps = camera.get_fps()
                start_fps = time.monotonic()

            if PRESSED_KEY == 'q': # QUIT
                if label_file is not None:
                    label_file.close()
                listener.stop()
                listener.join()
                print("closing!")
                break

            elif PRESSED_KEY == 'r': # REGISTER/STOP RECORDING
                if not RECORDING:
                    print("recording started...")
                    camera.register(os.path.join(current, "Recordings", "{}__{}.mp4".format(opt.filename, datetime.now().strftime("%d_%m_%Y__%H_%M_%S"))))
                    RECORDING = True
                else:
                    camera.stop_recording()
                    print("recording stopped!")
                    RECORDING = False

            elif PRESSED_KEY == 'g' and not RECORDING: # CHANGE GAIN
                gain = int(input("please insert the gain: "))
                camera.set_gain(gain)

            elif PRESSED_KEY == 'e' and not RECORDING: # CHANGE EXPOSURE
                exp = int(input("please insert the exposure: "))
                camera.set_exposure(exp)

            elif PRESSED_KEY == 's' and not RECORDING: # SAVE CURRENT FRAME
                path = os.path.join(current, 'Calibration', 'frame_{}.jpg'.format(datetime.now().strftime("%d_%m_%Y__%H_%M_%S")))
                camera.save_frame(path)

                print("saved frame {} ".format(path))

            elif PRESSED_KEY == 'c' and not RECORDING: # CALIBRATE CAMERA
                print("Calibration in process, please wait...\n")
                cv2.destroyAllWindows()
                geometry = Geometry(os.path.join(current, 'Calibration'))
                calibrated, mtx, dist, rvecs, tvecs = geometry.get_calibration()
                camera.calibrate(calibrated, mtx, dist, rvecs, tvecs)

            elif PRESSED_KEY == 'i': # SHOW MEAN VALUE OF CURRENT FRAME
                print("Frame AVG value: {}".format(frame.mean(axis=(0, 1, 2))))

            elif PRESSED_KEY == 'b': # BLUR FRAME
                BLUR = not BLUR
                print("blur: {}".format(BLUR))

            elif PRESSED_KEY == 't' and not RECORDING: # APPLY TRANSFORMS TO FRAME
                TRANSFORMS = not TRANSFORMS
                print("transform: {}".format(TRANSFORMS))

            elif PRESSED_KEY == 'f' and not RECORDING: # SHOW CHESSBOARD
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

            if not opt.jetson:
                # Object Recognition
                img, _ = preprocessor.Transform_base(frame)
                img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(dim=0)
                out, train_out = tester.predict(img)
                out = non_max_suppression(out, conf_thres=opt.conf_thres, iou_thres=opt.iou_thres, multi_label=True)

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
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 0, 255), 2)
                            cv2.circle(frame, (x + (w//2), y + (h//2)), 4, (40, 55, 255), 4)
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
            found, c, s, u = sd.detect(frame, h, w, show_results = False)
            if found and s != 0:
                circles, speed, updates = c, s, u

                if circles is not None:
                    sign_bb = sd.extract_bb(circles, h, w)
                    frame = an.draw_bb(frame, sign_bb)

                    if opt.label:
                        fname = 'frame_{}.jpg'.format(datetime.now().strftime("%d_%m_%Y__%H_%M_%S"))
                        fpath = os.path.join(current, "ItalianSigns" , 'images', fname)
                        if not os.path.isfile(fpath):
                            saved = cv2.imwrite(fpath, original)
                            if saved:
                                sign_label = [fname, sign_bb[0][0], sign_bb[0][1], sign_bb[1][0], sign_bb[1][1], speed, 1]
                                label_writer.writerow(sign_label)

                if opt.save_sign:    
                    path = os.path.join(current, 'signs', 'sign_{}.jpg'.format(datetime.now().strftime("%d_%m_%Y__%H_%M_%S")))
                    cv2.imwrite(path, frame)
                
            an.write(frame, speed, updates)
            cv2.putText(frame, str(fps) + " fps", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("frame", frame)

    camera.stop()
    cv2.destroyAllWindows()
    print("closed")
