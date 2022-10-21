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

from Camera.RTCamera import RTCamera
from Geometry import Geometry
from Preprocessing import Preprocessing
from Distance import Distance
from traffic.traffic_video import Sign_Detector, Annotator
from Tracking import Tracking
from traffic.lane_assistant import LaneDetector

from collections import Counter

from Models.YOLOv7.yolo_test import Test
from Models.YOLOv7.utils.general import increment_path, non_max_suppression, scale_coords, xyxy2xywh

def resolution(s):
    try:
        y, x = map(int, s.split(','))
        return (y, x)
    except:
        raise argparse.ArgumentTypeError("Resolution must be W, H")

def main(opt):
    RECORDING = False
    BLUR = False
    TRANSFORMS = False
    CHESSBOARD = False
    ROTATION = None

    opt.save_txt |= opt.save_hybrid
    opt.camera_device = int(opt.camera_device) if opt.camera_device.isnumeric() else opt.camera_device

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

    if not os.path.isdir(os.path.join(current, "ItalianSigns", 'images')):
        os.makedirs(os.path.join(current, "ItalianSigns", 'images'))

    if not os.path.isdir(os.path.join(current, "ItalianSigns", 'rawimages')):
        os.makedirs(os.path.join(current, "ItalianSigns", 'rawimages'))

    if not os.path.isdir(os.path.join(current, "ItalianSigns", 'labels')):
        os.makedirs(os.path.join(current, "ItalianSigns", 'labels'))

    if not os.path.isfile(os.path.join(current, "ItalianSigns", 'labels', 'RawSigns.csv')):
        f = open(os.path.join(current, "ItalianSigns", 'labels', 'RawSigns.csv'), 'w')
        writer = csv.writer(f)
        writer.writerow(
            ["filename", "x top left", "y top left", "x bottom right", "y bottom right", "speed limit", "valid"])
        f.close()

    print("Running with resolution {}x{}".format(opt.resolution[0], opt.resolution[1]))
    camera = None
    if opt.source == "live":
        camera = RTCamera(opt.camera_device, fps=opt.fps, resolution=opt.resolution, cuda=False, auto_exposure=False, rotation=ROTATION, exposure=opt.exposure)
        camera.start()

    start_fps = time.monotonic()
    fps = 0

    sd = Sign_Detector()
    an = Annotator(*opt.resolution)
    an.org = (20, 50)
    circles = None
    speed = 0
    updates = 0

    if opt.calibrate and opt.source == "live":
        geometry = Geometry(os.path.join(current, 'Calibration'), (9, 6))
        calibrated, mtx, dist, rvecs, tvecs = geometry.get_calibration()
        camera.calibrate(calibrated, mtx, dist, rvecs, tvecs)

    tester = None
    names = None
    tracker = None

    if not opt.jetson:
        tester = Test(opt.weights, opt.batch_size, opt.device, save_dir, imgsz=opt.resolution[0])
        names = tester.model.names
        tracker = Tracking()

    label_file = None
    label_writer = None
    if opt.label:
        label_file = open(os.path.join(current, "ItalianSigns", 'labels', 'RawSigns.csv'), 'a')
        label_writer = csv.writer(label_file)

    # Main infinite loop
    preprocessor = Preprocessing(opt.resolution)
    cap = None

    if opt.source == "video":
        cap = cv2.VideoCapture(opt.path)

    while True:
        frame = None
        available = False

        if opt.source == "live":
            available = camera.available()
        elif opt.source == "video":
            available = cap.isOpened()
        elif opt.source == "image":
            available = True

        if available:
            if opt.source == "live":
                frame = camera.get_frame()
            elif opt.source == "video":
                ret, frame = cap.read()
            else:
                frame = cv2.imread(opt.path)

            if frame is None:
                print("empty None, stopping")
                break

            if frame.size == 0:
                print("frame is empty, stopping")
                break

            frame = cv2.resize(frame, opt.resolution)
            original = frame.copy()

            if time.monotonic() - start_fps > 1:
                fps = 0
                if opt.source == "live":
                    fps = camera.get_fps()
                elif opt.source == "video":
                    fps = round(cap.get(cv2.CAP_PROP_FPS))

                start_fps = time.monotonic()

            key = cv2.waitKey(1)

            if key == ord('q'):  # QUIT
                if label_file is not None:
                    label_file.close()
                print("closing!")
                break

            if opt.source == "live":
                if key == ord('r'):  # REGISTER/STOP RECORDING
                    if not RECORDING:
                        print("recording started...")
                        camera.register(os.path.join(current, "Recordings", "{}__{}.mp4".format(opt.filename, datetime.now().strftime("%d_%m_%Y__%H_%M_%S"))))
                        RECORDING = True
                    else:
                        camera.stop_recording()
                        print("recording stopped!")
                        RECORDING = False

                elif key == ord('g') and not RECORDING:  # CHANGE GAMMA
                    gamma = float(input("please insert gamma value: "))
                    camera.set_gamma(gamma)

                elif key == ord('a') and not RECORDING:  # CHANGE ALPHA AND BETA
                    clip = float(input("insert clip percentage: "))
                    camera.calc_bc(clip)

                elif key == ('e') and not RECORDING:  # CHANGE EXPOSURE
                    try:
                        exp = float(input("please insert the exposure: "))
                        camera.set_exposure(exp)
                    except:
                        print("Error during exposure read")

                elif key == ord('s') and not RECORDING:  # SAVE CURRENT FRAME
                    path = os.path.join(current, 'Calibration', 'frame_{}.jpg'.format(datetime.now().strftime("%d_%m_%Y__%H_%M_%S")))
                    camera.save_frame(path)

                    print("saved frame {} ".format(path))

                elif key == ord('c') and not RECORDING:  # CALIBRATE CAMERA
                    print("Calibration in process, please wait...\n")
                    cv2.destroyAllWindows()
                    geometry = Geometry(os.path.join(current, 'Calibration'), (9, 6))
                    calibrated, mtx, dist, rvecs, tvecs = geometry.get_calibration()
                    camera.calibrate(calibrated, mtx, dist, rvecs, tvecs)

                elif key == ord('i'):  # SHOW MEAN VALUE OF CURRENT FRAME
                    print("Frame AVG value: {}".format(frame.mean(axis=(0, 1, 2))))

                elif key == ord('b'):  # BLUR FRAME
                    BLUR = not BLUR
                    print("blur: {}".format(BLUR))

                elif key == ord('t') and not RECORDING:  # APPLY TRANSFORMS TO FRAME
                    TRANSFORMS = not TRANSFORMS
                    print("transform: {}".format(TRANSFORMS))

                elif key == ord('f') and not RECORDING:  # SHOW CHESSBOARD
                    CHESSBOARD = not CHESSBOARD
                    print("Chessboard: {}".format(CHESSBOARD))
                    cv2.destroyAllWindows()

                if BLUR:
                    frame = preprocessor.GaussianBlur(frame, 1)

                if TRANSFORMS:
                    (frame, _) = preprocessor.Transform_base(frame)

                if CHESSBOARD:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    ret, corners = cv2.findChessboardCorners(gray, (9, 6), cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK)
                    if ret:
                        cv2.drawChessboardCorners(frame, (9, 6), corners, ret)

            if not opt.jetson:
                # Object Recognition
                img, _ = preprocessor.Transform_base(original.copy())
                img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(dim=0)
                out, train_out = tester.predict(img)
                out = non_max_suppression(out, conf_thres=opt.conf_thres, iou_thres=opt.iou_thres, multi_label=True)

                pred = out[0] # because we infer only on one image

                ratio = ((1, 1), (0, 0))
                scale_coords(img.shape[2:], pred[:, :4], opt.resolution, ratio)  # native-space pred
                pred[:, :4] = xyxy2xywh(pred[:, :4])
                pred[:, 0] -= pred[:, 2] / 2
                pred[:, 1] -= pred[:, 3] / 2

                sub = 0
                if opt.resolution[0] > opt.resolution[1]:
                    sub = (opt.resolution[0] - opt.resolution[1]) / 2
                    pred[:, 1] -= sub
                elif opt.resolution[0] < opt.resolution[1]:
                    sub = (opt.resolution[1] - opt.resolution[0]) / 2
                    pred[:, 0] -= sub

                obj_counter = {'truck': 0, 'other person': 0, 'motorcycle': 0, 'bus': 0, 'other vehicle': 0, 'rider': 0, 'pedestrian': 0, 'bicycle': 0, \
                        'train': 0, 'car': 0, 'trailer': 0, 'traffic sign': 0, 'traffic light': 0}
                
                for *xywh, conf, cls in pred.tolist():
                    if conf > opt.confidence:
                        obj_counter[names[int(cls)]] += 1
                        xywh = [int(k) for k in xywh]
                        x, y, w, h = xywh

                        if h == 0 or w == 0:
                            continue

                        distance = Distance().get_Distance(xywh)

                        if opt.verbose:
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                            cv2.circle(frame, (x + (w // 2), y + (h // 2)), 4, (40, 55, 255), 2)
                            cv2.putText(frame, "{:.2f} {} {:.2f}".format(conf, names[int(cls)], distance), (x + 5, y + 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 2)

                if opt.verbose:
                    y_pos = frame.shape[0] // 2
                    for el in obj_counter.keys():
                        if obj_counter[el] > 0:
                            cv2.putText(frame, "{}: {}".format(el, obj_counter[el]), (15, y_pos + 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 25, 25), 2)
                            y_pos += 20

                # Tracking
                if opt.track:
                    hsvframe = cv2.cvtColor(original.copy(), cv2.COLOR_RGB2HSV)
                    tracker.zero_objects()

                    for *box, conf, cls in pred.tolist():
                        if names[int(cls)] in ['truck', 'other person', 'motorcycle', 'bus', 'other vehicle', 'rider', 'pedestrian', 'bicycle', 'train', 'car', 'trailer']:   
                            if conf > opt.confidence:
                                box = [int(k) for k in box]
                                x, y, w, h = box

                                if h == 0 or w == 0:
                                    continue

                                id = tracker.update_obj(cls, box)
                                prediction, pts = tracker.track(hsvframe, box, id)
                                if opt.verbose:
                                    cv2.putText(frame, "ID: {}".format(id), (x - 60, y + 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 255), 1)
                                    cv2.putText(frame, "ID: {}".format(id), (int(prediction[0] - (0.5 * w)) + 5, int(prediction[1] - (0.5 * h)) + 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 255), 1)
                                    cv2.rectangle(frame, (int(prediction[0] - (0.5 * w)), int(prediction[1] - (0.5 * h))), (int(prediction[0] + (0.5 * w)), int(prediction[1] + (0.5 * h))), (0, 255, 0), 2)
                    tracker.clear_objects()

            # lane detection
            h = frame.shape[0]
            w = frame.shape[1]
            if opt.lane_assistant:
                ld = LaneDetector(w, h)
                lines = ld.detect(frame, bilateral=True)
                danger = ld.is_danger(lines=lines)
                if opt.verbose:
                    frame = ld.draw_lines(frame, lines, ld.choose_colors(danger))

            # traffic sign detection
            h_perc = 5
            w_perc = 50

            height, width, _ = frame.shape
            h = (h * h_perc) // 100
            w = (w * w_perc) // 100

            if opt.verbose:
                frame = cv2.line(frame, (w, h), (width, h), (255, 0, 0), 2)
                frame = cv2.line(frame, (w, h), (w, height - h), (255, 0, 0), 2)
                frame = cv2.line(frame, (w, height - h), (width, height - h), (255, 0, 0), 2)

            found, c, s, u = sd.detect(original.copy(), h_perc, w_perc, show_results=False)
            if found and s != 0:
                circles, speed, updates = c, s, u

                if circles is not None:
                    sign_bb = sd.extract_bb(circles, h, w)
                    frame = an.draw_bb(frame, sign_bb)

                    if opt.label:
                        fname = 'frame_{}.jpg'.format(datetime.now().strftime("%d_%m_%Y__%H_%M_%S"))
                        fpath = os.path.join(current, "ItalianSigns", 'rawimages', fname)
                        if not os.path.isfile(fpath):
                            saved = cv2.imwrite(fpath, original.copy())
                            if saved:
                                sign_label = [fname, sign_bb[0][0], sign_bb[0][1], sign_bb[1][0], sign_bb[1][1], speed, 1]
                                label_writer.writerow(sign_label)

                if opt.save_sign:
                    path = os.path.join(current, 'signs', 'sign_{}.jpg'.format(datetime.now().strftime("%d_%m_%Y__%H_%M_%S")))
                    cv2.imwrite(path, frame)

            if opt.verbose:
                an.write(frame, speed, updates)
                cv2.putText(frame, str(fps) + " fps", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("frame", frame)

        if opt.source == "image":
            cv2.waitKey(0)
            answer = input("Do you want to save the frame? Y/N ")
            if answer.upper() == 'Y':
                print(cv2.imwrite("{}.jpg".format(datetime.now().strftime("%d_%m_%Y__%H_%M_%S")), frame))
            break

    if opt.source == "live":
        camera.stop()

    cv2.destroyAllWindows()
    print("closed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-b', '--batch-size', type=int, default=1, help='YOLOv7 batch-size')
    parser.add_argument('-c', '--calibrate', action='store_true', default=False, help='true if you want to calibrate the camera')
    parser.add_argument('-cd', '--camera-device', type=str, default='0', help='Camera device ID')
    parser.add_argument('-ct', '--conf-thres', type=float, default=0.001, help='YOLOv7 conf threshold')
    parser.add_argument('-cf', '--confidence', type=float, default=0.25, help='YOLOv7 confidence on prediction threshold')
    parser.add_argument('-d', '--device', type=str, default='0', help='cuda device(s)')
    parser.add_argument('-exp', '--exposure', type=int, default=-5, help='Sets camera exposure')
    parser.add_argument('-it', '--iou-thres', type=float, default=0.65, help='YOLOv7 iou threshold')
    parser.add_argument('-f', '--filename', type=str, default='out', help='filename for recordings')
    parser.add_argument('-fps', '--fps', type=int, default=60, help='Sets camera FPS')
    parser.add_argument('-j', '--jetson', action='store_true', default=False, help='true if you are using the Nvidia Jetson Nano')
    parser.add_argument('-l', '--label', action='store_true', default=False, help='true if you want to save labelled signs')
    parser.add_argument('-ln', '--lane-assistant', action='store_true', default=False, help='true if you want to use lane assistant')
    parser.add_argument('-n', '--name', type=str, default='camera', help='YOLOv7 result test directory name')
    parser.add_argument('-p', '--project', type=str, default=os.path.join(parent, 'Models', 'YOLOv7', 'runs', 'test') , help='YOLOv7 project save directory')
    parser.add_argument('-pt', '--path', type=str, default="", help='path file in case of image or video as source')    
    parser.add_argument('-r', '--resolution', type=tuple, default=(1280, 720), help='camera resolution')
    parser.add_argument('-rt', '--rotate', action='store_true', default=False, help='rotate frame for e-con camera')
    parser.add_argument('-s', '--save-sign', action='store_true', default=False, help='save frames which contain signs')
    parser.add_argument('-sc', '--source', type=str, default="live", help='source: video, image, live')    
    parser.add_argument('-sh', '--save-hybrid', action='store_true', default=False, help='YOLOv7 save hybrid')
    parser.add_argument('-st', '--save-txt', action='store_true', default=False, help='YOLOv7 save txt')
    parser.add_argument('-t', '--track', action='store_true', default=False, help='track objects recognized by YOLOv7')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help='show stuff on the frame')
    parser.add_argument('-w', '--weights', type=str, default=os.path.join(parent, 'Models', 'YOLOv7', '50EPOCHE.pt') , help='YOLOv7 weights')

    opt = parser.parse_args()

    # FOR A QUICK DEBUG
    #opt.path = r"C:\Users\daniel\Documents\GitHub Repositories\ComputerVisionProject\TEST_VIDEO.mp4"
    #opt.source = "video"
    #opt.jetson = True
    #opt.verbose = True
    #opt.track = True
    #opt.confidence = 0.85

    assert opt.source in ["image", "video", "live"], "invalid source"
    if opt.source in ["image", "video"]:
        assert os.path.isfile(opt.path), "invalid source file"

    main(opt)


