import argparse
import os
import sys
import cv2
import time
import torch
import numpy as np

from Models.YOLOv7 import detect as yolo
from Models.YOLOv7.utils.general import non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from Models.YOLOv7.utils.datasets import LoadStreams, LoadImages
from Models.YOLOv7.utils.plots import plot_one_box
from Models.YOLOv7.utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from traffic import lane_assistant
from traffic import traffic_video as traffic
from Camera import camera
from Models.YOLOv7 import yolo_test
from pathlib import Path
from numpy import random

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

#Results is the path in which you want to save the processed video if you enable
#save_video in the main(), the name of the file by default is the same as the input video
RESULTS = Path(os.path.dirname(os.path.abspath(__file__))+'/results')

ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__))) # This is your Project Root
RESULTS_DIR = ROOT_DIR / 'detected_circles'
SIGNS_DIR = ROOT_DIR / 'signs'
TEMPLATES_DIR = ROOT_DIR / 'templates_solo_bordo'

def main(opt):

    assert opt.dataset is not None or opt.camera is not None or opt.test is not None, 'specify if you want to use bdd100k dataset or the camera or do a test'

    if opt.test:

        filename = opt.test_data

        sd = traffic.Sign_Detector()

        # image
        if not opt.test_video:

            original = cv2.imread(filename)

            cv2.imwrite(current + 'frame.jpg', original)

            #detection
            with torch.no_grad():
                detect_yolo = yolo.Detect(weights=current + r'/Models/YOLOv7/50EPOCHE.pt')
                frame = detect_yolo.static_detect(source=current + 'frame.jpg', save_path=current, view_img=True)

                #frame = cv2.resize(frame, (1280, 720))

            if original.size == 0:
                exit(0)

            height, width, _ = original.shape

            ld = lane_assistant.LaneDetector(width, height)
            lines = ld.detect(original, bilateral=True)
            danger = ld.is_danger(lines=lines)
            frame = ld.draw_lines(frame, lines, ld.choose_colors(danger))

            h_perc = 5
            w_perc = 50

            h = (height * h_perc) // 100
            w = (width * w_perc) // 100

            an = traffic.Annotator(width, height)
            an.org = (20, 50)
            found, circles, speed, updates = sd.detect(original, h_perc, w_perc, show_results=False)
            if found:
                an.write(frame, speed, updates)
                # frame_out = draw_circles(frame, circles, (height, width), (height, width), (h, w))
                frame = an.draw_bb(frame, sd.extract_bb(circles, h, w))

            cv2.imshow("frame", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        #video
        else:

            cap = cv2.VideoCapture(filename)
            # dims = (720, 1280)
            dims = (1280, 720)
            # dims = (450, 800)
            if not cap.isOpened():
                print("Cannot open camera")
                exit()
            i = 0
            start_time = time.time()
            ret, frame = cap.read()
            frame = cv2.flip(cv2.flip(frame, 0), 1)
            frame = cv2.resize(frame, dims)
            ld = lane_assistant.LaneDetector(frame.shape[1], frame.shape[0])

            if not opt.save_video:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        print("Can't receive frame (stream end?). Exiting ...")
                        break

                    frame = cv2.resize(frame, dims)

                    original = frame.copy()
                    cv2.imwrite(current + 'frame.jpg', original)

                    # detection

                    detect_yolo = yolo.Detect(weights=current + r'/Models/YOLOv7/50EPOCHE.pt')
                    frame = detect_yolo.static_detect(source=current + 'frame.jpg', save_path=current, view_img=True)

                    frame = cv2.resize(frame, (1280, 720))

                    height, width, _ = original.shape

                    h_perc = 5
                    w_perc = 50

                    h = (height * h_perc) // 100
                    w = (width * w_perc) // 100

                    an = traffic.Annotator(width, height)
                    an.org = (20, 50)
                    found, circles, speed, updates = sd.detect(original, h_perc, w_perc, show_results=False)
                    if found:
                        an.write(frame, speed, updates)
                        frame = an.draw_bb(frame, sd.extract_bb(circles, h, w))



                    # Only one of the following 2 lines must be commented:
                    lines = ld.detect(original, bilateral=opt.bilateral)  # show real frame
                    danger = ld.is_danger(lines=lines)


                    frame_out = ld.draw_lines(frame, lines, ld.choose_colors(danger))
                    cv2.imshow('frame', cv2.resize(frame_out, (1280, 720)))
                    if cv2.waitKey(1) == ord('q'):
                        break
                        # cv.waitKey(0)
                    i += 1
                    # When everything done, release the capture
                end_time = time.time()
                cap.release()
                cv2.destroyAllWindows()
                delta_time = round(end_time - start_time, 3)
                fps = round(i / delta_time, 3)
                print(f"Processed frames: {i}, total time: {delta_time} seconds, fps: {fps}.")
            else:
                if not os.path.isdir(str(RESULTS)):
                    os.mkdir(str(RESULTS))
                index = opt.file.find('.')
                file = opt.file[0:index]
                ext = '.avi'
                results_dir = str(RESULTS)
                if os.path.isfile(results_dir + '\\' + file+ext):
                    for j in range(1, 100):
                        if not os.path.isfile(results_dir + '\\' + file + '(' + str(j) + ')' + ext):
                            file = file + '(' + str(j) + ')'
                            break
                out = cv2.VideoWriter(str(RESULTS / (file +'.avi')), cv2.VideoWriter_fourcc(*'XVID'), 30, (1280, 720))
                while True:

                    ret, frame = cap.read()
                    if not ret:
                        print("Can't receive frame (stream end?). Exiting ...")
                        break
                    frame = cv2.resize(frame, dims)

                    original = frame.copy()
                    # cv2.imwrite(current + 'frame.jpg', original)
                    #
                    # # detection
                    #
                    # detect_yolo = yolo.Detect(weights=current + r'/Models/YOLOv7/50EPOCHE.pt')
                    # frame = detect_yolo.static_detect(source=current + 'frame.jpg', save_path=current,
                    #                                       view_img=True)

                    frame = cv2.resize(frame, (1280, 720))

                    height, width, _ = original.shape

                    h_perc = 5
                    w_perc = 50

                    h = (height * h_perc) // 100
                    w = (width * w_perc) // 100

                    an = traffic.Annotator(width, height)
                    an.org = (20, 50)
                    found, circles, speed, updates = sd.detect(original, h_perc, w_perc, show_results=False)
                    if found:
                        an.write(frame, speed, updates)
                        frame = an.draw_bb(frame, sd.extract_bb(circles, h, w))

                    # Only one of the following 2 lines must be commented:
                    lines = ld.detect(original, bilateral=opt.bilateral)  # show real frame
                    danger = ld.is_danger(lines=lines)

                    frame_out = ld.draw_lines(frame, lines, ld.choose_colors(danger))
                    cv2.imshow('frame', cv2.resize(frame_out, (1280, 720)))
                    if cv2.waitKey(1) == ord('q'):
                        break
                        # cv.waitKey(0)
                    i += 1

                # When everything done, release the capture
                end_time = time.time()
                cap.release()
                out.release()
                cv2.destroyAllWindows()
                delta_time = round(end_time - start_time, 3)
                fps = round(i / delta_time, 3)
                print(f"Processed frames: {i}, total time: {delta_time} seconds, fps: {fps}.")

    else:

        if opt.dataset:

            yolo_test.main(opt)

        if opt.camera:

            camera.main(opt)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='main.py')

    # input choice
    parser.add_argument('-d', '--dataset', action='store_true', help='the source is bdd100k')
    parser.add_argument('-c', '--camera', action='store_true', help='the source is the camera')

    # test case
    parser.add_argument('-t', '--test', action='store_true', default=False, help='Enable for doing test on image or video')
    parser.add_argument('-tv', '--test-video', action='store_true', default=False, help='Enable for use video like test data test')
    parser.add_argument('-sv', '--save-video', action='store_true', default=False, help='Enable if you want save the test video')
    parser.add_argument('-td', '--test-data', type=str, default=os.path.join(ROOT_DIR, 'traffic/photos', '0.jpg'), help='*.test_data path')

    # required parameters
    parser.add_argument('-b', '--batch-size', type=int, default=1, help='YOLOv7 batch-size')
    parser.add_argument('-ct', '--conf-thres', type=float, default=0.001, help='YOLOv7 conf threshold')
    parser.add_argument('-dev', '--device', type=str, default='0', help='cuda device(s)')
    parser.add_argument('-it', '--iou-thres', type=float, default=0.65, help='YOLOv7 iou threshold')
    parser.add_argument('-n', '--name', type=str, default='test_dir', help='YOLOv7 result test directory name')
    parser.add_argument('-p', '--project', type=str, default=os.path.join(parent, 'Models', 'YOLOv7', 'runs', 'test') , help='YOLOv7 project save directory')
    parser.add_argument('-sh', '--save-hybrid', action='store_true', default=False, help='YOLOv7 save hybrid')
    parser.add_argument('-st', '--save-txt', action='store_true', default=False, help='YOLOv7 save txt')
    parser.add_argument('-w', '--weights', type=str, default=os.path.join(parent, 'Models', 'YOLOv7', 'last.pt'),
                        help='YOLOv7 weights')

    # yolo_test parameters
    parser.add_argument('-a', '--augment', action='store_true', help='augmented inference')
    parser.add_argument('-comp', '--compute-loss', default=None, help='')
    parser.add_argument('-dt', '--data', type=str, default=os.path.join(current, 'data', 'bdd100k'), help='*.data path')
    parser.add_argument('-ok', '--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('-hy', '--hyp', type=str, default=os.path.join(current, 'data', 'hyp.scratch.p5.yaml'), help='')
    parser.add_argument('-iw', '--image-weights', type=bool, default=False, help='')
    parser.add_argument('-is', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('-nt', '--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('-plt', '--plots', action='store_true', help='')
    parser.add_argument('-sc', '--save-conf', default=True, action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('-sj', '--save-json', default=True, action='store_true', help='save a compatible JSON results file')
    parser.add_argument('-sng', '--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('-str', '--stride', type=int, default=32, help='')
    parser.add_argument('-ts', '--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('-v', '--verbose', default=True, action='store_true', help='report mAP by class')
    parser.add_argument('-wr', '--workers', type=int, default=6, help='')

    # camera parameters
    parser.add_argument('-cal', '--calibrate', action='store_true', default=False, help='true if you want to calibrate the camera')
    parser.add_argument('-cd', '--camera-device', type=int, default=0, help='Camera device ID')
    parser.add_argument('-f', '--filename', type=str, default='out', help='filename for recordings')
    parser.add_argument('-j', '--jetson', action='store_true', default=False, help='true if you are using the Nvidia Jetson Nano')
    parser.add_argument('-lb', '--label', action='store_true', default=False, help='true if you want to save labelled signs')
    parser.add_argument('-r', '--resolution', type=tuple, default=(1280, 720), help='camera resolution')
    parser.add_argument('-rt', '--rotate', action='store_true', default=False, help='rotate frame for e-con camera')
    parser.add_argument('-s', '--save-sign', action='store_true', default=False, help='save frames which contain signs')
    parser.add_argument('-ln', '--lane-assistant', action='store_false', default=True, help='Enable the lane assistant')

    # lane_assistant parameters
    parser.add_argument('-rtm', '--real-time', action='store_true', default=False,
                        help='Enable if you want to process a video in real-time with the IP Webcam App')
    parser.add_argument('-ps', '--post', action='store_true', default=False,
                        help='Enable if you want to see the video with the mask applied, in order to see what the program see (useful if you need to set the mask properly)')
    parser.add_argument('-bl', '--bilateral', action='store_false', default=True,
                        help='Enable if you want to use the bilateral filter, if False the the standard Gaussian Blur is performed (but bilateral works better)')
    # parser.add_argument('-sv', '--save-video', action='store_true', default=False,help='Enable if you want to save the video in the directory RESULTS, with the same file name as the input (default extension is ".avi")')
    parser.add_argument('-fl', '--file', type=str, help='Name of the file')




    opt = parser.parse_args()

    print(opt)

    main(opt)




