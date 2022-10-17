import argparse
import os
import sys
import torch
import cv2
import time
import numpy as np
from numpy import random

from Camera import camera
from pathlib import Path
from Preprocessing import Preprocessing
from Distance import Distance
from Tracking import Tracking

from traffic.traffic_video import Sign_Detector, Annotator
from traffic import lane_assistant

from Models.YOLOv7.yolo_test import Test
from Models.YOLOv7.utils.general import increment_path, non_max_suppression, scale_coords, xyxy2xywh

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

# Results is the path in which you want to save the processed video if you enable
# save_video in the main(), the name of the file by default is the same as the input video
RESULTS = Path(os.path.dirname(os.path.abspath(__file__)) + '/results')

ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))  # This is your Project Root


def resolution(s):
    try:
        y, x = map(int, s.split(','))
        return (y, x)
    except:
        raise argparse.ArgumentTypeError("Resolution must be W, H")


def object_recognition(frame, tester, names, colors, counters):
    preprocessor = Preprocessing(opt.resolution)

    # Object Recognition
    img, _ = preprocessor.Transform_base(frame.copy())
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(dim=0)
    out, train_out = tester.predict(img)
    out = non_max_suppression(out, conf_thres=opt.conf_thres, iou_thres=opt.iou_thres, multi_label=True)

    pred = out[0]  # because we infer only on one image

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

    for *xywh, conf, cls in pred.tolist():
        if conf > opt.confidence:
            xywh = [int(k) for k in xywh]
            x, y, w, h = xywh

            if h == 0 or w == 0:
                continue

            distance = Distance().get_Distance(xywh)

            class_ = names[int(cls)]
            counters[class_] += 1

            label = "{:.2f} {} {:.2f}".format(conf, names[int(cls)], distance)
            # plot_one_box(xywh, frame, label=label, color=colors[int(cls)], line_thickness=2)

            c1, c2 = (int(xywh[0]), int(xywh[1])), (int(xywh[2]), int(xywh[3]))
            # cv2.rectangle(frame, c1, c2, colors[int(cls)], thickness=2, lineType=cv2.LINE_AA)

            cv2.rectangle(frame, (x, y), (x + w, y + h), colors[int(cls)], 2)

            tf = max(2 - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=2 / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(frame, c1, c2, colors[int(cls)], -1, cv2.LINE_AA)  # filled
            cv2.putText(frame, label, (c1[0], c1[1] - 2), 0, 2 / 3, [225, 255, 255], thickness=tf,
                        lineType=cv2.LINE_AA)
            cv2.circle(frame, (x + (w // 2), y + (h // 2)), 4, (40, 55, 255), 2)

    # object counting
    if not opt.remove_object_counting:
        margin = 0
        for i in counters:

            value = counters.get(i)
            if (value != 0):
                label = i + ": " + str(value)
                t_size = cv2.getTextSize(label, 0, fontScale=2 / 3, thickness=1)[0]
                x = 20
                y = 100 + (margin * 20)
                c1, c2 = (x, y), (x+t_size[0], y-t_size[1]-3)
                cv2.rectangle(frame, c1, c2, [0,0,0], -1, cv2.LINE_AA)  # filled
                cv2.putText(frame, label, (c1[0], c1[1] - 2), 0, 2 / 3, [255, 255, 255], thickness=1,lineType=cv2.LINE_AA)

                margin += 1

            counters[i] = 0

    return frame

def signs_detector(frame, original, speed, updates):
    sd = Sign_Detector()
    an = Annotator(*opt.resolution)
    an.org = (20, 50)

    height, width, _ = frame.shape
    h_perc = 5
    w_perc = 50

    h, w, _ = frame.shape
    h = (h * h_perc) // 100
    w = (w * w_perc) // 100

    found, c, s, u = sd.detect(original.copy(), h_perc, w_perc, show_results=False)

    if found and s != 0:
        circles, speed, updates = c, s, u

        if circles is not None:
            sign_bb = sd.extract_bb(circles, h, w)
            frame = an.draw_bb(frame, sign_bb)

    an.write(frame, speed, updates)

    return frame, speed, updates


def lane_detector(frame, original):
    ld = lane_assistant.LaneDetector(original.shape[1], original.shape[0])
    lines = ld.detect(original, bilateral=opt.bilateral)  # show real frame
    danger = ld.is_danger(lines=lines)
    frame_out = ld.draw_lines(frame, lines, ld.choose_colors(danger))

    return frame_out


def main(opt):
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=False))  # increment run
    (save_dir / 'labels' if opt.save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    tester = Test(opt.weights, opt.batch_size, opt.device, save_dir)

    names = tester.model.names

    counters = dict.fromkeys(names, 0)

    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    assert opt.camera is not None or opt.test is not None, 'specify if you want use camera or do a test'

    if opt.test:

        speed = 0
        updates = 0

        filename = opt.test_data

        # image
        if not opt.test_video:

            print('image')

            frame = cv2.imread(filename)

            original = frame.copy()
            original = cv2.resize(original, opt.resolution)

            # Object Recognition
            print('object recognition')
            if not opt.remove_yolo:
                frame = object_recognition(original, tester, names, colors, counters)

            #  signs detection
            print('signs detection')
            frame = cv2.resize(frame, opt.resolution)
            if not opt.remove_signs_detector:
                frame, speed, updates = signs_detector(frame, original, speed, updates)

            # lane detection
            print('lane detection')
            if not opt.remove_lane_assistant:
                frame = lane_detector(frame, original)

            cv2.imwrite("frame.jpg", frame)

        # video
        else:

            print('video')

            cap = cv2.VideoCapture(filename)
            dims = opt.resolution
            if not cap.isOpened():
                print("Cannot open camera")
                exit()
            i = 0
            start_time = time.time()
            ret, frame = cap.read()
            frame = cv2.flip(cv2.flip(frame, 0), 1)
            frame = cv2.resize(frame, dims)

            if not opt.save_video:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        print("Can't receive frame (stream end?). Exiting ...")
                        break

                    frame = cv2.resize(frame, dims)

                    original = frame.copy()

                    # Object Recognition
                    if not opt.remove_yolo:
                        #print('object recognition')
                        frame = object_recognition(original, tester, names, colors, counters)

                    # signs detection
                    frame = cv2.resize(frame, opt.resolution)
                    if not opt.remove_signs_detector:
                        #print('signs detection')
                        frame, speed, updates = signs_detector(frame, original, speed, updates)

                    # lane assistant
                    if not opt.remove_lane_assistant:
                        #print('lane assistant')
                        frame = lane_detector(frame, original)

                    cv2.imshow('frame', cv2.resize(frame, opt.resolution))

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

                print('saving video')

                if not os.path.isdir(str(RESULTS)):
                    os.mkdir(str(RESULTS))
                index = opt.file.find('.')
                file = opt.file[0:index]
                ext = '.avi'
                results_dir = str(RESULTS)
                if os.path.isfile(results_dir + '\\' + file + ext):
                    for j in range(1, 100):
                        if not os.path.isfile(results_dir + '\\' + file + '(' + str(j) + ')' + ext):
                            file = file + '(' + str(j) + ')'
                            break
                out_video = cv2.VideoWriter(str(RESULTS / (file + '.avi')), cv2.VideoWriter_fourcc(*'XVID'), 30,
                                            opt.resolution)
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        print("Can't receive frame (stream end?). Exiting ...")
                        break

                    frame = cv2.resize(frame, dims)

                    original = frame.copy()

                    # Object Recognition
                    if not opt.remove_yolo:
                        #print('object recognition')
                        frame = object_recognition(original, tester, names, colors, counters)

                    # signs detection
                    frame = cv2.resize(frame, opt.resolution)
                    if not opt.remove_signs_detector:
                        #print('signs detector')
                        frame, speed, updates = signs_detector(frame, original, speed, updates)

                    # lane assistant
                    if not opt.remove_lane_assistant:
                        #print('lane assistant')
                        frame = lane_detector(frame, original)

                    out_video.write(frame)
                    print('frame: ' + str(i+1))
                    i += 1

                # When everything done, release the capture
                end_time = time.time()
                cap.release()
                out_video.release()
                cv2.destroyAllWindows()
                delta_time = round(end_time - start_time, 3)
                fps = round(i / delta_time, 3)
                print(f"Processed frames: {i}, total time: {delta_time} seconds, fps: {fps}.")

    elif opt.camera:

        camera.main(opt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='main.py')

    # input choice
    parser.add_argument('-t', '--test', action='store_true', default=False,
                        help='Enable for doing test on image or video')
    parser.add_argument('-c', '--camera', action='store_true', help='the source is the camera')

    # test case
    parser.add_argument('-tv', '--test-video', action='store_true', default=False,
                        help='Enable for use video like test data test')
    parser.add_argument('-sv', '--save-video', action='store_true', default=False,
                        help='Enable if you want save the test video')
    parser.add_argument('-td', '--test-data', type=str, default=os.path.join(ROOT_DIR, 'traffic/photos', '4.jpg'),
                        help='*.test_data path')
    parser.add_argument('-fl', '--file', type=str, help='Name of the file')
    parser.add_argument('-rml', '--remove-lane-assistant', action='store_true', default=False,
                        help='Disable the lane assistant')
    parser.add_argument('-rms', '--remove-signs-detector', action='store_true', default=False,
                        help='Disable the signs detector')
    parser.add_argument('-rmy', '--remove-yolo', action='store_true', default=False,
                        help='Disable the object recognition with Yolo')
    parser.add_argument('-rmc', '--remove-object-counting', action='store_true', default=False,
                        help='Disable the object counting')

    # camera parameters
    parser.add_argument('-b', '--batch-size', type=int, default=1, help='YOLOv7 batch-size')
    parser.add_argument('-cl', '--calibrate', action='store_true', default=False,
                        help='true if you want to calibrate the camera')
    parser.add_argument('-cd', '--camera-device', type=str, default='0', help='Camera device ID')
    parser.add_argument('-ct', '--conf-thres', type=float, default=0.001, help='YOLOv7 conf threshold')
    parser.add_argument('-cf', '--confidence', type=float, default=0.25,
                        help='YOLOv7 confidence on prediction threshold')
    parser.add_argument('-d', '--device', type=str, default='0', help='cuda device(s)')
    parser.add_argument('-exp', '--exposure', type=int, default=-5, help='Sets camera exposure')
    parser.add_argument('-it', '--iou-thres', type=float, default=0.65, help='YOLOv7 iou threshold')
    parser.add_argument('-f', '--filename', type=str, default='out', help='filename for recordings')
    parser.add_argument('-fps', '--fps', type=int, default=60, help='Sets camera FPS')
    parser.add_argument('-j', '--jetson', action='store_true', default=False,
                        help='true if you are using the Nvidia Jetson Nano')
    parser.add_argument('-l', '--label', action='store_true', default=False,
                        help='true if you want to save labelled signs')
    parser.add_argument('-n', '--name', type=str, default='camera', help='YOLOv7 result test directory name')
    parser.add_argument('-p', '--project', type=str, default=os.path.join(parent, 'Models', 'YOLOv7', 'runs', 'test'),
                        help='YOLOv7 project save directory')
    parser.add_argument('-pt', '--path', type=str, default="", help='path file in case of image or video as source')
    parser.add_argument('-rt', '--rotate', action='store_true', default=False, help='rotate frame for e-con camera')
    parser.add_argument('-s', '--save-sign', action='store_true', default=False, help='save frames which contain signs')
    parser.add_argument('-sc', '--source', type=str, default="live", help='source: video, image, live')
    parser.add_argument('-sh', '--save-hybrid', action='store_true', default=False, help='YOLOv7 save hybrid')
    parser.add_argument('-st', '--save-txt', action='store_true', default=False, help='YOLOv7 save txt')
    parser.add_argument('-tr', '--track', action='store_true', default=False, help='track objects recognized by YOLOv7')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help='show stuff on the frame')
    parser.add_argument('-w', '--weights', type=str, default=os.path.join(parent, 'Models', 'YOLOv7', '50EPOCHE.pt'),
                        help='YOLOv7 weights')

    # resolution
    parser.add_argument('-r', '--resolution', type=tuple, default=(1280, 720), help='define the resolution')

    # lane_assistant parameters
    parser.add_argument('-bl', '--bilateral', action='store_false', default=True,
                        help='Enable if you want to use the bilateral filter, if False the the standard Gaussian Blur is performed (but bilateral works better)')

    opt = parser.parse_args()

    print(opt)

    main(opt)




