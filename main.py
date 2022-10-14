import argparse
import os
import sys
import torch
import cv2
import time

from Camera import camera
from pathlib import Path
from Preprocessing import Preprocessing
from Distance import Distance

from traffic.traffic_video import Sign_Detector, Annotator
from traffic import lane_assistant
from traffic import traffic_video as traffic

from Models.YOLOv7.yolo_test import Test
from Models.YOLOv7.utils.general import increment_path, non_max_suppression, scale_coords, xyxy2xywh

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

# Results is the path in which you want to save the processed video if you enable
# save_video in the main(), the name of the file by default is the same as the input video
RESULTS = Path(os.path.dirname(os.path.abspath(__file__)) + '/results')

ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))  # This is your Project Root
RESULTS_DIR = ROOT_DIR / 'detected_circles'
SIGNS_DIR = ROOT_DIR / 'signs'
TEMPLATES_DIR = ROOT_DIR / 'templates_solo_bordo'


def object_recognition(frame, save_dir):
    preprocessor = Preprocessing((640, 640))
    frame = cv2.resize(frame, (640, 640))

    tester = Test(opt.weights, opt.batch_size, opt.device, save_dir)

    names = tester.model.names

    img, _ = preprocessor.Transform_base(frame.copy())
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
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1)  # xywh
                xywh = [int(x) for x in xywh]
                x, y, w, h = xywh
                x -= w // 2
                y -= h // 2

                detections.append((cls, xywh))
                distance = Distance().get_Distance(xywh)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 0, 255), 2)
                cv2.circle(frame, (x + (w // 2), y + (h // 2)), 4, (40, 55, 255), 4)
                cv2.putText(frame, "{:.2f} {} {:.2f}".format(conf, names[int(cls)], distance), (x + 5, y + 20),
                            cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 0, 255), 1)

    return frame


def detector_signs(frame, original, speed, updates):
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
    # frame = cv2.resize(frame, (1280, 720))

    return frame, speed, updates


def detector_lane(frame, original):
    ld = lane_assistant.LaneDetector(original.shape[1], original.shape[0])
    lines = ld.detect(original, bilateral=opt.bilateral)  # show real frame
    danger = ld.is_danger(lines=lines)
    frame_out = ld.draw_lines(frame, lines, ld.choose_colors(danger))

    return frame_out


def main(opt):

    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=False))  # increment run
    (save_dir / 'labels' if opt.save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

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
            original = cv2.resize(original, (1280, 720))

            # Object Recognition
            print('object recognition')
            frame = object_recognition(frame, save_dir)

            # detection signs
            print('detection signs')
            frame = cv2.resize(frame, (1280, 720))
            frame, speed, updates = detector_signs(frame, original, speed, updates)

            # lane detection
            print('lane detection')
            frame_out = detector_lane(frame, original)

            cv2.imwrite("frame.jpg", frame_out)

        # video
        else:

            cap = cv2.VideoCapture(filename)
            dims = (1280, 720)
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
                    frame = object_recognition(frame, save_dir)
                    frame = cv2.resize(frame, (1280, 720))

                    # detection signs
                    frame, speed, updates = detector_signs(frame, original, speed, updates)

                    # lane assistant
                    frame_out = detector_lane(frame, original)
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
                                            (1280, 720))
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        print("Can't receive frame (stream end?). Exiting ...")
                        break

                    frame = cv2.resize(frame, dims)

                    original = frame.copy()

                    # Object Recognition
                    frame = object_recognition(frame, save_dir)
                    frame = cv2.resize(frame, (1280, 720))

                    # detection signs
                    frame, speed, updates = detector_signs(frame, original, speed, updates)

                    # lane assistant
                    frame_out = detector_lane(frame, original)

                    out_video.write(frame_out)
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
    parser.add_argument('-d', '--dataset', action='store_true', help='the source is bdd100k')
    parser.add_argument('-c', '--camera', action='store_true', help='the source is the camera')

    # test case
    parser.add_argument('-t', '--test', action='store_true', default=False,
                        help='Enable for doing test on image or video')
    parser.add_argument('-tv', '--test-video', action='store_true', default=False,
                        help='Enable for use video like test data test')
    parser.add_argument('-sv', '--save-video', action='store_true', default=False,
                        help='Enable if you want save the test video')
    parser.add_argument('-td', '--test-data', type=str, default=os.path.join(ROOT_DIR, 'traffic/photos', '4.jpg'),
                        help='*.test_data path')
    parser.add_argument('-fl', '--file', type=str, help='Name of the file')

    # required parameters
    parser.add_argument('-b', '--batch-size', type=int, default=1, help='YOLOv7 batch-size')
    parser.add_argument('-ct', '--conf-thres', type=float, default=0.001, help='YOLOv7 conf threshold')
    parser.add_argument('-dev', '--device', type=str, default='0', help='cuda device(s)')
    parser.add_argument('-it', '--iou-thres', type=float, default=0.65, help='YOLOv7 iou threshold')
    parser.add_argument('-n', '--name', type=str, default='test_dir', help='YOLOv7 result test directory name')
    parser.add_argument('-p', '--project', type=str, default=os.path.join(current, 'Models', 'YOLOv7', 'runs', 'test'),
                        help='YOLOv7 project save directory')
    parser.add_argument('-sh', '--save-hybrid', action='store_true', default=False, help='YOLOv7 save hybrid')
    parser.add_argument('-st', '--save-txt', action='store_true', default=False, help='YOLOv7 save txt')
    parser.add_argument('-w', '--weights', type=str, default=os.path.join(parent, 'Models', 'YOLOv7', 'last.pt'),
                        help='YOLOv7 weights')

    # camera parameters
    parser.add_argument('-cal', '--calibrate', action='store_true', default=False,
                        help='true if you want to calibrate the camera')
    parser.add_argument('-cd', '--camera-device', type=int, default=0, help='Camera device ID')
    parser.add_argument('-f', '--filename', type=str, default='out', help='filename for recordings')
    parser.add_argument('-j', '--jetson', action='store_true', default=False,
                        help='true if you are using the Nvidia Jetson Nano')
    parser.add_argument('-lb', '--label', action='store_true', default=False,
                        help='true if you want to save labelled signs')
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

    opt = parser.parse_args()

    print(opt)

    main(opt)




