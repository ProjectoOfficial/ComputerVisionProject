import argparse
import os
import sys

from Camera import camera
from Models.YOLOv7 import yolo_test

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

def main(opt):

    assert opt.dataset is not None or opt.camera is not None or opt.lane_assistant is not None, 'specify if you want to use bdd100k dataset or the camera or the lane assistant'

    if opt.dataset:

        yolo_test.main(opt)

    if opt.camera:

        camera.main(opt)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='main.py')

    # input choice
    parser.add_argument('-d', '--dataset', action='store_true', help='the source is bdd100k')
    parser.add_argument('-c', '--camera', action='store_true', help='the source is the camera')

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
    parser.add_argument('-t', '--task', default='val', help='train, val, test, speed or study')
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

    opt = parser.parse_args()

    print(opt)

    main(opt)




