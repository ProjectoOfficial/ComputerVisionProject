import argparse
import os
import sys
from models.YOLOv7 import yolo_test
from Camera import camera

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='main.py')
    parser.add_argument('--dataset', action='store_true', help='the source is bdd100k')
    parser.add_argument('--camera', action='store_true', help='the source is the camera')

    # yolo_test parameters
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--batch-size', type=int, default=9, help='size of each image batch')
    parser.add_argument('--compute-loss', default=None, help='')
    parser.add_argument('--conf-threes', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--data', type=str, default=os.path.join(current, 'data', 'bdd100k'), help='*.data path')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--hyp', type=str, default=os.path.join(current, 'data', 'hyp.scratch.p5.yaml'), help='')
    parser.add_argument('--image-weights', type=bool, default=False, help='')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--is-coco', type=bool, default=False, help='')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='IOU threshold for NMS')
    parser.add_argument('--name', type=str, default='custom', help='')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--plots', action='store_true', help='')
    parser.add_argument('--project', default=os.path.join(current, 'runs', 'test'), help='save to project/name')
    parser.add_argument('--save-txt', default=False, action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', default=False, action='store_true',
                        help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', default=True, action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', default=True, action='store_true',
                        help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--stride', type=int, default=32, help='')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--verbose', default=True, action='store_true', help='report mAP by class')
    parser.add_argument('--weights', nargs='+', type=str, default=os.path.join(current, 'last.pt'),
                        help='model.pt path(s)')
    parser.add_argument('--workers', type=int, default=6, help='')
    opt = parser.parse_args()

    print(opt)

    assert opt.dataset is not None or opt.camera is not None, 'specify if you want to use bdd100k dataset or the camera'

    if opt.dataset:

        yolo_test.Test(opt.weights, opt.batch_size, opt.device, opt.project, opt.name, opt.save_txt, opt.halp_precision, opt.imgsz, opt.live_test)

    if opt.camera:

        print('')




