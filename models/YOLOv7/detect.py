import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from Models.YOLOv7.models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

class Detect:
    def __init__(self, weights: str, device: str = 'cpu'):
        self.weights = weights
        self.device = select_device(device)

        self.model = attempt_load(weights, map_location=device)  # load FP32 model

    def static_detect(self, source: str, save_path: str, conf_thresh: float = 0.25, iou_thresh: int=0.45, 
                view_img: bool= False, imgsz: int = 640, save_img: bool = False):
        """
        parameters:
            weigths: path to weigths.pt
            source: folder containing inference images
            view_img: if true displays images with detection
        """

        # Directories
        save_dir = Path(increment_path(Path(save_path) / 'exp', False))  # increment run
        if save_img:
            view_img = False

        # Initialize
        set_logging()
        half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        stride = int(self.model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size

        if half:
            self.model.half()  # to FP16

        # Second-stage classifier
        classify = False
        if classify:
            modelc = load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=self.device)['model']).to(self.device).eval()

        # Set Dataloader
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

        # Get names and colors
        names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, imgsz, imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once
        t0 = time.time()


        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = self.model(img)[0]

            # Apply NMS
            pred = non_max_suppression(pred, conf_thresh, iou_thresh)
            t2 = time_synchronized()

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # img.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if save_img or view_img:  # Add bbox to image
                            label = f'{names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                # Stream results
                # if view_img:
                #     resized = cv2.resize(im0, (1280, 720))
                #     cv2.imshow(str(p), resized)
                #     cv2.waitKey(0)  # 1 millisecond
                #     cv2.destroyAllWindows()

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                        print(f" The image with the result is saved in: {save_path}")

        print(f'Done. ({time.time() - t0:.3f}s)')

        return im0

import os
import sys

current = os.path.dirname(os.path.realpath(__file__))  
parent = os.path.dirname(current)
sys.path.append(parent)

if __name__ == '__main__':
    with torch.no_grad():
        detect = Detect(weights=current+r'/50EPOCHE.pt')
        detect.static_detect(source=current+r'/inference/images', save_path=current+r'/runs/detect', view_img=True)
