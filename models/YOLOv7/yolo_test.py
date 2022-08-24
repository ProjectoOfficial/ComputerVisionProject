import argparse
import json
import os
from pathlib import Path
from statistics import mode
from threading import Thread

import os
import sys

current = os.path.dirname(os.path.realpath(__file__))  
parent = os.path.dirname(current)
sys.path.append(parent)

import numpy as np
import torch
import yaml
from tqdm import tqdm

from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, check_requirements, \
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import plot_images, output_to_target, plot_study_txt
from utils.torch_utils import select_device, time_synchronized, TracedModel

class Test():
    def __init__(self, weigths: str, batch_size: int, device: str, project: str, name: str = 'exp', save_txt: bool = False, half_precision : bool = True, imgsz: tuple = (1280, 1280)):
        self.weigths = weigths
        self.batch_size = batch_size
        self.device = device
        self.project = project
        self.name = name
        self.save_txt = save_txt

        self.device = select_device(device, batch_size=batch_size)

        # Set save directory
        save_dir = Path(increment_path(Path(project) / name, exist_ok=False))  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
        
        # Load model
        self.model = attempt_load(weigths, map_location=device)  # load FP32 model
        gs = max(int(self.model.stride.max()), 32)  # grid size (max stride)
        imgsz = check_img_size(imgsz, s=gs)  # check img_size

        # Set half precision model
        self.half = device.type != 'cpu' and half_precision  # half precision only supported on CUDA
        if self.half:
            self.model.half()

    def predict(self, img: torch.Tensor):
        img = img.to(self.device, non_blocking=True)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32

        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        nb, _, height, width = img.shape  # batch size, channels, height, width

        out = None
        with torch.no_grad():
            out, train_out = self.model(img, augment=False)  # inference and training outputs

        return out

if __name__ == '__main__':
    model_path = default=os.path.join(current, 'best.pt')
    batch_size = 1
    device = 'cuda'

    tester = Test(model_path, batch_size, device)