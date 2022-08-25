import argparse
from distutils.errors import PreprocessError
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

from BDDDataset import BDDDataset
from Preprocessing import Preprocessing

import torch
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
    def __init__(self, weigths: str, batch_size: int, device: str, project: str, name: str = 'exp', save_txt: bool = False, half_precision : bool = True, imgsz: int = 640,
    live_test: bool = False):
        self.batch_size = batch_size
        self.device = device
        self.name = name
        self.names = tuple()
        self.nc = 0
        self.project = project
        self.save_txt = save_txt
        self.weigths = weigths

        self.device = select_device(device, batch_size=batch_size)
        
        # Load model
        self.model = attempt_load(weigths, map_location=self.device)  # load FP32 model
        gs = max(int(self.model.stride.max()), 32)  # grid size (max stride)
        imgsz = check_img_size(imgsz, s=gs)  # check img_size

        self.names = self.model.names if hasattr(self.model, 'names') else self.model.module.names # load classes

        # Set half precision model
        self.half = self.device.type != 'cpu' and half_precision  # half precision only supported on CUDA
        if self.half:
            self.model.half()

        self.nc = int(len(self.names))  # number of classes

        if not live_test:
            if self.device.type != 'cpu':
                self.model(torch.zeros(1, 3, imgsz, imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once


    def predict(self, img: torch.Tensor):
        img = img.to(self.device, non_blocking=True)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        out = None
        with torch.no_grad():
            out, train_out = self.model(img, augment=False)  # inference and training outputs

        return out, train_out

if __name__ == '__main__':
    DATA_DIR = os.path.join(current, 'data', 'bdd100k')
    DEVICE = '0'
    BATCH_SIZE = 8
    COMPUTE_LOSS = None
    CONF_THRES= 0.001
    IMG_SIZE = 640
    IMAGE_WEIGHTS = False
    IOU_THRES= 0.65  # for NMS
    IS_COCO = False
    HYP = os.path.join(current, 'data', 'hyp.scratch.p5.yaml')
    NAME = 'custom'
    PLOTS = True
    PROJECT = os.path.join(current, 'runs', 'test') # save dir
    SAVE_CONF = True
    SAVE_HYBRID = False
    SAVE_JSON = True
    SAVE_TXT = False | SAVE_HYBRID
    STRIDE = 20
    TASK = 'val'
    VERBOSE = True
    WEIGHTS = os.path.join(current, 'last.pt')
    WORKERS = 6

    # Set save directory
    save_dir = Path(increment_path(Path(PROJECT) / NAME, exist_ok=False))  # increment run
    (save_dir / 'labels' if SAVE_TXT else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    
    if isinstance(DATA_DIR, str):
        IS_COCO = DATA_DIR.endswith('coco.yaml')
    
    if not IS_COCO:
        data_size = (1280, 720)
        preprocessor = Preprocessing((IMG_SIZE, IMG_SIZE))
        valset = BDDDataset(DATA_DIR, TASK, HYP, data_size, preprocessor=preprocessor ,mosaic=False, augment=False, rect=True, image_weights=IMAGE_WEIGHTS, stride=STRIDE, batch_size=BATCH_SIZE) 
        valloader = torch.utils.data.DataLoader(valset, BATCH_SIZE, collate_fn=BDDDataset.collate_fn, num_workers=WORKERS)
    else:
        with open(DATA_DIR) as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)
            check_dataset(data)  # check

            task = TASK if TASK in ('train', 'val', 'test') else 'val'  # path to train/val/test images
            valloader, valset = create_dataloader(data[task], IMG_SIZE, BATCH_SIZE, STRIDE, pad=0.5, rect=True,
                                            prefix=colorstr(f'{task}: '))[0]

    tester = Test(WEIGHTS, BATCH_SIZE, DEVICE, save_dir)

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=tester.nc)
    coco91class = coco80_to_coco91_class()
    names = tester.names

    iouv = torch.linspace(0.5, 0.95, 10).to(tester.device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=tester.device)
    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []

    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(valloader, desc=s)):
        targets = targets.to(tester.device)
        nb, _, height, width = img.shape  # batch size, channels, height, width

        with torch.no_grad():
            t = time_synchronized()
            out, train_out = tester.predict(img)  # inference and training outputs
            t0 += time_synchronized() - t

            if COMPUTE_LOSS:
                loss += COMPUTE_LOSS([x.float() for x in train_out], targets)[1][:3]  # box, obj, cls

            # Run NMS
            targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(tester.device)  # to pixels
            lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if SAVE_HYBRID else []  # for autolabelling
            t = time_synchronized()
            out = non_max_suppression(out, conf_thres=CONF_THRES, iou_thres=IOU_THRES, labels=lb, multi_label=True)
            t1 += time_synchronized() - t

        # Statistics per image
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path = Path(paths[si])
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictions
            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

            # Append to text file
            if SAVE_TXT:
                gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
                for *xyxy, conf, cls in predn.tolist():
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if SAVE_CONF else (cls, *xywh)  # label format
                    with open(save_dir / 'labels' / (path.stem + '.txt'), 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')
            
            # Append to pycocotools JSON dictionary
            if SAVE_JSON:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                image_id = int(path.stem) if path.stem.isnumeric() else path.stem
                box = xyxy2xywh(predn[:, :4])  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for p, b in zip(pred.tolist(), box.tolist()):
                    jdict.append({'image_id': image_id,
                                  'category_id': coco91class[int(p[5])] if IS_COCO else int(p[5]),
                                  'bbox': [round(x, 3) for x in b],
                                  'score': round(p[4], 5)})

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=tester.device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels
                if PLOTS:
                    confusion_matrix.process_batch(predn, torch.cat((labels[:, 0:1], tbox), 1))

            # Per target class
            for cls in torch.unique(tcls_tensor):
                ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                # Search for detections
                if pi.shape[0]:
                    # Prediction to target ious
                    ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices

                    # Append detections
                    detected_set = set()
                    for j in (ious > iouv[0]).nonzero(as_tuple=False):
                        d = ti[i[j]]  # detected target
                        if d.item() not in detected_set:
                            detected_set.add(d.item())
                            detected.append(d)
                            correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                            if len(detected) == nl:  # all targets already located in image
                                break

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

        # Plot images
        if PLOTS and batch_i < 3:
            f = save_dir / f'test_batch{batch_i}_labels.jpg'  # labels
            Thread(target=plot_images, args=(img, targets, paths, f, names), daemon=True).start()
            f = save_dir / f'test_batch{batch_i}_pred.jpg'  # predictions
            Thread(target=plot_images, args=(img, output_to_target(out), paths, f, names), daemon=True).start()
    
    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=PLOTS, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=tester.nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%12i' * 2 + '%12.3g' * 4  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    if (VERBOSE or tester.nc < 50) and tester.nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (IMG_SIZE, IMG_SIZE, BATCH_SIZE)  # tuple
    print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    if PLOTS:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))

    if SAVE_JSON and len(jdict):
        w = Path(WEIGHTS[0] if isinstance(WEIGHTS, list) else WEIGHTS).stem if WEIGHTS is not None else ''  # weights
        anno_json = './coco/annotations/instances_val2017.json'  # annotations json
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        print('\nEvaluating pycocotools mAP... saving %s...' % pred_json)
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            if IS_COCO:
                eval.params.imgIds = [int(Path(x).stem) for x in valloader.dataset.img_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            print(f'pycocotools unable to run: {e}')

    # Return results
    tester.model.float()  # for training
    s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if SAVE_TXT else ''
    print(f"Results saved to {save_dir}{s}")
    maps = np.zeros(tester.nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    
    #return (mp, mr, map50, map, *(loss.cpu() / len(valloader)).tolist()), maps, t