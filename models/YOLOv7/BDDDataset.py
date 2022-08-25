import torch
import os
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from typing import List, Set, Union
from tqdm import tqdm
import json
import yaml
import random
from utils.datasets import load_mosaic, load_mosaic9, load_image, letterbox, segments2boxes, exif_size
from utils.general import check_file, xyxy2xyxyn, xywhn2xyxy, xyxy2xywhn

import argparse
import sys
import cv2
current = os.path.dirname(os.path.realpath(__file__))  
parent = os.path.dirname(current)
sys.path.append(str(Path(parent).parent))
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from Preprocessing import Preprocessing

class BDDDataset(Dataset):
    def __init__(self, data_dir: str, flag: str, hyp: dict, shape: tuple=(360,480), preprocessor: Preprocessing=None ,mosaic: bool=False, augment: bool=False, rect: bool=False, image_weights:bool =False,
     stride: int=32, batch_size: int=16, pad: float=0.0, concat_coco_names: bool = False):
        self.data_dir = data_dir
        self.flag = flag
        self.base_shape = (720, 1280)
        self.hyp = hyp

        self.preprocessor = preprocessor

        self.mosaic = mosaic
        self.augment = augment
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.img_size = max(self.base_shape)
        self.mosaic_border = [-self.img_size // 2, -self.img_size // 2]
        self.stride = stride
        self.concat_coco_names = concat_coco_names

        self.cache_name = flag

        if self.concat_coco_names:
            self.names = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', \
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', \
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', \
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', \
            'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', \
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', \
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', \
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', \
            'hair drier', 'toothbrush', 'other person', 'other vehicle', 'rider', 'traffic sign', 'trailer' )
            self.cache_name = self.cache_name + '_' + 'coco'
        else:
            self.names = ('truck', 'other person', 'motorcycle', 'bus', 'other vehicle', 'rider', 'traffic sign', 'pedestrian', 'bicycle', 'traffic light', 'train', 'car', 'trailer')

        if flag == 'train':
            self.img_dir = os.path.join(data_dir, "images", '100k', 'train')
            self.json_dir = os.path.join(data_dir, "labels", 'det_20','det_train.json')

        elif flag == 'val':
            self.img_dir = os.path.join(data_dir, "images", '100k', 'val')
            self.json_dir = os.path.join(data_dir, "labels", 'det_20','det_val.json')

        elif flag == 'test':
            self.img_dir = os.path.join(data_dir, "images", '100k', 'test')

        # DATA LOAD
        self.img_files = []
        
        if flag != 'test':
            self.labels = list()
            self.segments = list()
            self.shapes = np.array([])

            # Check cached labels data
            cache = None
            cache_path = (Path(os.path.join(self.data_dir, self.cache_name))).with_suffix('.cache')  # cached labels
            if cache_path.is_file():
                cache = torch.load(cache_path)
                cache.pop('version')

                hash_names = cache.pop('hash_names')

                labels, shapes, segments = zip(*cache.values())
                files = list(cache.keys())

                self.img_files = files
                self.shapes = np.array(shapes, dtype=np.float64)
                self.names = list(hash_names)
                self.labels = labels
                self.segments = segments

            else:
                (files, shapes, hash_names, labels, segments) = self.load_labels(cache_path)

                self.img_files = files
                self.shapes = np.array(shapes, dtype=np.float64)
                self.names = list(hash_names)
                self.labels = labels
                self.segments = segments

            assert len(self.labels) == len(self.img_files), 'There are some images without labels: labels-->{}, images-->{}'.format(len(self.labels), len(self.img_files))

        self.n = len(self.img_files)

        self.imgs = [None] * self.n
        bi = np.floor(np.arange(self.n) / batch_size).astype(np.int)  # batch index
        nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch index of image
        self.indices = range(self.n)

            # Rectangular Training
        if self.rect:
            self.batch_shapes = np.flip(self.shapes, axis=1)

    def __getitem__(self, index):
        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp['mosaic']

        img = None

        if mosaic:
            # Load mosaic
            if random.random() < 0.8:
                img, labels = load_mosaic(self, index)
            else:
                img, labels = load_mosaic9(self, index)
            shapes = None

            # MixUp https://arxiv.org/pdf/1710.09412.pdf
            if random.random() < hyp['mixup']:
                if random.random() < 0.8:
                    img2, labels2 = load_mosaic(self, random.randint(0, len(self.labels) - 1))
                else:
                    img2, labels2 = load_mosaic9(self, random.randint(0, len(self.labels) - 1))
                r = np.random.beta(8.0, 8.0)  # mixup ratio, alpha=beta=8.0
                img = (img * r + img2 * (1 - r)).astype(np.uint8)
                labels = np.concatenate((labels, labels2), 0)
        else:
            # Load image
            img, (h0, w0), (h, w) = load_image(self, index)

            # Letterbox
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            labels = self.labels[index].copy()
            if labels.size:  # normalized xywh to pixel xyxy format
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w=ratio[0] * w, h=ratio[1] * h, padw=pad[0], padh=pad[1])

            if self.preprocessor is not None:
                img, labels = self.preprocessor.Transform_base(img, labels)
            else:
                print("WARNING: Preprocessor has not been initialized and it won't be used!")
            
        if self.preprocessor is not None:
            img, labels = self.preprocessor.Transform_train(img, labels)

        nL = len(labels)  # number of labels
        if nL:
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0])  # convert xyxy to xywh
        

        labels_out = torch.zeros((nL, 6))
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return torch.from_numpy(img), labels_out, self.img_files[index], shapes

    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes

    def __len__(self):
        if len(self.img_files) == 0:
            raise Exception("\n{} is an empty dir, please download the dataset and the labels".format(self.data_dir))
        return len(self.img_files)

    def num_classes(self):
        return len(self.names)

    def load_labels(self, cache_path: Path) -> Union[List, List, Set, List]:
        '''
        Loads labels data with automatic caching
        '''
        self.json_labels = json.load(open(self.json_dir, 'r', encoding='UTF-8'))
        self.label_data = {x['name']: x for x in tqdm(self.json_labels, desc="Loading labels: ") }

        cache = dict()
        cache['version'] = 0.1

        shapes = list() 
        labels = list()
        files = list()

        i = -1
        for name in tqdm(self.label_data.keys(), desc="Preparing labels for YOLO: "):
            i+=1
            l = np.array([])
            segments = list()
            shape = None
            img = None

            try:
                path_img = os.path.join(self.img_dir, name)
                img = Image.open(path_img).convert("RGB")
                shape = exif_size(img)
                img.verify()
            except:
                continue # Image file is corrupted

            if 'labels' in self.label_data[name]:
                for element in self.label_data[name]['labels']:
                    cat = element['category']
                    if cat == 'pedestrian' and self.concat_coco_names:
                        cat = 'person'
                    bbox = np.array([self.names.index(cat), element['box2d']['x1'], element['box2d']['y1'], element['box2d']['x2'] , element['box2d']['y2']])
                    l = np.vstack([l, bbox]) if l.size else bbox
            else:
                continue # skipping empty labels
            
            l = np.reshape(l, (-1, 5)) # useful when there's only one label

            # saving boxes in xyxy format with normalization
            l[:, 1:5] = xyxy2xyxyn(l[:, 1:5], h=min(shape), w=max(shape))

            classes = np.array([x[0] for x in l], dtype=np.float32)
            segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in l] # (cls, x1y1x2y2...)
            l = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
            l = np.array(l, dtype=np.float32)
            
            if len(l):
                assert l.shape[1] == 5, 'labels require 5 columns each'
                assert (l >= 0).all(), 'negative labels'
                assert (l[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels'
                assert np.unique(l, axis=0).shape[0] == l.shape[0], 'duplicate labels'
                
            files.append(path_img)
            labels.append(l)
            shapes.append(shape)

            cache[path_img] = [l, shape, segments]
        cache['hash_names'] = self.names

        torch.save(cache, cache_path)
        return files, shapes, self.names, labels, segments


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--datadir', type=str, default=os.path.join(current, 'data', 'bdd100k'), help='*.data path')
    parser.add_argument('--hyp', type=str, default=os.path.join(current,'data', 'hyp.scratch.custom.yaml'), help='hyperparameters path')
    parser.add_argument('--task', default='val', help='train, val, test')
    parser.add_argument('--batch-size', type=int, default=2, help='size of each image batch')
    opt = parser.parse_args()

    assert opt.task in ['train', 'val','test'], 'invalid task'
    assert os.path.isdir(opt.datadir), 'data directory does not exists'
    assert os.path.exists(opt.hyp), 'hyperparameters file does not exists'
    assert opt.batch_size > 0, 'batch size should be a positive int'

    DATA_DIR = os.path.join(current, 'data', 'bdd100k')
    HYP = os.path.join(current, 'data', 'hyp.scratch.custom.yaml')
    TASK = 'val'

    hyp = check_file(HYP)
    with open(HYP) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps

    size = (1280, 1280)
    preprocess = Preprocessing(size=size)

    dataset = BDDDataset(data_dir=DATA_DIR, flag=TASK, hyp=hyp, shape=(720, 1280), preprocessor=preprocess, mosaic=False, augment=False, 
    rect=True, image_weights=False, stride=32, batch_size=1, concat_coco_names=True)

    it = iter(dataset)
    for i in range(10):
        img, labels, file, shape = next(it)
        img = Preprocessing.to_np_frame(img.cpu().detach().numpy())
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        for label in labels.cpu().detach().numpy():
            cat = dataset.names[int(label[1])]
            label[2:] = xywhn2xyxy(label[2:].reshape((-1, 4)), w=max(img.shape[:2]), h=min(img.shape[:2]))
            x, y, x2, y2 = label[2:].astype('uint32')

            print("{} {} {} {}".format(x, y, x2, y2))
            cv2.rectangle(img, (x, y), (x2, y2), (255,0,0), 2)
            cv2.putText(img,"{} - {}".format(cat, int(label[1])), (x + 5, y + 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255,0,255), 1)

        if np.ndarray((*size, 3), dtype=img.dtype).size < img.size:
            img = cv2.resize(img, (1000, 1000))

        cv2.imshow("frame", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("\n")
