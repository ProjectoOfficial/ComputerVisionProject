import torch
import os
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from typing import List, Dict, Set, Union
from torchvision import transforms
from tqdm import tqdm
import json
import collections

import sys
import cv2
current = os.path.dirname(os.path.realpath(__file__))  
parent = os.path.dirname(current)
sys.path.append(str(Path(parent).parent))

from Preprocessing import Preprocessing

def collate_fn(batch):
    """
    the collate_fn receives a list of tuples if your __getitem__ function
    from a Dataset subclass returns a tuple, or just a normal list if your Dataset subclass returns only one element
    """
    batch = list(zip(*batch))
    return tuple(batch)

class BDDDataset(Dataset):
    def __init__(self, data_dir: str, flag: str, shape: tuple=(360,480)):

        self.data_dir = data_dir
        self.flag = flag
        self.base_shape = shape

        if flag == 'train':
            self.img_dir = os.path.join(data_dir, "images", '100k', 'train')
            self.json_dir = os.path.join(data_dir, "labels", 'det_20','det_train.json')

        elif flag == 'val':
            self.img_dir = os.path.join(data_dir, "images", '100k', 'val')
            self.json_dir = os.path.join(data_dir, "labels", 'det_20','det_val.json')

        elif flag == 'test':
            self.img_dir = os.path.join(data_dir, "images", '100k', 'test')

        # DATA LOAD
        self.filenames = [name[:-4] for name in list(filter(lambda x: x.endswith(".jpg"), os.listdir(self.img_dir)))]
        
        if flag != 'test':
            self.names = list()
            self.labels = list()
            self.shapes = np.array([])

            # Check cached labels data
            cache = None
            cache_path = (Path(self.data_dir + r'\\' + flag)).with_suffix('.cache')  # cached labels
            if cache_path.is_file():
                cache = torch.load(cache_path)
                cache.pop('version')

                hash_names = cache.pop('hash_names')

                labels, shapes = zip(*cache.values())
                files = list(cache.keys())

                self.filenames = files
                self.shapes = np.array(shapes, dtype=np.float64)
                self.names = list(hash_names)
                self.labels = labels

            else:
                (files, shapes, hash_names, labels) = self.load_labels(cache_path)

                self.filenames = files
                self.shapes = np.array(shapes, dtype=np.float64)
                self.names = list(hash_names)
                self.labels = labels

            assert len(self.labels) == len(self.filenames), 'There are some images without labels: labels-->{}, images-->{}'.format(len(self.labels), len(self.filenames))

            self.n = len(self.names)
            self.indices = range(self.n)

    def __getitem__(self, index):
        name = self.filenames[index]
        path_img = os.path.join(self.img_dir, name)
        shapes = None

        # load images
        img = Image.open(path_img).convert("RGB")
        print(img.size)
        # load boxes and label
        labels = self.labels[index]

        # base transform
        img, labels = Preprocessing.Transform_base(img, labels, self.base_shape)

        return img, labels, self.data_dir + r'\\' + self.filenames[index], shapes

    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes

    def __len__(self):
        if len(self.filenames) == 0:
            raise Exception("\n{} is an empty dir, please download the dataset and the labels".format(self.data_dir))
        return len(self.filenames)

    def num_classes(self):
        if self.flag != 'test':
            return len(self.names)
        else:
            return 0

    def load_labels(self, cache_path: Path) -> Union[List, List, Set, List]:
        '''
        Loads labels data with automatic caching
        '''
        self.json_labels = json.load(open(self.json_dir, 'r', encoding='UTF-8'))
        self.label_data = {x['name']: x for x in tqdm(self.json_labels, desc="Loading labels: ") }

        cache = dict()
        cache['version'] = 0.1

        shapes = list()
        hash_names = ('truck', 'other person', 'motorcycle', 'bus', 'other vehicle', 'rider', 'traffic sign', 'pedestrian', 'bicycle', 'traffic light', 'train', 'car', 'trailer')
        labels = list()
        files = list()

        for name in tqdm(self.label_data.keys(), desc="Preparing labels for YOLO: "):
            boxes = np.array([])
            img = None

            try:
                path_img = os.path.join(self.img_dir, name)
                img = Image.open(path_img).convert("RGB")
                img.verify()
            except:
                continue # Image file is corrupted

            if 'labels' in self.label_data[name]:
                for element in self.label_data[name]['labels']:
                    # retrieving bounding box labels
                    bbox = np.array([list(hash_names).index(element['category']), element['box2d']['x1'], element['box2d']['y1'], element['box2d']['x2'], element['box2d']['y2']])
                    boxes = np.vstack([boxes, bbox]) if boxes.size else bbox

            # retrieving image shape
            shapes.append(self.base_shape)

            labels.append(np.reshape(boxes,(-1,5)))

            files.append(name)

            # cache[filename] = [labels, image shape]
            cache[name] = [np.reshape(boxes,(-1,5)), self.base_shape]
        cache['hash_names'] = hash_names

        torch.save(cache, cache_path)
        return files, shapes, hash_names, labels


if __name__ == "__main__":
    DATA_DIR = current+r'\data\bdd100k'
    preprocess = Preprocessing()
    trainset = BDDDataset(data_dir=DATA_DIR, flag='train', shape=(720, 1280))

    it = iter(trainset)
    img, boxes, file, shape = next(it)
    img, boxes, file, shape = next(it)
    img, boxes, file, shape = next(it)
    img, boxes, file, shape = next(it)
    img, boxes, file, shape = next(it)
    img = Preprocessing.to_np_frame(img.cpu().detach().numpy())
    
    for box in boxes.cpu().detach().numpy():
        (cat, x, y, w, h) = int(box[0]), int(box[1]), int(box[2]), int(box[3]), int(box[4]),

        print("{} {} {} {}".format(x, y, w, h))
        cv2.rectangle(img, (x, y), (w, h), (255,0,0), 2)
        cv2.putText(img,"{}".format(trainset.names[cat]), (x + 5, y + 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0,0,255), 2)

    cv2.imshow("frame", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()