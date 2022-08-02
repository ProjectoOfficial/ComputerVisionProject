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

def collate_fn(batch):
    """
    the collate_fn receives a list of tuples if your __getitem__ function
    from a Dataset subclass returns a tuple, or just a normal list if your Dataset subclass returns only one element
    """
    batch = list(zip(*batch))
    return tuple(batch)

class BDDDataset(Dataset):
    def __init__(self, data_dir: str, flag: str, base_trans: transforms.Compose = None, train_trans: transforms.Compose = None):

        self.data_dir = data_dir
        self.flag = flag
        self.base_trans = base_trans
        self.train_trans = train_trans

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
                self.shapes = np.array([shapes], dtype=np.float64)
                self.names = list(hash_names)
                self.labels = labels

            else:
                (files, shapes, hash_names, labels) = self.load_labels(cache_path, self.base_trans)

                self.filenames = files
                self.shapes = np.array([shapes], dtype=np.float64)
                self.names = list(hash_names)
                self.labels = labels

            assert len(self.labels) == len(self.filenames), 'There are some images without labels: labels-->{}, images-->{}'.format(len(self.labels), len(self.filenames))

            self.n = len(self.names)
            self.indices = range(self.n)

    def __getitem__(self, index):
        name = self.names[index]
        path_img = os.path.join(self.img_dir, name + ".jpg")

        # load images
        img = Image.open(path_img).convert("RGB")

        # load boxes and label
        points = self.label_data[name + '.jpg']['labels']
        boxes_list = list()
        labels_list = list()

        for point in points:
            if 'box2d' in point.keys():
                box = point['box2d']
                boxes_list.append([box['x1'], box['y1'], box['x2'], box['y2']])
                label = point['category']
                labels_list.append(self.label_list.index(label))

        boxes = torch.tensor(boxes_list, dtype=torch.float)
        labels = torch.tensor(labels_list, dtype=torch.long)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

        if self.transforms is not None:
            img, target = self.transforms(img), target

        return img, target

    def __len__(self):
        if len(self.filenames) == 0:
            raise Exception("\n{} is an empty dir, please download the dataset and the labels".format(self.data_dir))
        return len(self.filenames)

    def num_classes(self):
        if self.flag != 'test':
            return len(self.names)
        else:
            return 0

    def load_labels(self, cache_path: Path, trans: transforms) -> Union[List, List, Set, List]:
        '''
        Loads labels data with automatic caching
        '''
        self.json_labels = json.load(open(self.json_dir, 'r', encoding='UTF-8'))
        self.label_data = {x['name']: x for x in tqdm(self.json_labels, desc="Loading labels: ") }

        cache = dict()
        cache['version'] = 0.1

        shapes = list()
        hash_names = set()
        labels = list()
        files = list()

        to_tensor = transforms.PILToTensor()

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
                    # filling categories list
                    hash_names.add(element['category'])

                    # retrieving bounding box labels
                    bbox = np.array([list(hash_names).index(element['category']), element['box2d']['x1'], element['box2d']['y1'], element['box2d']['x2'], element['box2d']['y2']])
                    boxes = np.vstack([boxes, bbox]) if boxes.size else bbox

            # retrieving image shape
            img = trans(to_tensor(img))
            shapes.append(img.shape[1:])

            labels.append(np.reshape(boxes,(-1,5)))

            files.append(name)

            # cache[filename] = [labels, image shape]
            cache[name] = [np.reshape(boxes,(-1,5)), img.shape[1:]]
        cache['hash_names'] = hash_names

        torch.save(cache, cache_path)
        return files, shapes, hash_names, labels
