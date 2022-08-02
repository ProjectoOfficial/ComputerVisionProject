import torch
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
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
    def __init__(self, data_dir, flag, transforms = None):

        self.data_dir = data_dir
        self.flag = flag
        self.transforms = transforms
        
        if flag == 'train':
            self.img_dir = os.path.join(data_dir, "images", '100k', 'train')
            self.json_dir = os.path.join(data_dir, "labels", 'det_20','det_train.json')

        elif flag == 'val':
            self.img_dir = os.path.join(data_dir, "images", '100k', 'val')
            self.json_dir = os.path.join(data_dir, "labels", 'det_20','det_val.json')

        elif flag == 'test':
            self.img_dir = os.path.join(data_dir, "images", '100k', 'test')

        self.filenames = [name[:-4] for name in list(filter(lambda x: x.endswith(".jpg"), os.listdir(self.img_dir)))]
        
        if flag != 'test':
            self.json_labels = json.load(open(self.json_dir, 'r', encoding='UTF-8'))
            self.label_data = {x['name']: x for x in tqdm(self.json_labels, desc="Loading labels: ") }
            
            # Remove images without bounding boxes
            for i, label in enumerate(self.json_labels):
                if not 'labels' in self.json_labels[i]:
                    self.label_data.pop(label['name'])

            self.names = []
            hash_names = set()
            self.labels = []
            for name in tqdm(self.label_data.keys(), desc="Preparing labels for YOLO: "):
                boxes = np.array([])
                for element in self.label_data[name]['labels']:
                    hash_names.add(element['category'])
                    bbox = np.array([list(hash_names).index(element['category']), element['box2d']['x1'], element['box2d']['y1'], element['box2d']['x2'], element['box2d']['y2']])
                    boxes = np.vstack([boxes, bbox]) if boxes.size else bbox
                self.labels.append(np.reshape(boxes,(-1,5)))
            self.names = list(hash_names)

            assert len(self.labels) == len(self.label_data), 'There are some images without labels: labels-->{}, images-->{}'.format(len(self.labels), len(self.filenames))

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
        if len(self.label_data) == 0:
            raise Exception("\n{} is an empty dir, please download the dataset and the labels".format(self.data_dir))
        return len(self.label_data)

    def num_classes(self):
        if self.flag != 'test':
            return len(self.names)
        else:
            return 0
