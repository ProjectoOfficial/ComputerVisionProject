import torch
import os
from PIL import Image
from torch.utils.data import Dataset
import json

def collate_fn(batch):
    """
    the collate_fn receives a list of tuples if your __getitem__ function
    from a Dataset subclass returns a tuple, or just a normal list if your Dataset subclass returns only one element
    """
    batch = list(zip(*batch))
    return tuple(batch)

class BDDDataset(Dataset):
    def __init__(self, data_dir, flag, label_list, transforms = None):

        self.data_dir = data_dir
        self.transforms = transforms
        
        if flag == 'train':
            self.img_dir = os.path.join(data_dir, "images", '100k', 'train')
            self.json_dir = os.path.join(data_dir, "labels", 'det_20','det_train.json')

        if flag == 'val':
            self.img_dir = os.path.join(data_dir, "images", '100k', 'val')
            self.json_dir = os.path.join(data_dir, "labels", 'det_20','det_val.json')

        if flag == 'test':
            self.img_dir = os.path.join(data_dir, "images", '100k', 'test')

        self.names = [name[:-4] for name in list(filter(lambda x: x.endswith(".jpg"), os.listdir(self.img_dir)))]
        self.label_data = json.load(open(self.json_dir, 'r', encoding='UTF-8'))
        self.label_data = {x['name']: x for x in self.label_data}
        self.label_list = label_list

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
        if len(self.names) == 0:
            raise Exception("\n{} is an empty dir, please download the dataset and the labels".format(self.data_dir))
        return len(self.names)