import torch
import torchvision
from torch.utils.data import Dataset

class Cityscapes(Dataset):
    N_CLASSES = 19

    def __init__(self, root, split="train"):
        super(self, Cityscapes).__init__()

        self.root = root
        self.split = split
        self.files = {}

        self.class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence', \
                            'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain', \
                            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', \
                            'motorcycle', 'bicycle']
    
    def __len__(self):
        return len(self.files[self.split])