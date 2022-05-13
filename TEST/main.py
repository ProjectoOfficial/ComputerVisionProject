import torch
import torchvision

dataset = torchvision.datasets.Cityscapes('./data/cityscapes', split='train', mode='fine',
                     target_type='instance')