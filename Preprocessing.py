__author__ = "Daniel Rossi, Riccardo Salami, Filippo Ferrari"
__copyright__ = "Copyright 2022"
__credits__ = ["Daniel Rossi", "Riccardo Salami", "Filippo Ferrari"]
__license__ = "GPL-3.0"
__version__ = "1.0.0"
__maintainer__ = "Riccardo Salami"
__email__ = "miniprojectsofficial@gmail.com"
__status__ = "Computer Vision Exam"

import torch
import numpy as np
import cv2
from torchvision import transforms
import random
import albumentations as A
from typing import Union

class Preprocessing():

    def __init__(self, size: tuple=(1280, 1280)):
        self.size = size

        self.crop_transform = A.Compose([
            A.Crop(0, 0, size[0], size[1]),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

        self.train_trainsform = A.Compose([
            A.AdvancedBlur(p=0.2),
            A.RandomFog(p=0.1),
            A.RandomRain(p=0.1),
            A.MotionBlur(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.ColorJitter(p=0.2),
            A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.3),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    @staticmethod
    def GaussianBlur(frame: np.ndarray, sigma:float):
        return cv2.GaussianBlur(frame, (5, 5), sigma)

    @classmethod
    def to_np_frame(cls, frame: np.ndarray):
        return np.swapaxes(np.swapaxes(np.uint8(frame), 0, 2), 0, 1)

    def Transform_base(self, frame: np.ndarray, labels: np.ndarray, np_tensor: bool=False):
        '''
        Transform_base contains image transformation used both on camera and dataset images. It takes images coming
        from different sources and and modifies them so that the output images all have the same structure
        ''' 
        
        if frame.shape != (1280, 1280, 3):
            if frame.size > np.ndarray((*self.size, 3), dtype=frame.dtype).size:
                class_labels = labels[:, 0]
                boxes = labels[:, 1:]
                transformed = self.crop_transform(image=frame, bboxes=boxes, class_labels=class_labels)

                frame = transformed['image']
                transformed_bboxes = np.array(transformed['bboxes'])
                transformed_class_labels = np.array(transformed['class_labels']).reshape(-1, 1)

                if transformed_bboxes.size != 0 and transformed_class_labels.size != 0:
                    labels = np.hstack((transformed_class_labels, transformed_bboxes))
            else:
                frame, labels = Preprocessing.pad_image(frame, labels)

        return frame, labels

    def Transform_train(self, frame: np.ndarray, labels: np.ndarray) -> Union[np.ndarray, np.ndarray]:
        class_labels = labels[:, 0]
        boxes = labels[:, 1:]

        transformed = self.train_trainsform(image=frame, bboxes=boxes, class_labels=class_labels)

        frame = transformed['image']
        transformed_bboxes = np.array(transformed['bboxes'])
        transformed_class_labels = np.array(transformed['class_labels']).reshape(-1, 1)

        if transformed_bboxes.size != 0 and transformed_class_labels.size != 0:
            labels = np.hstack((transformed_class_labels, transformed_bboxes))
        return frame, labels

    @staticmethod
    def pad_image(img: np.ndarray, labels: np.ndarray=None):
        im = np.zeros((img.shape[1], img.shape[1], 3), dtype=np.uint8)
        start = (img.shape[1] - img.shape[0]) // 2
        im[start: start + img.shape[0], :, :] = img
        if labels is not None:
            labels[:, 2] += start
            labels[:, 4] += start
        return im, labels