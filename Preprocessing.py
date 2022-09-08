import torch
import numpy as np
import cv2
from torchvision import transforms
import random
import albumentations as A
from typing import Union

class Preprocessing():

    def __init__(self, size: tuple=(1280, 1280)):
        self.im_size = size

        self.resize_transform = A.Compose([
            A.Resize(size[0], size[1], p=1, always_apply=True),
        ], bbox_params=A.BboxParams(format='pascal_voc', min_area=3, label_fields=['class_labels']))

        self.train_trainsform = A.Compose([
            A.AdvancedBlur(p=0.2),
            A.RandomFog(p=0.1),
            A.RandomRain(p=0.1),
            A.MotionBlur(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.ColorJitter(p=0.1),
            A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.2),
        ], bbox_params=A.BboxParams(format='pascal_voc', min_area=3, label_fields=['class_labels']))

    @staticmethod
    def GaussianBlur(frame: np.ndarray, sigma:float):
        return cv2.GaussianBlur(frame, (5, 5), sigma)

    @classmethod
    def to_np_frame(cls, frame: np.ndarray):
        return np.swapaxes(np.swapaxes(np.uint8(frame), 0, 2), 0, 1)

    def Transform_base(self, frame: np.ndarray, labels: np.ndarray = np.array([])):
        '''
        Transform_base contains image transformation used both on camera and dataset images. It takes images coming
        from different sources and modifies them so that the output images all have the same structure
        ''' 

        if labels.size == 0:
            labels = np.array([[0,1,1,2,2],[0,1,1,2,2]]).reshape((-1, 5)) #dummy array

        if frame.shape != (*self.im_size, 3):
            if frame.size > np.ndarray((*self.im_size, 3), dtype=frame.dtype).size:
                class_labels = labels[:, 0]
                boxes = labels[:, 1:]
                frame, labels = self.pad_image(frame, labels)
                transformed = self.resize_transform(image=frame, bboxes=boxes, class_labels=class_labels)

                frame = transformed['image']
                transformed_bboxes = np.array(transformed['bboxes'])
                transformed_class_labels = np.array(transformed['class_labels']).reshape(-1, 1)

                if transformed_bboxes.size != 0 and transformed_class_labels.size != 0:
                    labels = np.hstack((transformed_class_labels, transformed_bboxes))
            else:
                frame, labels = self.pad_image(frame, labels)

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

    def pad_image(self, img: np.ndarray, labels: np.ndarray=None):
        dim_max = max(img.shape)

        im = np.zeros((dim_max, dim_max, 3), dtype=np.uint8)

        start_w = (dim_max - img.shape[0]) // 2
        start_h = (dim_max - img.shape[1]) // 2

        im[start_w: start_w + img.shape[0], start_h: start_h + img.shape[1], :] = img
        if labels is not None:
            labels[:, 1] += start_h
            labels[:, 2] += start_w
            labels[:, 3] += start_h
            labels[:, 4] += start_w
        return im, labels