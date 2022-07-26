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

class Preprocessing(object):
    def __init__(self):
        pass

    @staticmethod
    def GaussianBlur(frame: np.ndarray, sigma:float):
        return cv2.GaussianBlur(frame, (5, 5), sigma)

    @classmethod
    def to_np_frame(cls, frame: np.ndarray):
        return np.swapaxes(np.swapaxes(np.uint8(frame), 0, 2), 0, 1)

    @staticmethod
    def Transform_base(frame: np.ndarray):
        '''
        Transform_base contains image transformation used both on camera and dataset images. It takes images coming
        from different sources and and modifies them so that the output images all have the same structure
        '''

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=(360, 480)),             # 360p
            transforms.PILToTensor(),
            ])

        image_transformed = transform(frame)
        frame = Preprocessing.to_np_frame(image_transformed.numpy())

        return frame

    @staticmethod
    def Transform_train(frame: np.ndarray):

        rand = transforms.Compose([
            transforms.RandomRotation(degrees=(-90, 90), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.RandomHorizontalFlip(p=0.1),
            transforms.ColorJitter(),
            transforms.RandomAutocontrast(p=0.1),
            transforms.RandomAffine(degrees=(-90, 90), interpolation=transforms.InterpolationMode.BILINEAR),
        ])

        transform = transforms.Compose([
            transforms.GaussianBlur(5, sigma=(0.5, 0.5)),
            transforms.RandomApply(rand, p=0.3),
        ])
        
        image_transformed = transform(frame)
        frame = Preprocessing.to_np_frame(image_transformed.numpy())

        return frame





# Robaccia di test da cavare via ma utile da scopiazzare per fare la classe preprocessing
'''
TRESH_MODE = "ADAPTIVE_GAUSSIAN" # OTSU ADAPTIVE_GAUSSIAN ADAPTIVE_MEAN


def processing(img: np.ndarray):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        th = None

        if TRESH_MODE == "OTSU":
            ret, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif TRESH_MODE == "ADAPTIVE_GAUSSIAN":
            th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 19, 4)
        elif TRESH_MODE == "ADAPTIVE_MEAN":
            th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

        canny = cv2.Canny(blur, 50, 5)

        rgb_th = cv2.cvtColor(th ,cv2.COLOR_GRAY2RGB)
        rgb_canny = cv2.cvtColor(canny ,cv2.COLOR_GRAY2RGB)
        rgb_blur = cv2.cvtColor(blur, cv2.COLOR_GRAY2RGB)

        H_stack = np.hstack((rgb_blur, rgb_th, rgb_canny))

        cv2.imshow("images", H_stack)
'''