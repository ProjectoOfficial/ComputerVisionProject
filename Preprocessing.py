import numpy as np
import cv2

class Preprocessing(object):
    def __init__(self):
        pass

    @staticmethod
    def GaussianBlur(frame: np.ndarray, sigma:float):
        return cv2.GaussianBlur(frame, (5, 5), sigma)


TRESH_MODE = "ADAPTIVE_GAUSSIAN" # OTSU ADAPTIVE_GAUSSIAN ADAPTIVE_MEAN


# Robaccia di test da cavare via ma utile da scopiazzare per fare la classe preprocessing
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