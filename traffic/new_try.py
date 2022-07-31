from types import NoneType
import torch
import cv2 as cv
import numpy as np
import argparse
import os
import sys
from pathlib import Path
from torch import nn
import typing
import time
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.realpath(__file__))

class Sign_Matcher():
    def __init__(self, stride: int = 1) -> typing.List:
        self.nccs = {}
        self.keys = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]
        self.speeds = {}
        
        for i in range(13):
            filename = str(ROOT) + '\\cartello' + str(self.keys[i]) + '_bello2.jpg' 
            img = cv.imread(filename=filename)
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            img = cv.GaussianBlur(img, (5,5), 1)
            img = cv.resize(img, (200, 200), interpolation=cv.INTER_LINEAR)
            _, img = cv.threshold(img,0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
            #cv.imshow('img' + str(i), img)
            self.speeds[self.keys[i]] = img
        
    def match(self, sign: np.ndarray) -> None:
        start_time = time.time()
        
        # All the 6 methods for comparison in a list
        methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
                    'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']
        stats = [[]]
        max_methods = []
        min_methods = []
        img2 = sign
        cv.imshow('detected sign', img2)
        for speed, template in self.speeds.items():
            w, h = template.shape[::-1]
            min_meth_list = []
            max_meth_list = []
            for meth in methods:
                img = img2.copy()
                method = eval(meth)
                # Apply template Matching
                res = cv.matchTemplate(img,template,method)
                min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
                # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
                if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
                    top_left = min_loc
                    best_val = min_val
                    top = 'Minimum'
                else:
                    top_left = max_loc
                    best_val = max_val
                    top = 'Maximum'
                if top == 'Minimum':
                    min_meth_list.append(best_val)
                else:
                    max_meth_list.append(best_val)
            max_methods.append(max_meth_list)
            min_methods.append(min_meth_list)
        print(f"Minimum Values:\n")
        print('\t', end='')
        for method in ['cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']: print(method, end='\t')
        i=0
        print() #line feed
        for speed in self.speeds.keys():
            print(speed, end = '\t')
            print(min_methods[i])
            i+=1

        print("\nMaximum Values:\n")
        print('\t', end='')
        methods.remove('cv.TM_SQDIFF')
        methods.remove('cv.TM_SQDIFF_NORMED')
        for method in methods : print(method, end='\t')
        print()
        i=0
        for speed in self.speeds.keys():
            print(speed, end = '\t')
            print(max_methods[i])
            i+=1
        
        bests = []
        for el in min_methods:
            bests.append(el.index(min(el)))
        for el in max_methods:
            bests.append(el.index(max(el)))
        print("--- %s seconds ---" % (time.time() - start_time))
        return bests
        
        


class Sign_Detector():
    def __init__(self, hough_gradient: bool = True, dp: float = 1, minDist: int = 100, param1: int = 400, param2: int = 80, minRadius: int = 0, maxRadius: int = 0) -> None:
        method_simple = True if hough_gradient else 'HOUGH_GRADIENT_ALT'
        self.method = cv.HOUGH_GRADIENT if method_simple else cv.HOUGH_GRADIENT_ALT
        self.dp = dp
        self.minDist = minDist
        self.param1 = param1
        self.param2 = param2
        self.minRadius = minRadius
        self.maxRadius = maxRadius

    def detect(self, img: np.ndarray) -> np.ndarray:
        previous = self.maxRadius
        if self.maxRadius == 0:
            minore = img.shape[0] if img.shape[0] < img.shape[1] else img.shape[1]
            self.maxRadius = minore //10 #the circle must not be bigger than 10% of the image
        circles = cv.HoughCircles(img, method=self.method, dp=self.dp, minDist=self.minDist, param1=self.param1, param2=self.param2, minRadius=self.minRadius, maxRadius=self.maxRadius)
        self.maxRadius = previous
        if type(circles) == NoneType: #cv.HoughCircles() return a NoneType object when it cannot find any circles
            return np.zeros(1)
        circles = np.uint16(np.around(circles))
        
        return circles


def draw_circle(img, circle):
    i = circle
    # draw the outer circle
    cv.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv.circle(img,(i[0],i[1]),2,(0,0,255),3)
    top_left = (i[0] - i[2], i[1] - i[2])
    bottom_right = (i[0] + i[2], i[1] + i[2])
    cv.rectangle(img, top_left, bottom_right, (0, 255, 255))
    sign = img[top_left[1] : bottom_right[1], top_left[0] : bottom_right[0]]
        
        
    cv.imshow('Circles detected by Hough Transform:', img)

def main():
    
    ROOT = os.path.dirname(os.path.realpath(__file__))
    dir = ROOT + '\\photos'
    img = cv.imread(str(dir + '\\IMG_20220731_153855.jpg'),0) #already in grayscale
    img = cv.GaussianBlur(img, (5,5), 0)
    img = cv.resize(img, (750, 1000), interpolation=cv.INTER_LINEAR)
    cv.imshow('Canny', cv.Canny(img, 200, 400))
    
    original = img
        
    
    sd = Sign_Detector(hough_gradient=False, param2=55)
    circles = sd.detect(img=img)
    if len(circles.shape) < 3:
        print(f"Non sono stati trovati cerchi!")
        
    else:
        circle = circles[0, 0] #we take the first of the detected circles
        draw_circle(img, circle)
        sm = Sign_Matcher()
        i = circle
        top_left = (i[0] - i[2], i[1] - i[2])
        bottom_right = (i[0] + i[2], i[1] + i[2])
        sign = original[top_left[1] : bottom_right[1], top_left[0] : bottom_right[0]]
        #now we scale it
        sign = cv.resize(sign, (200, 200), interpolation=cv.INTER_LINEAR)
        #then we binarize the piece of image that contains the traffic sign
        _, sign = cv.threshold(sign, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
        
        bests = sm.match(sign)
        #devo finire di mettere a posto la restituzione dei valori
        #print(f"The best results, based on the various methods, are:")
        #print(bests)



    

    cv.waitKey(0)
    cv.destroyAllWindows()




if __name__ == '__main__':
    main()
