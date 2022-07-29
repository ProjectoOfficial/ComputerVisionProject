import torch
import cv2 as cv
import numpy as np
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True)
    args = parser.parse_args()
    img = cv.imread(args.source)
    cv.imshow('Original Image:', img)
    img = cv.GaussianBlur(img, (5,5), 1)
    cv.imshow('Gaussian blurred Image:', img)
    edges = cv.Canny(img, 170, 340)
    cv.imshow('Edges detected by Canny:', edges)
    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    cimg = cv.imread(args.source)
    circles = cv.HoughCircles(img,cv.HOUGH_GRADIENT,1,20,
                                param1=340,param2=80,minRadius=1,maxRadius=80)
    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        # draw the outer circle
        cv.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
        cv.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
        
    cv.imshow('Circles detected by Hough Transform:', cimg)
    cv.waitKey(0)
    cv.destroyAllWindows()




if __name__ == '__main__':
    main()
