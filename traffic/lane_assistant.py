from audioop import cross
from genericpath import isdir, isfile
import os
import numpy as np
import cv2 as cv
import math
import argparse

import time
import shutil
from typing import Union, Tuple
from pathlib import Path

#Change Input path, the specific filename is specified and appended in the main
INPUT = Path(os.path.dirname(os.path.abspath(__file__))+'/photos')

#Results is the path in which you want to save the processed video if you enable
#save_video in the main(), the name of the file by default is the same as the input video
RESULTS = Path(os.path.dirname(os.path.abspath(__file__))+'/results')

#class used for writing and drawing on the image
class LaneAnnotator():
  def __init__(self, w: int = 1080, h: int = 1920):
    self.font = cv.FONT_HERSHEY_SIMPLEX
    self.org = (w//2, round(h//10*9.5))
    major = w if w > h else h
    self.fontScale = 1.2 * major / 1920
    self.color = (0, 0, 255)
    self.thickness = 2

  def write(self, img: np.ndarray, text: str):
    return cv.putText(img, text, self.org, self.font, self.fontScale, self.color, self.thickness, cv.LINE_AA, False)
  def write_left(self, img: np.ndarray, text: str):
    org = (0, self.org[1])
    return cv.putText(img, text, org, self.font, self.fontScale, self.color, self.thickness, cv.LINE_AA, False)
  def write_right(self, img: np.ndarray, text: str):
    org = (round(self.org[0]*1.3), self.org[1])
    return cv.putText(img, text, org, self.font, self.fontScale, self.color, self.thickness, cv.LINE_AA, False)


  def reset_params(self, w: int, h: int):
    self.font = cv.FONT_HERSHEY_SIMPLEX
    self.org = (w//2, round(h//10*9.5))
    self.fontScale = 1.6
    self.color = (0, 0, 255)
    self.thickness = 3

#all the commented lines below are different configurations for the bilateral filter
#and for the mask (trapezoid), the uncommented ones are those that work better in general
#of course, the mask is the best for the point of view of the phone in my car, it probably needs
#a little set-up for other cars

class LaneDetector():
    def __init__(self, w=1080, h = 1920) -> None:
        self.thresholds = (50, 60, 70, 80, 90, 100, 120, 130, 140, 150)
        self.blurs = ((3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (5, 5), (5, 5), (7, 7), (7, 7), (7, 7))
        #self.sigmas = (9, 27, 45, 63, 79, 99, 119, 140, 160, 180) #full HD parameters
        self.sigmas = (30, 40, 50, 65, 79, 99, 119, 140, 160, 180) #full HD parameters NOW
        #self.sigmas = (3, 9, 19, 33, 45, 59, 79, 99, 115, 135) #HD parameters
        #self.sigmas = (3, 9, 19, 33, 45, 55, 70, 80, 90, 100) #HD parameters
        #self.sigmas = (3, 9, 19, 33, 70, 70, 70, 70, 70, 70)
        #self.sigmas = (220, 220, 220, 220, 220, 220, 220, 220, 220, 220)
        #self.sigmas = (3, 3, 3, 3, 3, 3, 3, 3, 3, 3)
        self.an = LaneAnnotator(w, h)
        #self.trapezoid = np.array([[(0, h//100*70), (0, h//100*50), (w//100*40, h//100*40), (w//100*60, h//100*40), (w, h//100*50), (w, h//100*70)]], dtype=np.int32)
        #self.trapezoid = np.array([[(0, h//100*90), (0, h//100*75), (w//100*40, h//100*50), (w//100*60, h//100*50), (w, h//100*75), (w, h//100*90)]], dtype=np.int32)
        #self.trapezoid = np.array([[(w//100*45, h//100*75), (w//100*45, h//100*90), (0, h//100*90), (0, h//100*75), (w//100*40, h//100*50), (w//100*60, h//100*50), (w, h//100*75), (w, h//100*90), (w//100*55, h//100*90), (w//100*55, h//100*75)]], dtype=np.int32)
        self.trapezoid = np.array([[(w//100*50, h//100*65), (w//100*40, h//100*90), (w//100*15, h//100*90), (w//100*10, h//100*75), (w//100*40, h//100*60), (w//100*60, h//100*60), (w//100*90, h//100*75), (w//100*85, h//100*90), (w//100*60, h//100*90), (w//100*50, h//100*65)]], dtype=np.int32) #MINE
        #self.trapezoid = np.array([[(w//100*50, h//100*80), (w//100*40, h//100*95), (w//100*15, h//100*95), (w//100*10, h//100*80), (w//100*40, h//100*65), (w//100*60, h//100*65), (w//100*90, h//100*80), (w//100*85, h//100*95), (w//100*60, h//100*95), (w//100*50, h//100*80)]], dtype=np.int32) #DANIEL
        mask = np.zeros(shape=(h, w), dtype=np.uint8)
        #self.points = np.array((([0, 100], [450, 100],[0, 500], [450, 500])))
        self.mask = cv.fillPoly(mask, self.trapezoid, 255)
        self.colors = [()]
    
    def choose_colors(self, danger):
        colors = [(0, 255, 0), (0, 255, 0)]
        if danger[0]:
            colors[0] = (0, 0, 255)
        if danger[1]:
            colors[1] = (0, 0, 255)
        return colors
    
    def find_edges_easy2(self, img: np.array, t_low = 100, t_high = 200):
        #blur = img.copy()
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        h = gray.shape[0]
        mean_lum = np.mean(gray[h//2:, :])
        eq = True if mean_lum > 100 else False
        #gray = self.palette(gray, 6)
        if mean_lum > self.thresholds[0]:
            for i in range(len(self.thresholds) - 1, - 1, -1):
                if mean_lum > self.thresholds[i]:
                    gray = cv.GaussianBlur(gray, self.blurs[i], 0)
                    break
        #blur = cv.GaussianBlur(blur, (7, 7), 0)
        if eq == True:
            #gray = cv.equalizeHist(gray)
            clahe = cv.createCLAHE(clipLimit=1.5, tileGridSize=(16,16))
            gray = clahe.apply(gray)
        #gray = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
        return cv.Canny(gray, t_low, t_high, apertureSize=3, L2gradient=True)
        #return cv.Sobel(gray, -1, 1, 1, ksize=3)
        #return cv.Laplacian(gray, -1, ksize=1)

    def find_edges_bilateral2(self, img: np.array, t_low = 100, t_high = 200):
        #eq = True
        #blur = img.copy()
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        h = gray.shape[0]
        mean_lum = np.mean(gray[:, :])
        lower_lum = np.mean(gray[h//2:, :])
        #gray = self.palette(gray, 4)
        eq = True if mean_lum > 100 else False
        t_high = 300 if lower_lum > 120 else 250 if lower_lum > 80 else 200
        if mean_lum > self.thresholds[0]:
            for i in range(len(self.thresholds) - 1, - 1, -1):
                if mean_lum > self.thresholds[i]:
                    gray = cv.bilateralFilter(gray, 1, sigmaColor=self.sigmas[i], sigmaSpace=self.sigmas[i]) #good setup
                    #gray = cv.bilateralFilter(gray, -1, sigmaColor=self.sigmas[i], sigmaSpace=2)
                    break
        if eq == True:
            #gray = cv.equalizeHist(gray)
            clahe = cv.createCLAHE(clipLimit=0.5, tileGridSize=(8,8))
            gray = clahe.apply(gray)
            #pass
        
        return cv.Canny(gray, t_low, t_high)
        #return cv.Sobel(gray, -1, 1, 1, ksize=3)

    def processing(self, edges: np.array):
        return cv.bitwise_and(edges, self.mask, mask=None)

    def find_lines(self, edges: np.array):
        found = False
        lines = [None, None]
        h, w = edges.shape
        edges1 = edges.copy()
        edges2 = edges.copy()
        edges1[:, w//10*5:] = 0
        edges2[:, 0:w//10*5] = 0
        lines1 = cv.HoughLines(edges1, rho=1, theta=np.pi / 180 * 2, threshold=70, lines=None, min_theta=np.pi / 180 * 23, max_theta=np.pi / 180 * 62)
        lines2 = cv.HoughLines(edges2, rho=1, theta=np.pi / 180 * 2, threshold=80, lines=None, min_theta=np.pi / 180 * 118, max_theta=np.pi / 180 * 142)
        #lines2 = None
        #lines = cv.HoughLinesP(edges, 1, np.pi / 180, 200, None, 50, 5)
        # Draw the lines
        if lines1 is not None:
            lines[0] = lines1
            found = True
        if lines2 is not None:
            lines[1] = lines2
            found = True
        return lines
    
    def is_danger(self, lines):
        danger = [False, False]
        if lines[0] is not None:
            lines1 = lines[0]
            if lines1[0][0][1] < 0.8:
                danger[0] = True
        if lines[1] is not None:
            lines2 = lines[1]
            if lines2[0][0][1] > 2.3:
                danger[1] = True
        return danger

    def draw_lines(self, original, lines: list, colors: list):
        lines1 = lines[0]
        lines2 = lines[1]
        cdst = original.copy()
        h = original.shape[0]
        #lines2 = None
        #lines = cv.HoughLinesP(edges, 1, np.pi / 180, 200, None, 50, 5)
        # Draw the lines
        text1 = text2 = 'Safe'
        if lines1 is not None:
            for i in range(0, 1): #len(lines)
                rho = lines1[i][0][0]
                theta = lines1[i][0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 5000*(-b)), int(y0 + 5000*(a)))
                pt2 = (int(x0 - 5000*(-b)), int(y0 - 5000*(a)))
                cv.line(cdst, pt1, pt2, colors[0], 5, cv.LINE_AA)
                if theta < 0.8:
                    text1 = "Beware!"
                #print(f"theta_1: {round(theta * 180 / np.pi, 2)}")
        if lines2 is not None:
            for i in range(0, 1):
                rho = lines2[i][0][0]
                theta = lines2[i][0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 5000*(-b)), int(y0 + 5000*(a)))
                pt2 = (int(x0 - 5000*(-b)), int(y0 - 5000*(a)))
                cv.line(cdst, pt1, pt2, colors[1], 5, cv.LINE_AA)
                if theta > 2.30:
                    text2 = "Beware!"
                #print(f"theta_2: {round(theta * 180 / np.pi, 2)}")
        cdst = self.an.write_left(cdst, text1)
        cdst = self.an.write_right(cdst, text2)
        cdst[0:h//100*72, :] = original[0:h//100*72, :]
        return cdst
    def remove_grass(self, img: np.array):
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        tmp1 = np.logical_and(hsv[:, :, 0] > 20, hsv[:, :, 0] < 60)
        tmp2 = hsv[:, :, 1] > 35 # around 13% saturation
        tmp = np.logical_and(tmp1, tmp2)
        hsv[tmp, 2] = 0
        return cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    def palette(self, img: np.array, level: int = 4):
        return img//level*level
    
    def detect(self, img: np.array, t_low = 105, t_high = 210, bilateral = False):
        filtered = self.remove_grass(img)
        if not bilateral:
            edges = self.find_edges_easy2(img = filtered, t_low = t_low, t_high = t_high)
        else:
            edges = self.find_edges_bilateral2(img = filtered, t_low = t_low, t_high = t_high)
        post_edges = self.processing(edges)
        lines = self.find_lines(post_edges)
        return lines
    
    def detect_debug(self, img: np.array, t_low = 100, t_high = 200, bilateral = False, post=False):
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        h, w = gray.shape
        gray_level = np.around(np.mean(gray[:, :]), 2)
        filtered = self.remove_grass(img)
        if not bilateral:
            edges = self.find_edges_easy2(img = filtered, t_low = t_low, t_high = t_high)
        else:
            edges = self.find_edges_bilateral2(img = filtered, t_low = t_low, t_high = t_high)
        post_edges = self.processing(edges)
        lines = self.find_lines(post_edges)
        frame = cv.cvtColor(post_edges, cv.COLOR_GRAY2BGR) if post else cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
        self.an.write(frame, str(gray_level))
        return frame, lines

#the program can take both an image and a video, if the extension is jpg or png it will
#treat it as an image, otherwise as a video.
def main(opt):

    filepath = INPUT / opt.file
    filename = str(filepath)

    if not opt.real_time:
        ext = filepath.suffix
        images_ext = [".jpg", ".png"]
        #IMAGE
        if ext in images_ext:
            frame = cv.imread(filename)
            frame = cv.resize(frame, (900, 1200))
            h = frame.shape[0]
            w = frame.shape[1]
            ld = LaneDetector(w, h)
            if frame.size == 0:
                exit(0)
            #Only one of the following 2 lines must be commented:
            lines = ld.detect(frame, bilateral= opt.bilateral)                              #show real frame
            #frame, lines = ld.detect_debug(frame, bilateral = bilateral, post = post)  #show edges frame (for debug purposes)
            danger = ld.is_danger(lines=lines)
            frame_out = ld.draw_lines(frame, lines, ld.choose_colors(danger))
            name = "bilateral " if opt.bilateral else "blur "
            cv.imshow(name, cv.resize(frame_out, (600, 800)))
            cv.waitKey(0)
            cv.destroyAllWindows()
        #VIDEO
        else:
            cap = cv.VideoCapture(filename)
            #dims = (720, 1280)
            dims = (1280, 720)
            #dims = (450, 800)
            if not cap.isOpened():
                print("Cannot open camera")
                exit()
            i = 0
            start_time = time.time()
            ret, frame = cap.read()
            frame = cv.flip(cv.flip(frame, 0), 1)
            frame = cv.resize(frame, dims)
            cv.imwrite(str(INPUT / 'blabla.jpg'), frame)
            ld = LaneDetector(frame.shape[1], frame.shape[0])
            if not opt.save_video:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        print("Can't receive frame (stream end?). Exiting ...")
                        break
                    frame = cv.resize(frame, dims)
                    #Only one of the following 2 lines must be commented:
                    lines = ld.detect(frame, bilateral= opt.bilateral)                              #show real frame
                    #frame, lines = ld.detect_debug(frame, bilateral = bilateral, post = post)  #show edges frame (for debug purposes)
                    danger = ld.is_danger(lines=lines)
                    frame_out = ld.draw_lines(frame, lines, ld.choose_colors(danger))
                    cv.imshow('frame', cv.resize(frame_out, (1280, 720)))
                    if cv.waitKey(1) == ord('q'):
                        break 
                    #cv.waitKey(0)
                    i+=1
                # When everything done, release the capture
                end_time = time.time()
                cap.release()
                cv.destroyAllWindows()
                delta_time = round(end_time - start_time, 3)
                fps = round(i / delta_time, 3)
                print(f"Processed frames: {i}, total time: {delta_time} seconds, fps: {fps}.")
            else:
                if not os.path.isdir(str(RESULTS)):
                    os.mkdir(str(RESULTS))
                index = opt.file.find('.')
                file = opt.file[0:index]
                ext = '.avi'
                results_dir = str(RESULTS)
                if os.path.isfile(results_dir + '\\' + file+ext):
                    for j in range(1, 100):
                        if not os.path.isfile(results_dir + '\\' + file + '(' + str(j) + ')' + ext):
                            file = file + '(' + str(j) + ')'
                            break
                out = cv.VideoWriter(str(RESULTS / (file +'.avi')), cv.VideoWriter_fourcc(*'XVID'), 30, (1280, 720))
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        print("Can't receive frame (stream end?). Exiting ...")
                        break
                    frame = cv.resize(frame, dims)
                    #Only one of the following 2 lines must be commented:
                    lines = ld.detect(frame, bilateral= opt.bilateral)                              #show real frame
                    #frame, lines = ld.detect_debug(frame, bilateral = bilateral, post = post)  #show edges frame (for debug purposes)
                    danger = ld.is_danger(lines=lines)
                    frame_out = ld.draw_lines(frame, lines, ld.choose_colors(danger))
                    #cv.imshow('frame', cv.resize(frame_out, (1280, 720)))
                    out.write(frame_out)
                    i+=1
                # When everything done, release the capture
                end_time = time.time()
                cap.release()
                out.release()
                cv.destroyAllWindows()
                delta_time = round(end_time - start_time, 3)
                fps = round(i / delta_time, 3)
                print(f"Processed frames: {i}, total time: {delta_time} seconds, fps: {fps}.")
    #REAL-TIME VIDEO
    else:
        cap = cv.VideoCapture('http://place your ip and port here/video')
        if not cap.isOpened():
            print("Cannot open camera")
            exit()
        i = 0
        start_time = time.time()
        ret, frame = cap.read()
        frame = cv.flip(cv.flip(frame, 0), 1)
        ld = LaneDetector(frame.shape[1], frame.shape[0])
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            frame = cv.flip(cv.flip(frame, 0), 1)
            #frame = cv.resize(frame, dims)
            frame = cv.resize(frame, dims)
            #Only one of the following 2 lines must be commented:
            lines = ld.detect(frame, bilateral= opt.bilateral)                              #show real frame
            #frame, lines = ld.detect_debug(frame, bilateral = bilateral, post = post)  #show edges frame (for debug purposes)
            danger = ld.is_danger(lines=lines)
            frame_out = ld.draw_lines(frame, lines, ld.choose_colors(danger))
            cv.imshow('frame', cv.resize(frame_out, (1280, 720)))
            if cv.waitKey(1) == ord('q'):
                break
            i+=1
        # When everything done, release the capture
        end_time = time.time()
        cap.release()
        cv.destroyAllWindows()
        delta_time = round(end_time - start_time, 3)
        fps = round(i / delta_time, 3)
        print(f"Processed frames: {i}, total time: {delta_time} seconds, fps: {fps}.")

    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-rtm', '--real-time', action='store_true', default=False, help='Enable if you want to process a video in real-time with the IP Webcam App')
    parser.add_argument('-ps', '--post', action='store_true', default=False, help='Enable if you want to see the video with the mask applied, in order to see what the program see (useful if you need to set the mask properly)')
    parser.add_argument('-bl', '--bilateral', action='store_false', default=True, help='Enable if you want to use the bilateral filter, if False the the standard Gaussian Blur is performed (but bilateral works better)')
    parser.add_argument('-sv', '--save-video', action='store_true', default=False, help='Enable if you want to save the video in the directory RESULTS, with the same file name as the input (default extension is ".avi")')
    parser.add_argument('-fl', '--file', type=str, help='Name of the file')

    opt = parser.parse_args()

    main(opt)
