from audioop import cross
from genericpath import isdir, isfile
import numpy as np
import cv2 as cv

import os
import time
import shutil
from typing import Union, Tuple
from pathlib import Path

ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__))) # This is your Project Root
RESULTS_DIR = ROOT_DIR / 'detected_circles'
SIGNS_DIR = ROOT_DIR / 'signs'
TEMPLATES_DIR = ROOT_DIR / 'templates_solo_bordo'

"""
def check_rotation(path_video_file):
  # this returns meta-data of the video file in form of a dictionary
  meta_dict = ffmpeg.probe(path_video_file)  
  # from the dictionary, meta_dict['streams'][0]['tags']['rotate'] is the key
  # we are looking for
  rotateCode = None
  if int(meta_dict['streams'][0]['tags']['rotate']) == 90:
      rotateCode = cv.ROTATE_90_CLOCKWISE
  elif int(meta_dict['streams'][0]['tags']['rotate']) == 180:
      rotateCode = cv.ROTATE_180
  elif int(meta_dict['streams'][0]['tags']['rotate']) == 270:
      rotateCode = cv.ROTATE_90_COUNTERCLOCKWISE  
  return rotateCode

def correct_rotation(frame, rotateCode):  
  return cv.rotate(frame, rotateCode) 

"""

"""
Class to perform preprocessing on the input frames, we have:
1) Gaussian Kernel sliding on the image in order to smooth the noise
2) Conversion from BGR to HSV colorspace, in order to better detect the "redness" of the traffic signs' border
3) Shutting down (to black) all the pixels that aren't red enough
4) Conversion from HSV to Grayscale
"""
class Preprocessor():
  def __init__(self):
    pass
  def prep(self, img: np.ndarray) -> np.ndarray:
    #cv.imshow('Resized Image',img)
    img2 = cv.GaussianBlur(img, (3, 3), 0)
    img2 = cv.cvtColor(img2, cv.COLOR_BGR2HSV)
    temp = np.logical_and(img2[:, :, 0] > 11, img2[:, :, 0] < 169)  #colors not red
    img2[temp, 2] = 0
    h, w, _ = img2.shape
    h, w, _ = img2.shape
    gray_img = np.reshape(img2[:, :, 2], (h, w)) #from HSV to GRAYSCALE
    return gray_img

"""
Detector class, in which we perform the Canny Edge Detection and the Circles detection via the Hough Transform
"""
class Detector():
  def __init__(self, HOUGH_GRADIENT=True, dp=2.2, minDist = 500, param1=230, param2=85, minRadiusRatio=100, maxRadiusRatio = 20):
    self.method = cv.HOUGH_GRADIENT
    if not HOUGH_GRADIENT:
      self.method = cv.HOUGH_GRADIENT_ALT
    self.dp = dp
    self.minDist = minDist
    self.param1 = param1
    self.param2 = param2
    self.minRadiusRatio = minRadiusRatio
    self.maxRadiusRatio = maxRadiusRatio
  #single-channel image
  def detect(self, gray_img: np.ndarray, print_canny : bool= False):
    if print_canny:
      cv.imwrite(ROOT_DIR + '\\canny_edges.jpg', cv.Canny(gray_img, self.param1, self.param1//2))
    minimum = gray_img.shape[0] if gray_img.shape[0] < gray_img.shape[1] else gray_img.shape[1]
    maxR = minimum // self.maxRadiusRatio
    minR = minimum // self.minRadiusRatio
    circles = cv.HoughCircles(image=gray_img, method=self.method, dp=self.dp, minDist=self.minDist, param1=self.param1, param2=self.param2, minRadius=minR, maxRadius=maxR)
    return circles


"""
Matcher class, in which, during the set-up phase (when the program is started), we extract the keypoints from the
"template" speed limit signs: the program can work with different resizing and different blurs, althouhg in the
current configuration the templates are not blurred since, after testing, we saw that it lowered the accuracy
"""
class Matcher():
  def __init__(self, sift = False, path = ''):
    self.kp = []
    self.features = []
    self.num = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120] #traffic signs
    self.dims = [(100, 100)]
    self.blurs = [(3, 3)]
    self.det = cv.SIFT_create() if sift else cv.ORB_create()
    self.sift = sift
    for i in range(len(self.num)):
      dir = path / ('cartelli' + str(self.num[i]))
      if os.path.isdir(dir): #better safe than sorry
        files = os.listdir(dir)
        kp_list = []
        des_list = []
        for file in files:
          file = os.path.join(dir, file)  
          if os.path.isfile(file):
            img = cv.imread(file)
            for dim in self.dims:
              for blur in self.blurs:
                img = cv.resize(img, dim, interpolation = cv.INTER_AREA)
                gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
                _, gray = cv.threshold(gray, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
                kp = self.det.detect(gray,None)
                kp, des = self.det.compute(gray, kp)
                kp_list.append(kp)
                des_list.append(des)
        #once we have the keypoints for all the templates of A SPECIFIC traffic sign,
        #we add the list of keypoints to self.kp, which is a list of lists
        self.kp.append(kp_list)
        self.features.append(des_list) #keypoints for the i-th image
    self.bf = cv.BFMatcher_create(normType=cv.NORM_HAMMING2) if not sift else cv.BFMatcher_create(normType=cv.NORM_L2)
    #now we have the keypoints for all the template images
  
  #the match methodresizes the detected sign and extracts the keypoints, using either SIFT or ORB (in our configuration
  # we decided to use SIFT), then performs the knn method in order to find the template that best corresponds to the
  # detected sign
  def match(self, sign: np.ndarray, show_scores = False) -> int:
    sign = cv.resize(sign, self.dims[0], interpolation = cv.INTER_LINEAR_EXACT)
    gray= cv.cvtColor(sign,cv.COLOR_BGR2GRAY)
    _, gray = cv.threshold(gray, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    det = cv.SIFT_create() if self.sift else cv.ORB_create()
    kp = det.detect(gray,None)
    # compute the descriptors with ORB
    kp, key_ps = det.compute(gray, kp)
    alpha = 0.8
    knn = 2
    #append = True
    if key_ps is not None:
      scores = {}
      for i in range(len(self.num)):
        best = 0
        for features in self.features[i]:
          matches = self.bf.knnMatch(features, key_ps, k=knn)
          good = []
          for j in range(len(matches)):
            """
            if len(matches[j]) == knn:
              for k in range(0, knn-1):
                if not matches[j][k].distance < alpha*matches[j][k+1].distance:
                  append = False
                  break
              if append:
                good.append([matches[j][0]])
            """
            if len(matches[j]) == knn:
              if matches[j][0].distance < alpha*matches[j][1].distance :#and matches[j][1].distance < alpha*matches[j][2].distance:
                good.append([matches[j][0]])
            
          #print(len(good))
          if len(good) > best:
            best = len(good)
        scores[self.num[i]] = best
      #ora bisognerebbe mettere che se il primo ed il secondo score non sono ad una 
      #sufficiente distanza, bisoga buttare il risultato
      #scores.sort(reverse = True)
      max_key = max(scores, key=scores.get)
      values = list(scores.values()) #list of 'scores' values, so the real scores
      
      values.sort(reverse = True)
      if show_scores:
        print(scores)
      if values[0] > 1.25*values[1] and values[0] > 7:
        return max_key
      else:
        return 0
    else:
      return 0

#class used for writing and drawing on the image
class Annotator():
  def __init__(self, w: int = 1080, h: int = 1920):
    self.font = cv.FONT_HERSHEY_SIMPLEX
    self.org = (w//2, round(h//10*9.5))
    self.fontScale = 2
    self.color = (0, 0, 255)
    self.thickness = 3

  def write(self, img: np.ndarray, speed: int, updates: int):
    if speed == 0:
      text = 'Speed Limit: None'
    else:
      text = 'Speed Limit: ' + str(speed) + 'km/h, ' + str(updates)
    return cv.putText(img, text, self.org, self.font, self.fontScale, self.color, self.thickness, cv.LINE_AA, False)

  def reset_params(self, w: int, h: int):
    self.font = cv.FONT_HERSHEY_SIMPLEX
    self.org = (w//2, round(h//10*9.5))
    self.fontScale = 1.6
    self.color = (0, 0, 255)
    self.thickness = 3

  def draw_circles(self, img: np.ndarray, circles_small:np.ndarray, initial_dim: tuple, final_dim: tuple, crop: tuple):
    circles = circles_small.copy()
    if circles is not None:
      #starting by the assumption that we cut away the upper and lower 25% of the image and the 33% on the left
      circles[0, 0, 1] = (circles[0, 0, 1] + crop[0]) * (final_dim[0] /initial_dim[0])
      circles[0, 0, 0] = (circles[0, 0, 0] + crop[1]) * (final_dim[1] /initial_dim[1])
      #assumption: the aspect ratio is more or less the same
      circles[0, 0, 2] = circles[0, 0, 2] * (final_dim[0] / initial_dim[0])
      circles = np.uint16(np.around(circles))
      for i in circles[0,:]:
      # draw the outer circle
          cv.circle(img,(i[0],i[1]),i[2],(0,255,0),3)
      # draw the center of the circle
      #cv.circle(img,(i[0],i[1]),2,(0,0,255),1)
    return img
  
  #This function expects the points to be correct with respect to the image's size, so it's advisable
  #to first compute the points attribute with the Sign_Detector.extract_bb() method, with points being
  # a tuple of 2 tuples, the first one being the (x, y) coordinates of the top left corner of the bb, the
  #second one being the bottom right
  def draw_bb(self, img: np.ndarray, points: tuple, color = (0, 255, 255), thickness = 3):
    if points is not None:
      cv.rectangle(img = img, pt1 = points[0], pt2 = points[1], color = color, thickness = thickness)
    return img



class Sign_Detector():
  def __init__(self) -> None:
    
    if os.path.exists(RESULTS_DIR):
      shutil.rmtree(RESULTS_DIR)
    os.mkdir(RESULTS_DIR)
    if os.path.exists(SIGNS_DIR):
      shutil.rmtree(SIGNS_DIR)
    os.mkdir(SIGNS_DIR)
    self.max_radius_ratio = 9
    self.dt = Detector(maxRadiusRatio=self.max_radius_ratio)
    self.pre = Preprocessor()
    self.mat = Matcher(sift = True, path = TEMPLATES_DIR)
    self.an = Annotator()

  def detect(self, frame: np.ndarray, h, w, show_results = False) -> Union[bool, np.ndarray, int, int, Tuple]:
    if frame is None:
      return False, None, 0, 0, (0, 0)

    if frame.size == 0:
      return False, None, 0, 0, (0, 0)

    frame = frame.copy()
    original = frame.copy()
    
    speed = 0
    updates = 0
    frame = frame[h : round(h*3), w : , :] 
    #cutting away upper and lower 25% (keeping central 50%) and left 33% (keeping right 67%)
    gray = self.pre.prep(frame)
    circles = self.dt.detect(gray)
    found = True if circles is not None else False
    if found:
      sign = self.extract_sign(original, circles, h, w)
      if sign is not None:
        res = self.mat.match(sign, show_results)

      if res != 0:
        speed = res
        updates += 1

    return found, circles, speed, updates

  def extract_sign(self, img: np.ndarray, circles: np.ndarray, h, w) -> np.ndarray:
    if circles is None: #better safe than sorry
      return
    for i in circles[0,:]:
      center = (i[0] + w, i[1] + h)
      axes = (i[2], i[2])
      center = np.uint(np.around(center))
      axes = np.uint(np.around(axes))
      top_left = (center[0] - axes[0], center[1] - axes[1])
      bottom_right = (center[0] + axes[0], center[1] + axes[1])
      sign = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0], :]
      if sign.shape[0] == 0 or sign.shape[1] == 0:
        sign = None
      return sign

  def extract_bb(self, circles_small: np.ndarray, h, w) :
    points = None
    if circles_small is not None: #better safe than sorry
      circles = circles_small.copy()
      for i in circles[0,:]:
        center = (i[0] + w, i[1] + h)
        axes = (i[2], i[2])
        center = np.uint(np.around(center))
        axes = np.uint(np.around(axes))
        top_left = (center[0] - axes[0], center[1] - axes[1])
        bottom_right = (center[0] + axes[0], center[1] + axes[1])
        points = (top_left, bottom_right)
        if points[0][0] >= points[1][0] or points[0][1] >= points[1][1]:
          points = None
    return points

def save_circles_from_video(sd: Sign_Detector, img: np.ndarray, circles:np.ndarray, n_detected: int, h, w, extract = False) -> int:
  #n_detected keeps track of the n° of frames in which a traffic sign was detected
  found = False
  sign = 0
  if circles is not None:
    if extract:
      sign = sd.extract_sign(img, circles, h, w, n_detected)
    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
      # draw the outer circle
      cv.circle(img,(i[0],i[1]),i[2],(0,255,0),1)
      # draw the center of the circle
      cv.circle(img,(i[0],i[1]),2,(0,0,255),2)
    name = str(n_detected) + '.jpg'
    cv.imwrite(RESULTS_DIR + '\\' + name, img)
    n_detected += 1
    found = True
  return n_detected, found, sign


def main():
  filename = str(ROOT_DIR / 'photos' / '2.jpg')
  #filename = 'C:\\Users\\ricca\\OneDrive\\Desktop\\scazzo\\traffic\\photos\\IMG_20220731_153050.jpg'
  sd = Sign_Detector()

  frame = cv.imread(filename)

  if frame.size == 0:
    exit(0)

  height, width, _ = frame.shape
  h = height // 4
  w = width // 10*4 #~40%
  an = Annotator(width, height)
  an.org = (20, 50)
  found, circles, speed, updates = sd.detect(frame, h, w, show_results = False)
  if found:
    an.write(frame, speed, updates)
    #frame_out = draw_circles(frame, circles, (height, width), (height, width), (h, w))
    frame_out = an.draw_bb(frame, sd.extract_bb(circles, h, w))
    #frame_out = cv.resize(frame_out, (720, 720))
    cv.imshow("frame", frame_out)
    cv.waitKey(0)
    cv.destroyAllWindows()
  
  else: 
    print('Not found')
  
    

if __name__ == '__main__':
  main()

