#import imp
import numpy as np
import cv2 as cv
import sys
import os
import shutil
from typing import Union, Tuple
from pathlib import Path
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

#from models.YOLOv7.utils.general import xyxy2xywhn
from Models.YOLOv7.utils.general import xyxy2xywhn

ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__))) # This is your Project Root
RESULTS_DIR = ROOT_DIR / 'detected_circles'
SIGNS_DIR = ROOT_DIR / 'signs'
TEMPLATES_DIR = ROOT_DIR / 'templates_solo_bordo'

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
    blur = cv.GaussianBlur(img, (3, 3), 0)
    hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)
    
    lower1 = np.array([0, 100, 20])
    upper1 = np.array([10, 255, 255])

    lower2 = np.array([160,100,20])
    upper2 = np.array([179,255,255])

    lower_mask = cv.inRange(hsv.copy(), lower1, upper1)
    upper_mask = cv.inRange(hsv.copy(), lower2, upper2)

    mask = lower_mask + upper_mask
    temp = cv.bitwise_and(hsv, hsv , mask=mask)

    blur_copy = blur.copy()
    blur_copy[np.where(temp==0)] = 0

    h, w, _ = blur.shape
    gray_filter = np.reshape(blur_copy[:, :, 0], (h, w)) #from HSV to GRAYSCALE
    
    _, binary = cv.threshold(gray_filter, 200, 255, cv.THRESH_OTSU)
    closing = cv.morphologyEx(binary, cv.MORPH_CLOSE, np.ones((4, 4), np.uint8), iterations=1)
    dilated = cv.dilate(closing, np.ones((15, 15), np.uint8), iterations=4)

    gray = np.reshape(blur[:, :, 0], (h, w))
    masked = cv.bitwise_and(gray, gray, mask=dilated) 
    contours, hierarchy = cv.findContours(masked, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    cnt = 0
    best_area = 0
    for contour in contours:
      if best_area < cv.contourArea(contour):
        best_area = cv.contourArea(contour)
        cnt = contour

    out = np.zeros_like(masked) # Extract out the object and place into output image
    topx = 0
    topy = 0
    if len(contours) > 0:
      cnt_mask = np.zeros_like(masked) # Create mask where white is what we want, black otherwise
      cv.drawContours(cnt_mask, cnt, -1, 255, -1) # Draw filled contour in mask
      out[cnt_mask == 255] = masked[cnt_mask == 255]

      # Now crop
      (y, x) = np.where(cnt_mask == 255)
      (topy, topx) = (np.min(y), np.min(x))
      (bottomy, bottomx) = (np.max(y), np.max(x))
      out = img[topy:bottomy+1, topx:bottomx+1, 0]

    return out, (topx, topy) # out, h, w

"""
Detector class, in which we perform the Canny Edge Detection and the Circles detection via the Hough Transform
"""
class Detector():
  def __init__(self, HOUGH_GRADIENT=True, dp=2.2, minDist = 500, param1=230, param2=95, minRadiusRatio=15, maxRadiusRatio = 4):
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
  
  def detect(self, gray: np.ndarray, print_canny : bool= False):
    if print_canny:
      cv.imwrite(ROOT_DIR + '\\canny_edges.jpg', cv.Canny(gray, self.param1, self.param1//2))
    
    expanded = False
    if gray.shape[0] < 65 or gray.shape[1] < 65:
      expanded = True
      gray = cv.resize(gray, (gray.shape[0]*2, gray.shape[1]*2))
    minimum = gray.shape[0] if gray.shape[0] < gray.shape[1] else gray.shape[1]

    maxR = round(minimum / self.maxRadiusRatio)
    minR = round(minimum / self.minRadiusRatio)
    circles = cv.HoughCircles(image=gray.copy(), method=self.method, dp=self.dp, minDist=self.minDist, param1=self.param1, param2=self.param2, minRadius=minR, maxRadius=maxR)
    
    if expanded and circles is not None:
      circles[:,:,:] /= 2
    
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
    self.num = [10, 20, 30, 40, 50, 60, 70, 80, 90] #traffic signs
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
          pth = os.path.join(dir, file)  
          if os.path.isfile(pth):
            img = cv.imread(pth)
            for dim in self.dims:
              for blur in self.blurs:
                img = cv.resize(img, dim, interpolation = cv.INTER_AREA)
                img = cv.GaussianBlur(img, blur, 0)
                gray = img[:,:,0]
                _, gray = cv.threshold(gray, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
                gray = cv.bitwise_not(gray)
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
    gray = sign[:,:,0]
    _, binary_inv = cv.threshold(gray, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    binary = cv.bitwise_not(binary_inv)

    det = cv.SIFT_create() if self.sift else cv.ORB_create()
    kp = det.detect(binary, None)
    # compute the descriptors with ORB
    kp, key_ps = det.compute(binary, kp)
    alpha = 0.8
    knn = 2
    if key_ps is not None:
      scores = {}
      for i in range(len(self.num)):
        best = 0
        for features in self.features[i]:
          matches = self.bf.knnMatch(features, key_ps, k=knn)
          good = []
          for j in range(len(matches)):
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
      if values[0] >= 1.20*values[1] and values[0] > 4:
        return max_key
    return 0

#class used for writing and drawing on the image
class Annotator():
  def __init__(self, w: int = 1080, h: int = 1920):
    self.font = cv.FONT_HERSHEY_SIMPLEX
    self.org = (w//2, round(h//10*9.5))
    self.fontScale = 1
    self.color = (0, 0, 255)
    self.thickness = 2

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
    self.dt = Detector()
    self.pre = Preprocessor()
    self.mat = Matcher(sift = True, path = TEMPLATES_DIR)
    self.an = Annotator()

  def detect(self, img: np.ndarray, h_perc, w_perc, show_results = False) -> Union[bool, np.ndarray, int, int, Tuple]:
    if img is None:
      return False, None, 0, 0, (0, 0)

    if img.size == 0:
      return False, None, 0, 0, (0, 0)

    frame = img.copy()
    original = img.copy()

    speed = 0
    updates = 0
    h, w, _ = frame.shape
    h = (h * h_perc) // 100
    w = (w * w_perc) //100
    frame = frame[h : frame.shape[0] - h, w : , :] 
    #cutting away upper and lower 25% (keeping central 50%) and left 33% (keeping right 67%)
    gray, (h_margin, w_margin) = self.pre.prep(frame)
    circles = self.dt.detect(gray)
    found = True if circles is not None else False
    if found:
      circles[:,:,0] += h_margin
      circles[:,:,1] += w_margin
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
      
      #margin = (i[2] - (i[2] * (math.pi/4))) / 4
      #axes = (i[2]* (math.pi/4) + margin, i[2] * (math.pi/4) + margin)
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
  
  def extract_bb_yolo(self, circles_small: np.ndarray, h, w, real_h, real_w) :
    """
    Parameters:
    circles_small = the circles detected on the pre-processed image
    h, w = the values of the cropping in height and width
    real_h, real_w = the dimensions of the original image, needed for the normalization of the BB
    """
    points = None
    if circles_small is not None: #better safe than sorry
      circles = circles_small.copy()
      for i in circles[0,:]:
        center = (i[0] + w, i[1] + h)
        axes = (i[2], i[2])
        center = np.uint(np.around(center))
        axes = np.uint(np.around(axes))
        points = np.array((center[0] - axes[0], center[1] - axes[1], center[0] + axes[0], center[1] + axes[1]), dtype=np.float32)
        if points[0] >= points[2] or points[1] >= points[3]:
          points = None
    return xyxy2xywhn(x=np.expand_dims(points, axis=0), w = real_w, h = real_h)
  
  

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
  filename = os.path.join(ROOT_DIR, 'photos', '0.jpg')
  #filename = 'C:\\Users\\ricca\\OneDrive\\Desktop\\scazzo\\traffic\\photos\\IMG_20220731_153050.jpg'
  sd = Sign_Detector()

  frame = cv.imread(filename)

  if frame.size == 0:
    exit(0)

  height, width, _ = frame.shape

  h_perc = 5
  w_perc = 50
  
  h = (height * h_perc) // 100
  w = (width * w_perc) // 100

  an = Annotator(width, height)
  an.org = (20, 50)
  found, circles, speed, updates = sd.detect(frame, h_perc, w_perc, show_results = False)
  if found:
    an.write(frame, speed, updates)
    #frame_out = draw_circles(frame, circles, (height, width), (height, width), (h, w))
    frame_out = an.draw_bb(frame, sd.extract_bb(circles, h, w))
    #frame_out = cv.resize(frame_out, (720, 720))
    """
    Test of function with BBs in numpy:
    """
    print(sd.extract_bb_yolo(circles_small = circles, h = h, w = w, real_h = height, real_w = width))
    cv.imshow("frame", frame_out)
    cv.waitKey(0)
    cv.destroyAllWindows()
  
  else: 
    print('Not found')
  

if __name__ == '__main__':
  main()

