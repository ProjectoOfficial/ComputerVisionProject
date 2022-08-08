import numpy as np
import cv2 as cv

from types import NoneType
import os
import time
import shutil

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) # This is your Project Root
RESULTS_DIR = ROOT_DIR + '\\detected_circles'
height = 1000
width = 750



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

class Preprocessor():
  def __init__(self):
    pass
  def prep(self, img: np.ndarray) -> np.ndarray:
    #cv.imshow('Resized Image',img)
    img2 = cv.GaussianBlur(img, (3, 3), 0)
    img2 = cv.cvtColor(img2, cv.COLOR_BGR2HSV)
    temp = np.logical_and(img2[:, :, 0] > 10, img2[:, :, 0] < 170)
    #temp = np.logical_and(img2 > 10, img2 < 170)
    #print(temp.sum(), img2[temp].shape, img2.shape, f"{temp.sum()/7500.0}% of all pixels will be cast to black.")
    
    num = 0
    """
    for i in range(img2.shape[0]):
        for j in range(img2.shape[1]):
            if 10 < img2[i, j, 0] < 170:
                img2[i, j, 2] = 0
                num += 1
    
    """
    
    

    
    img2[temp, 2] = 0
    h, w, _ = img2.shape
    #img2[0:h//3, 0:w, 2] = 0
    #img2[h//3*2+1:, 0:w, 2] = 0
    
    #img2 = cv.cvtColor(img2, cv.COLOR_HSV2BGR)
    #cv.imshow('Resized, HSV-space thresholded, converted back to BGR image:', img2)
    h, w, _ = img2.shape
    #gray_img = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    gray_img = np.reshape(img2[:, :, 2], (h, w)) #from HSV to GRAYSCALE
    #cv.imshow('Gray-scaled converted image:', gray_img)
    #cv.imwrite(ROOT_DIR + '\\gray-scaled_preprocessed.jpg', gray_img)
    return gray_img



class Detector():
  def __init__(self, HOUGH_GRADIENT=True, dp=2.2, minDist = 500, param1=230, param2=80, minRadius=0, maxRadiusRatio = 20):
    self.method = cv.HOUGH_GRADIENT
    if not HOUGH_GRADIENT:
      self.method = cv.HOUGH_GRADIENT_ALT
    self.dp = dp
    self.minDist = minDist
    self.param1 = param1
    self.param2 = param2
    self.minRadius = minRadius
    self.maxRadiusRatio = maxRadiusRatio
  #single-channel image
  def detect(self, gray_img: np.ndarray, print_canny : bool= False):
    if print_canny:
      cv.imwrite(ROOT_DIR + '\\canny_edges.jpg', cv.Canny(gray_img, self.param1, self.param1//2))
    minimum = gray_img.shape[0] if gray_img.shape[0] < gray_img.shape[1] else gray_img.shape[1]
    maxR = minimum // self.maxRadiusRatio
    #print(f"Maximum Radius: {maxR}")
    circles = cv.HoughCircles(image=gray_img, method=self.method, dp=self.dp, minDist=self.minDist, param1=self.param1, param2=self.param2, minRadius=self.minRadius, maxRadius=maxR)
    return circles
    #return cv.HoughCircles(image=gray_img, method = cv.HOUGH_GRADIENT, dp=1.5, minDist=1,param1=220)

def print_circles(img: np.ndarray, circles:np.ndarray):
  if isinstance(circles, NoneType):
      print(f"Non sono stati trovati cerchi!")
  else:
    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
      # draw the outer circle
      cv.circle(img,(i[0],i[1]),i[2],(0,255,0),1)
      # draw the center of the circle
      cv.circle(img,(i[0],i[1]),2,(0,0,255),2)
  cv.imshow('Detected Circles:', img)

def save_circles_from_video(img: np.ndarray, circles:np.ndarray, n_detected: int) -> int:
  #n_detected keeps track of the n° of frames in which a traffic sign was detected
  if not isinstance(circles, NoneType):
    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
      # draw the outer circle
      cv.circle(img,(i[0],i[1]),i[2],(0,255,0),1)
      # draw the center of the circle
      cv.circle(img,(i[0],i[1]),2,(0,0,255),2)
    name = str(n_detected) + '.jpg'
    cv.imwrite(RESULTS_DIR + '\\' + name, img)
    n_detected += 1
  return n_detected

def extract_sign(img: np.ndarray, circles: np.ndarray) -> np.ndarray:
  signs = np.zeros((1, 1, 1, 1))
  if isinstance(circles, NoneType):
    return np.array([])
  
  circles = np.uint16(np.around(circles))
  for i in circles[0,:]:
    #TODO:c'è da guardare se le dimensioni che ho preso sono giuste, cioè le x sono x e non y e viceversa
    top_left = (i[0] - i[2], i[1] - i[2])
    bottom_right = (i[0] + i[2], i[1] + i[2])
    sign = img[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1], :]
    #ora sono da concatenare
    
  cv.imshow('Detected Circles:', img)

def main():
    filename = ROOT_DIR +'\\photos\\VID20220808142933.mp4'
    shutil.rmtree(RESULTS_DIR)
    os.mkdir(RESULTS_DIR)
    
    cap = cv.VideoCapture(filename) #open video
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    n_frames = 0
     # open a pointer to the video file stream

    # check if video requires rotation
    #rotateCode = check_rotation(filename)

        # now your logic can start from here
    n_detected = 0
    start_time = time.time()
    h = height//4
    w = width // 3
    max_radius_ratio = 15
    dt = Detector(maxRadiusRatio=max_radius_ratio)
    pre = Preprocessor()
    while True:
      
      # Capture frame-by-frame
      ret, frame = cap.read()
      
      # if frame is read correctly ret is True
      if not ret:
          print("Can't receive frame (stream end?). Exiting ...")
          break
      # Our operations on the frame come here
      
      #gray = cv.flip(cv.flip(cv.resize(cv.cvtColor(frame, cv.COLOR_BGR2GRAY), (750, 1000), interpolation=cv.INTER_AREA), 0), 1)
      frame = cv.flip(cv.flip(cv.resize(frame, (width, height), interpolation=cv.INTER_AREA), 0), 1)
      
      frame = frame[h: h*3, w:, :]
      gray = pre.prep(frame)
      circles = dt.detect(gray)
      #print(type(circles))
      # Display the resulting frame
      #cv.imshow('frame', gray)
      n_detected = save_circles_from_video(frame, circles, n_detected)
      n_frames += 1
    end_time = time.time()
    delta_time = round(end_time - start_time, 3)
    print(start_time, end_time)
    print(f"Processed frames: {n_frames} in {delta_time}, found at least a traffic sign in {n_detected} frames.")

    """
    img = cv.imread(filename)
    h_, w_, _ = img.shape
    #first, we cut away the upper and lower 33.3% of the image, leaving only the 33.3% in the middle
    #img = img[h_//3: h_//3 * 2, :, :]
    #now for my phone's camera, i should have 1216x2736 pixels
    img = cv.resize(img, (750, 1000), interpolation = cv.INTER_AREA) #resize while preserving aspect ratio
    
    #img2 = cv.cvtColor(cv.GaussianBlur(img, (3,3), 0), cv.COLOR_BGR2GRAY)
    img2 = pre.prep(img)
    
    circles = dt.detect(img2, print_canny=True)
    print(type(circles))
    
    print_circles(img, circles)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    """
    
    

if __name__ == '__main__':
    main()

