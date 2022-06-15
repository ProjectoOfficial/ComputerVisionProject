import cv2
import numpy as np
import glob

class Geometry(object):

    def __init__(self):
        self.checkboard = (6, 9)

        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.obj_points = np.zeros((self.checkboard[0]*self.checkboard[1], 3), np.float32)
        self.obj_points[:, :2] = np.mgrid[0: self.checkboard[1], 0:self.checkboard[0]].T.reshape(-1, 2)

        self.object_points = []
        self.image_points = []

        self.images = glob.glob('*.jpg')

    def Calibrate(self, frame):
        for filename in self.images:
            img = cv2.imread(filename)
            gray = cv2.cvtColor(cv2.COLOR_BGR2GRAY)

            # find chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, self.checkboard, None)

            if ret:
                self.object_points.append(self.obj_points)

                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                cv2.drawChessboardCorners(img, self.checkboard, corners2, ret)
                cv2.imshow('image', img)
                cv2.waitKey(500)
                
