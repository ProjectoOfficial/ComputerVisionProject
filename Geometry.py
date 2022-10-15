import cv2
import numpy as np
import glob
from tqdm import tqdm

class Geometry(object):
    '''
        This class contains all the geometry transformation used inside this project:
            - Camera calibration
    '''

    def __init__(self, path, checkboard = (9, 6)):
        self.path = path

        self.checkboard = checkboard

        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.obj_points = np.zeros((self.checkboard[0] * self.checkboard[1], 3), np.float32)
        self.obj_points[:, :2] = np.mgrid[0: self.checkboard[0], 0:self.checkboard[1]].T.reshape(-1, 2)

        self.object_points = []
        self.image_points = []

        self.calibrated = False
        self.mtx = None
        self.dist = None
        self.rvecs = None
        self.tvecs = None

        self.images = glob.glob(self.path + r'\*.jpg')

    def get_calibration(self):
        '''
        this method calculates the camera calibration parameters through images previously acquired by the user by using chessboard method
        '''
        gray = None
        for filename in tqdm(self.images, desc="Calibrating camera"):
            img = cv2.imread(filename)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            find_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK
            ret, corners = cv2.findChessboardCorners(gray, self.checkboard, find_flags)

            if ret:
                self.object_points.append(self.obj_points)

                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                self.image_points.append(corners2)
                
                cv2.drawChessboardCorners(img, self.checkboard, corners, ret)
                cv2.imshow('image', img)
                cv2.waitKey(500)
                
        cv2.destroyAllWindows()
        
        if gray is None:
            return False, None, None, None, None
            
        self.calibrated, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(self.object_points, self.image_points, gray.shape[::-1], None, None)
        
        mean_error = 0
        for i in range(len(self.object_points)):
            imgpoints2, _ = cv2.projectPoints(self.object_points[i], self.rvecs[i], self.tvecs[i], self.mtx, self.dist)
            error = cv2.norm(self.image_points[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            mean_error += error
        print( "total error: {}".format(mean_error/len(self.object_points)) )

        return self.calibrated, self.mtx, self.dist, self.rvecs, self.tvecs