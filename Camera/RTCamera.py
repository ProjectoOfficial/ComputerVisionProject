'''
(C) Dott. Daniel Rossi - Università degli Studi di Modena e Reggio Emilia
    Computer Vision Project - Artificial Intelligence Engineering

    Real Time Camera class

    Camera model: See3Cam_CU27 REV X1
'''

import cv2
from threading import Thread
import time

class RTCamera(object):
    def __init__(self, src:int=0, fps:float=60, resolution:tuple=(1920, 1080)):

        self.src            = src
        self.cap            = cv2.VideoCapture(self.src, cv2.CAP_ANY)
        self.frame          = None

        self.resolution     = resolution
        self.FPS            = fps
        self.FPS_MS         = int(1/self.FPS * 1000)

        self.fps_frames     = 5
        self.fps_times      = []

        self.record         = False
        self.fourcc         = cv2.VideoWriter_fourcc(*'mp4v')
        self.output         = None

        self.thread         = None
        self.thread_alive   = False

        self.has_calibration    = False
        self.mtx = None
        self.dist = None
        self.rvecs = None
        self.tvecs = None

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 100)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_CONVERT_RGB , 1)

        self.cap.set(cv2.CAP_PROP_EXPOSURE, -15)
        self.cap.set(cv2.CAP_PROP_GAIN, -50)
        

    def start(self):
        self.thread_alive = True
        self.thread = Thread(target=self.__update, args=())
        self.thread.start()

    def stop(self):
        if self.record:
            self.output.release()
            self.record = False

        self.thread_alive = False
        self.thread.join()
        self.cap.release()

    def __update(self):
        while True:
            if self.cap.isOpened():
                (self.ret, self.frame) = self.cap.read()

                if self.record:
                    self.output.write(self.frame)

                if len(self.fps_times) <= self.fps_frames:
                    self.fps_times.append(time.time())
            
            if not self.thread_alive:
                break

    def get_frame(self):
        cv2.waitKey(self.FPS_MS)

        if self.frame is None:
            return None

        return self.frame.copy() if self.has_calibration is False else self.__adjust_frame()

    def available(self):
        return self.frame is not None

    def register(self, filename:str):
        self.output = cv2.VideoWriter(filename, self.fourcc, self.FPS, self.resolution)
        self.record = True

    def save_frame(self, path):
        cv2.imwrite(path, self.frame.copy())

    def get_resolution(self):
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        heigth = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("RESOLUTION: {}x{}".format(width, heigth))

    def get_fps(self):
        if len(self.fps_times) >= self.fps_frames:
            total = 0

            for i in range(self.fps_frames - 1):
                total += self.fps_times[i + 1] - self.fps_times[i]
            
            self.fps_times = []
            return 1//(total/(self.fps_frames-1))
        return 0
        
    def set_exposure(self, exp:int):
        try:
            assert exp < 15 and exp > -15 and isinstance(exp, int)
            self.cap.set(cv2.CAP_PROP_EXPOSURE, exp)
        except AssertionError:
            print("exposure must be an integer number in the range (15, -15)")

    def set_gain(self, gain:int):
        try:
            assert isinstance(gain, int)
            self.cap.set(cv2.CAP_PROP_GAIN, gain)
        except AssertionError:
            print("gain must be an integer number ")

    def calibrate(self, calibrated, mtx, dist, rvecs, tvecs):
        self.has_calibration = calibrated
        self.mtx = mtx
        self.dist = dist
        self.rvecs = rvecs
        self.tvecs = tvecs

    def __adjust_frame(self):        
        h, w = self.frame.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (w,h), 1, (w,h))
        mapx, mapy = cv2.initUndistortRectifyMap(self.mtx, self.dist, None, newcameramtx, (w,h), 5)
        dst = cv2.undistort(self.frame, self.mtx, self.dist, None, newcameramtx)
        # crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]

        return dst.copy()