'''
(C) Dott. Daniel Rossi - Università degli Studi di Modena e Reggio Emilia
    Computer Vision Project

    Real Time Camera class

    Camera model: See3Cam_CU27 REV X1
'''

import cv2
from threading import Thread
import time

class RTCamera(object):
    def __init__(self, src:int=0, fps:float=1/60, resolution:tuple=(1920, 1080)):

        self.src            = src
        self.cap            = cv2.VideoCapture(self.src, cv2.CAP_ANY )
        self.frame          = None

        self.resolution     = resolution
        self.FPS            = fps
        self.FPS_MS         = int(self.FPS * 1000)

        self.fps_frames = 5
        self.fps_times = []

        self.thread         = None
        self.thread_alive   = False

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 100)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

        self.cap.set(cv2.CAP_PROP_FPS, 100)
        self.cap.set(cv2.CAP_PROP_CONVERT_RGB , 1)

        self.cap.set(cv2.CAP_PROP_EXPOSURE, -5)
        

    def start(self):
        self.thread_alive = True
        self.thread = Thread(target=self.__update, args=())
        self.thread.start()

    def stop(self):
        self.thread_alive = False
        self.thread.join()
        self.cap.release()

    def __update(self):
        while True:
            if self.cap.isOpened():
                (self.ret, self.frame) = self.cap.read()
                if len(self.fps_times) <= self.fps_frames:
                    self.fps_times.append(time.time())

            
            if not self.thread_alive:
                break

    def get_frame(self):
        cv2.waitKey(self.FPS_MS)
        return self.frame

    def available(self):
        return self.frame is not None

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
        