__author__ = "Daniel Rossi, Riccardo Salami, Filippo Ferrari"
__copyright__ = "Copyright 2022"
__credits__ = ["Daniel Rossi", "Riccardo Salami", "Filippo Ferrari"]
__license__ = "GPL-3.0"
__version__ = "1.0.0"
__maintainer__ = "Daniel Rossi"
__email__ = "miniprojectsofficial@gmail.com"
__status__ = "Computer Vision Exam"

from pydoc import resolve
import cv2
from threading import Thread
import time
import numpy as np

class RTCamera(object):
    '''
        This class is used to manage the camera
    '''

    def __init__(self, src:int=0, fps:float=60, resolution:tuple=(1920, 1080), cuda : bool = False, auto_exposure : bool = False):

        self.src            = src
        self.cap            = cv2.VideoCapture(self.src)

        self.frame          = None

        self.cuda           = cuda
        self.auto_exposure  = auto_exposure

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

        self.exposure = 0

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 100)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_CONVERT_RGB , 1)

        self.cap.set(cv2.CAP_PROP_EXPOSURE, self.exposure)
        self.cap.set(cv2.CAP_PROP_GAIN, 0)

        self.EXPOSURE_RANGE = range(60, 75)
        

    def start(self):
        '''
        it automatically tries to adjust the exposure of the camera then starts the camera thread
        ''' 
        if self.cuda: # user chose to use cuda
            self.cuda = self._check_cuda_support(verbose=True) # we check if cuda is available
            if self.cuda: # if cuda us really available, then we can use it
                self.frame = cv2.cuda_GpuMat() 

        if self.auto_exposure:       
            self.__adjust_exposure()

        self.thread_alive = True
        self.thread = Thread(target=self.__update, args=())
        self.thread.start()

    def stop(self):
        '''
        it stops the camera thread and releases the camera. If recording was started, it saves the file and close it
        '''
        if self.record:
            self.output.release()
            self.record = False

        self.thread_alive = False
        self.thread.join()
        self.cap.release()

    @classmethod
    def _check_cuda_support(cls, verbose:bool = False) -> bool:
        try:
            devices = cv2.cuda.getCudaEnabledDeviceCount()

            if devices <= 0:
                if verbose:
                    print("Cannot find any GPU available")
                return False
            else:
                cv2.cuda.setDevice(0)
                if verbose:
                    print(cv2.cuda.printShortCudaDeviceInfo(0))
                return True
        except:
            if verbose:
                print("Sorry but OpenCV documentation is a joke")
            return False

    def __update(self):
        '''
        this function updates the current frame captured by the camera
        '''
        while True:
            if self.cap.isOpened():
                (ret, frame) = self.cap.read()
                if self.cuda:
                    self.frame.upload(frame)
                else:
                    self.frame = frame
                        
                if self.record:
                    self.output.write(self.frame)

                if len(self.fps_times) <= self.fps_frames:
                    self.fps_times.append(time.time())
                
            if not self.thread_alive:
                break

    def get_frame(self):
        '''
        this method returns the current frame
        '''
        cv2.waitKey(self.FPS_MS)
        
        if self.frame is None:
            return None

        if self.has_calibration:
            self.calibrate()

        if isinstance(self.frame, cv2.cuda_GpuMat):
            dst = self.frame.download()
            return dst

        if self.frame.shape[0] <= 0 or self.frame.shape[1] <= 0:
            return None
        
        return self.frame.copy()

    def available(self):
        '''
        it returns if the camera is available
        '''
        return self.frame is not None

    def register(self, filename:str):
        '''
        this method starts the video recording 
        '''
        self.output = cv2.VideoWriter(filename, self.fourcc, self.FPS, self.resolution)
        self.record = True

    def save_frame(self, path:str):
        '''
        this method saves the current frame
        '''
        cv2.imwrite(path, self.frame.copy())

    def get_resolution(self):
        '''
        this method returns the frame proportions used for acquiring the current frame and eventually for recording
        '''
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        heigth = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("RESOLUTION: {}x{}".format(width, heigth))

    def get_fps(self):
        '''
        this method calculates the framerate
        '''
        if len(self.fps_times) >= self.fps_frames:
            total = 0

            for i in range(self.fps_frames - 1):
                total += self.fps_times[i + 1] - self.fps_times[i]
            
            self.fps_times = []
            return 1//(total/(self.fps_frames-1))
        return 0
        
    def set_exposure(self, exp:int):
        '''
        this method allows to manually set the camera exposure
        '''
        try:
            assert exp < 15 and exp > -15 and isinstance(exp, int)
            self.cap.set(cv2.CAP_PROP_EXPOSURE, exp)
        except AssertionError:
            print("exposure must be an integer number in the range (15, -15)")

    def set_gain(self, gain:int):
        '''
        this method allows to manually set the camera gain (introduces noise)
        '''
        try:
            assert isinstance(gain, int)
            self.cap.set(cv2.CAP_PROP_GAIN, gain)
        except AssertionError:
            print("gain must be an integer number ")

    def calibrate(self, calibrated, mtx, dist, rvecs, tvecs):
        '''
        this method is a setter for calibration parameters
        '''
        self.has_calibration = calibrated
        self.mtx = mtx
        self.dist = dist
        self.rvecs = rvecs
        self.tvecs = tvecs

    def read_calibrated(self):
        '''
        this method takes the calibration parameters and calibrates the current frame
        '''
        if not self.cuda:
            h, w = self.frame.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (w,h), 1, (w,h))
            mapx, mapy = cv2.initUndistortRectifyMap(self.mtx, self.dist, None, newcameramtx, (w,h), 5)
            dst = cv2.undistort(self.frame, self.mtx, self.dist, None, newcameramtx)
            # crop the image
            x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        self.frame = cv2.cuda.resize(dst, self.resolution) if self.cuda else cv2.resize(dst, self.resolution)

    def __adjust_exposure(self):
        '''
        this method automatically tries to adjust the exposure
        '''
        start_time = time.time()
        
        read_interval = 0.5
        last_read = time.time()

        while True:
            if self.cap.isOpened():
                if time.time() - last_read > read_interval:
                    (ret, frame) = self.cap.read()
                    last_read = time.time()

                    avg = 0
                    if self.cuda:
                        n = np.array((frame.shape[0], frame.shape[1], frame.shape[2])).prod()
                        sum = cv2.cuda.sum(frame)
                        sum = np.array(sum).sum()
                        avg = sum / n
                    else:
                        avg = round(frame.mean(axis=(0, 1, 2)))

                    if avg not in list(self.EXPOSURE_RANGE):
                        print(avg)
                        if avg < self.EXPOSURE_RANGE[0]:
                            self.exposure +=1

                        if avg > self.EXPOSURE_RANGE[-1]:
                            self.exposure -=1
                        
                        self.cap.set(cv2.CAP_PROP_EXPOSURE, self.exposure)
                        
                    else:
                        print("Exposure adjusted at {}".format(self.exposure))
                        break

            if time.time() - start_time > 15:
                print("could not calibrate camera exposure: {}".format(self.exposure))
                break
    
    @staticmethod
    def rotate_image(image: np.ndarray, angle: int) -> np.ndarray:
        (h, w) = image.shape[:2]
        center = (w//2, h//2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)

        result = None
        if RTCamera._check_cuda_support():
            result = cv2.cuda_GpuMat()
            result.upload(image)
            result = cv2.cuda.warpAffine(src=result, M=rot_mat, dsize=(w, h), flags=cv2.INTER_LINEAR).download()
        else:
            result = cv2.warpAffine(src=image, M=rot_mat, dsize=(h, w), flags=cv2.INTER_LINEAR)
        return result