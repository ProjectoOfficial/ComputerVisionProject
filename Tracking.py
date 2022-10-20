from re import sub
import cv2
import numpy as np
import torch

from Models.YOLOv7.utils.general import xywh2xyxy
class Tracking:
    def __init__(self):
        self.objects = dict() # ID : [label, actual bbox, prev bbox, IOU, updated]
        self.trackers = dict() # ID : Kalman filter
        self.ids = 0

        self.term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
        self.s_lower = 60
        self.s_upper = 255
        self.v_lower = 32
        self.v_upper = 255

    def track(self, hsvframe , bbox, id):
        x, y, w, h = bbox
        roi = hsvframe[y: y + h, x : x + w]
        #cv2.imshow("ROI", cv2.resize(roi, (w*2, h*2)))

        mask = cv2.inRange(roi, np.array((0., float(self.s_lower), float(self.v_lower))), np.array((180., float(self.s_upper), float(self.v_upper))))
        roi_hist = cv2.calcHist([roi], [0, 1], mask, [180, 255], [0, 180, 0, 255])
        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

        track_window = (x, y, x + w, y + h)

        back = cv2.calcBackProject([hsvframe], [0, 1], roi_hist, [0, 180, 0, 255], 1)
        ret, track_window = cv2.CamShift(back, track_window, self.term_crit)
        
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)

        self.trackers[id].correct(self.center(pts))
        prediction = self.trackers[id].predict()

        return prediction, pts

    def center(self, points):
        x = np.float32((points[0][0] + points[1][0] + points[2][0] + points[3][0]) / 4.0)
        y = np.float32( (points[0][1] + points[1][1] + points[2][1] + points[3][1]) / 4.0)
        return np.array([np.float32(x), np.float32(y)], np.float32)

    def xywh2xyxy(self, x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[0] = x[0] - x[2] / 2  # top left x
        y[1] = x[1] - x[3] / 2  # top left y
        y[2] = x[0] + x[2] / 2  # bottom right x
        y[3] = x[1] + x[3] / 2  # bottom right y
        return y

    def box_iou(self, boxA, boxB):
    	# determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
        # return the intersection over union value
        return iou

    def zero_objects(self):
        '''
        zeroing the state
        '''
        if len(self.objects) > 0:
            for key in self.objects:
                self.objects[key][4] = 0

    def clear_objects(self):
        if len(self.objects) > 0:
            keys_to_remove = []
            for key in self.objects:
                if self.objects[key][4] == 0:
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                self.objects.pop(key, None)
                self.trackers.pop(key, None)
        else:
            self.ids = 0

    def kalman_create(self):
        kalman = cv2.KalmanFilter(4, 2)
        kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                            [0, 1, 0, 0]], np.float32)

        kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                            [0, 1, 0, 1],
                                            [0, 0, 1, 0],
                                            [0, 0, 0, 1]], np.float32)

        kalman.processNoiseCov = np.array([[1, 0, 0, 0],
                                        [0, 1, 0, 0],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]], np.float32) * 0.03
        return kalman

    def update_obj(self, cls, bbox):
        subkey = -1
        subval = 0
        iou = 0
        for key in self.objects:
            obj = self.objects[key] # cls actualbbox prevbbox iou updated
            if cls != obj[0]:
                continue

            prev_bbox = self.xywh2xyxy(obj[1])
            actual_bbox = self.xywh2xyxy(bbox)
            
            iou = self.box_iou(prev_bbox, actual_bbox)
            if subval < iou:
                subval = iou
                subkey = key
        
        id = -1
        if subkey != -1: # object already exists
            self.objects[subkey] = [self.objects[subkey][0], bbox, self.objects[subkey][1], iou, 1]
            id = subkey
        else: # add the new object
            self.ids += 1
            self.objects[self.ids] = [cls, bbox, bbox, 1, 1]
            id = self.ids
            self.trackers[id] = self.kalman_create()

        return id

        