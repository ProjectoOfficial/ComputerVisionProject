__author__ = "Daniel Rossi, Riccardo Salami, Filippo Ferrari"
__copyright__ = "Copyright 2022"
__credits__ = ["Daniel Rossi", "Riccardo Salami", "Filippo Ferrari"]
__license__ = "GPL-3.0"
__version__ = "1.0.0"
__maintainer__ = "Filippo Ferrari"
__email__ = "miniprojectsofficial@gmail.com"
__status__ = "Computer Vision Exam"

import cv2
import numpy as np
from typing import List
# distance from camera to object(face) measured
# centimeter
Known_distance = 76.2

# width of face in the real world or Object Plane
# centimeter
Known_width = 14.3

GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


class Distance(object):
    ''' Distance class uses a face as a reference object for calculating distances from generic objects, given a frame and a list of bounding boxes
        Reference distance unit is centimeters
    '''

    def __init__(self):
        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def get_Distances(self, bounding_boxes: np.ndarray) -> List[int]:
        ref_image = cv2.imread(r"Camera/Ref_image.png")

        gray_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
        face = self.face_detector.detectMultiScale(gray_image, 1.3, 5)
        (_, _, _, ref_image_width) = face.ravel()
        Focal_length_found = self.Focal_Length_Finder(Known_distance, Known_width, ref_image_width)

        distances = []
        for bbox in bounding_boxes:
            (_, _, _, frame_obj_width) = bbox

            if frame_obj_width != 0:
                distances.append(self.Distance_finder(Focal_length_found, Known_width, frame_obj_width))
            else:
                distances.append(0)

        return distances

    # focal length finder function
    def Focal_Length_Finder(self, measured_distance, real_width, width_in_rf_image) -> float:
        # finding the focal length
        focal_length = (width_in_rf_image * measured_distance) / real_width
        return focal_length


    # distance estimation function
    def Distance_finder(self, Focal_Length, real_width, frame_obj_width) -> float:
        distance = (real_width * Focal_Length) / frame_obj_width

        # return the distance
        return distance

