__author__ = "Daniel Rossi, Riccardo Salami, Filippo Ferrari"
__copyright__ = "Copyright 2022"
__credits__ = ["Daniel Rossi", "Riccardo Salami", "Filippo Ferrari"]
__license__ = "GPL-3.0"
__version__ = "1.0.0"
__maintainer__ = "Daniel Rossi"
__email__ = "miniprojectsofficial@gmail.com"
__status__ = "Computer Vision Exam"

import cv2

# distance from camera to object(face) measured
# centimeter
Known_distance = 76.2

# width of face in the real world or Object Plane
# centimeter
Known_width = 14.3

# Colors
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# face detector object
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


class Distance(object):

    @staticmethod
    def get_Distance(frame):
        # reading reference_image from directory
        ref_image = cv2.imread("Ref_image.png")

        # find the face width(pixels) in the reference_image
        ref_image_face_width = Distance.face_data(ref_image)

        # get the focal by calling "Focal_Length_Finder"
        # face width in reference(pixels),
        # Known_distance(centimeters),
        # known_width(centimeters)
        Focal_length_found = Distance.Focal_Length_Finder(
            Known_distance, Known_width, ref_image_face_width)

        face_width_in_frame = Distance.face_data(frame)

        if face_width_in_frame != 0:
            # finding the distance by calling function
            # Distance finder function need
            # these arguments the Focal_Length,
            # Known_width(centimeters),
            # and Known_distance(centimeters)
            Distance_ = Distance.Distance_finder(
                Focal_length_found, Known_width, face_width_in_frame)

        return Distance_

    # focal length finder function
    def Focal_Length_Finder(measured_distance, real_width, width_in_rf_image):
        # finding the focal length
        focal_length = (width_in_rf_image * measured_distance) / real_width
        return focal_length


    # distance estimation function
    def Distance_finder(Focal_Length, real_face_width, face_width_in_frame):
        distance = (real_face_width * Focal_Length) / face_width_in_frame

        # return the distance
        return distance


    def face_data(image):

        face_width = 0  # making face width to zero

        # converting color image to gray scale image
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # detecting face in the image
        faces = face_detector.detectMultiScale(gray_image, 1.3, 5)

        # looping through the faces detect in the image
        # getting coordinates x, y , width and height
        for (x, y, h, w) in faces:
            # draw the rectangle on the face
            cv2.rectangle(image, (x, y), (x + w, y + h), GREEN, 2)

            # getting face width in the pixels
            face_width = w

        # return the face width in pixel
        return face_width
