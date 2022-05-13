import cv2
import numpy as np

CAMERA_DEVICE = 0
TRESH_MODE = "ADAPTIVE_GAUSSIAN" # OTSU ADAPTIVE_GAUSSIAN ADAPTIVE_MEAN


if __name__ == "__main__":
    capture = cv2.VideoCapture(CAMERA_DEVICE)
    capture.set(cv2.CAP_PROP_EXPOSURE, 20) 

    while True:
        ret, img = capture.read()

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        th = None

        if TRESH_MODE == "OTSU":
            ret, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif TRESH_MODE == "ADAPTIVE_GAUSSIAN":
            th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 4)
        elif TRESH_MODE == "ADAPTIVE_MEAN":
            th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

        abs_64 = np.absolute(cv2.Sobel(th, cv2.CV_64F, 1, 0, ksize=3))
        sobel_8u = np.uint8(abs_64)

        rgb_th = cv2.cvtColor(th ,cv2.COLOR_GRAY2RGB)
        rgb_sobel = cv2.cvtColor(sobel_8u ,cv2.COLOR_GRAY2RGB)
        vertical_stack = np.hstack((img, rgb_th, rgb_sobel))

        cv2.imshow("images", vertical_stack)
        if cv2.waitKey(10) == ord('q'):
            break

capture.release()
cv2.destroyAllWindows()