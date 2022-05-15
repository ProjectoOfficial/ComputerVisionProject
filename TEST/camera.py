import cv2
import numpy as np

CAMERA_DEVICE = 0
TRESH_MODE = "ADAPTIVE_GAUSSIAN" # OTSU ADAPTIVE_GAUSSIAN ADAPTIVE_MEAN


if __name__ == "__main__":
    capture = cv2.VideoCapture(CAMERA_DEVICE)

    capture.set(cv2.CAP_PROP_EXPOSURE, 20) 
    capture.set(cv2.CAP_PROP_FPS, 30)
    capture.set(cv2.CAP_PROP_GAIN, 20)
    capture.set(cv2.CAP_PROP_CONVERT_RGB, 1)  

    while True:
        ret, img = capture.read()

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        th = None

        if TRESH_MODE == "OTSU":
            ret, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif TRESH_MODE == "ADAPTIVE_GAUSSIAN":
            th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 19, 4)
        elif TRESH_MODE == "ADAPTIVE_MEAN":
            th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

        canny = cv2.Canny(blur, 50, 5)

        rgb_th = cv2.cvtColor(th ,cv2.COLOR_GRAY2RGB)
        rgb_canny = cv2.cvtColor(canny ,cv2.COLOR_GRAY2RGB)
        rgb_blur = cv2.cvtColor(blur, cv2.COLOR_GRAY2RGB)

        H_stack = np.hstack((rgb_blur, rgb_th, rgb_canny))

        cv2.imshow("images", H_stack)
        if cv2.waitKey(10) == ord('q'):
            break

capture.release()
cv2.destroyAllWindows()