import cv2

CAMERA_DEVICE = 0
TRESH_MODE = "ADAPTIVE_MEAN" # OTSU ADAPTIVE_GAUSSIAN ADAPTIVE_MEAN


if __name__ == "__main__":
    capture = cv2.VideoCapture(CAMERA_DEVICE)
    capture.set(cv2.CAP_PROP_EXPOSURE, 30) 

    while True:
        ret, img = capture.read()

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        th = None

        if TRESH_MODE == "OTSU":
            ret, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif TRESH_MODE == "ADAPTIVE_GAUSSIAN":
            th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        elif TRESH_MODE == "ADAPTIVE_MEAN":
            th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

        cv2.imshow("original", img)
        cv2.imshow("original", th)
        if cv2.waitKey(10) == ord('q'):
            break

capture.release()
cv2.destroyAllWindows()