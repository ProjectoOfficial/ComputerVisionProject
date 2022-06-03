'''
(C) Dott. Daniel Rossi - Università degli Studi di Modena e Reggio Emilia
    Computer Vision Project

    Real Time Camera script for road analysis

    Camera model: See3Cam_CU27 REV X1
'''

import cv2
import time
import numpy as np
import keyboard

from RTCamera import RTCamera

CAMERA_DEVICE = 0
TRESH_MODE = "ADAPTIVE_GAUSSIAN" # OTSU ADAPTIVE_GAUSSIAN ADAPTIVE_MEAN


def processing(img: np.ndarray):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        th = None

        if TRESH_MODE == "OTSU":
            ret, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif TRESH_MODE == "ADAPTIVE_GAUSSIAN":
            th = cv2.adaptqiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 19, 4)
        elif TRESH_MODE == "ADAPTIVE_MEAN":
            th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

        canny = cv2.Canny(blur, 50, 5)

        rgb_th = cv2.cvtColor(th ,cv2.COLOR_GRAY2RGB)
        rgb_canny = cv2.cvtColor(canny ,cv2.COLOR_GRAY2RGB)
        rgb_blur = cv2.cvtColor(blur, cv2.COLOR_GRAY2RGB)

        H_stack = np.hstack((rgb_blur, rgb_th, rgb_canny))

        cv2.imshow("images", H_stack)


if __name__ == "__main__":
    camera = RTCamera(CAMERA_DEVICE)
    camera.start()

    start_fps = time.time()
    fps = 0
    while True:
        frame = camera.get_frame()
        if camera.available():
            cv2.putText(frame, str(fps) + " fps", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2, cv2.LINE_AA)
            frame = cv2.resize(frame, (640,480))
            cv2.imshow("frame", frame)

            if time.time() - start_fps > 2:
                fps = camera.get_fps()
                start_fps = time.time()

            if keyboard.is_pressed('e'):
                exp = int(input("please insert the exposure: "))
                camera.set_exposure(exp)

            if keyboard.is_pressed('g'):
                gain = int(input("please insert the gain: "))
                camera.set_gain(gain)

            if keyboard.is_pressed('r'):
                camera.register("output.mp4")
            
        if keyboard.is_pressed('q'):
            break

    camera.stop()
    cv2.destroyAllWindows()

