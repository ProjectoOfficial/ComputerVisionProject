import numpy as np
import cv2
import time

data = np.zeros((1024, 1024, 3)).astype(np.float32)
print(data.shape)

cap = cv2.VideoCapture(0)
ret, frame = cap.read()

cudaMat1 = cv2.cuda_GpuMat()
cudaMat1.upload(frame)

start_time = time.time()
cv2.cuda.resize(cudaMat1, (612, 612))
print("GPU: \t {}s ".format(time.time() - start_time))

start_time = time.time()
cv2.resize(frame, (612, 612))
print("CPU: \t {}s ".format(time.time() - start_time))

img = cudaMat1.download()
print(img.shape)
cv2.imshow("frame", img)
cv2.waitKey(0)