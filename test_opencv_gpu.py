import numpy as np
import cv2
import time

data = np.random.random((1024, 1024)).astype(np.float32)
mat = np.stack([data, data], axis=2)

cudaMat1 = cv2.cuda_GpuMat()
cudaMat2 = cv2.cuda_GpuMat()
cudaMat1.upload(mat)
cudaMat2.upload(mat)

start_time = time.time()
cv2.cuda.gemm(cudaMat1, cudaMat2, 1, None, 0, None, 1)
print("GPU: \t {}s ".format(time.time() - start_time))

start_time = time.time()
cv2.gemm(mat, mat, 1, None, 0, None, 1)
print("CPU: \t {}s ".format(time.time() - start_time))