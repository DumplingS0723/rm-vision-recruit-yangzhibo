import cv2
import numpy as np
import glob
import os

# === 配置区 (根据你的实际情况修改) ===
CHECKERBOARD = (8, 5) # 棋盘格内角点数 (宽, 高)，数一下你的图是几乘几
SQUARE_SIZE = 24.0    # 你刚才用尺子量的格子边长 (mm)

# 准备 3D 坐标 (这是固定的数学模板，不用改)
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = [] # 存放 3D 真实坐标
imgpoints = [] # 存放 2D 图像像素坐标

# 读取刚才拍的所有照片
images = glob.glob('q2_calib/calib_images/*.jpg')
print(f"找到了 {len(images)} 张照片，开始标定...")

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 转灰度图，找角点更准
    
    # 核心函数：寻找棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    
    if ret == True:
        objpoints.append(objp)
        
        # 亚像素细化 (让结果更精确，这是专业做法)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        
        # (可选) 画出来看看找没找对
        # cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        # cv2.imshow('img', img)
        # cv2.waitKey(100)

# === 核心计算 ===
# gray.shape[::-1] 是图像的 (宽, 高)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("\n=== 标定结果 ===")
print(f"重投影误差 (越小越好，通常<0.5): {ret}")
print(f"内参矩阵 (mtx):\n{mtx}")
print(f"畸变系数 (dist):\n{dist}")

# 把结果存起来，下一步测距要用
np.savez('q2_calib/calib_data.npz', mtx=mtx, dist=dist)
print("\n结果已保存到 q2_calib/calib_data.npz")
