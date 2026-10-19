import cv2
import numpy as np

# 加载数据
data = np.load('q2_calib/calib_data.npz')
mtx, dist = data['mtx'], data['dist']

# A4 真实坐标
obj_pts = np.array([[0,0,0], [210,0,0], [210,297,0], [0,297,0]], dtype=np.float32)

# 读取一张照片
img = cv2.imread('q2_calib/60degree.jpg')
pts = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pts.append([x, y])
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow('Image', img)
        print(f"Point {len(pts)}: ({x}, {y})")
        
        if len(pts) == 4:
            # 凑齐 4 个点，开始计算
            img_pts = np.array(pts, dtype=np.float32)
            
            # PnP 解算
            ret, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, mtx, dist)
            
            # 计算距离 (tvec 是平移向量，单位是 mm)
            distance_mm = np.linalg.norm(tvec)
            distance_m = distance_mm / 1000.0
            
            print(f"=== 计算结果 ===")
            print(f"距离: {distance_m:.2f} 米")
            
            # 拿尺子量一下真实距离，看看准不准！

cv2.imshow('Image', img)
cv2.setMouseCallback('Image', click_event)
print("请依次点击 A4 纸的: 左上 -> 右上 -> 右下 -> 左下")
cv2.waitKey(0)
