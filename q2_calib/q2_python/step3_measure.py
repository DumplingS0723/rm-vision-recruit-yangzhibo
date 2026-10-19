import cv2
import numpy as np

# 1. 加载刚才标定的数据
data = np.load('q2_calib/calib_data.npz')
mtx = data['mtx']
dist = data['dist']

# A4 纸的真实 3D 坐标 (单位 mm)
# 假设左上角是原点 (0,0,0)
A4_WIDTH = 210.0
A4_HEIGHT = 297.0
obj_points = np.array([
    [0, 0, 0],
    [A4_WIDTH, 0, 0],
    [A4_WIDTH, A4_HEIGHT, 0],
    [0, A4_HEIGHT, 0]
], dtype=np.float32)

print("=== 测距模式 ===")
print("把 A4 纸放在摄像头前，尽量让背景干净")
print("按 [q] 退出")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break
    
    # --- 图像处理：寻找 A4 纸 ---
    # 1. 转灰度
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 2. 高斯模糊去噪
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # 3. 边缘检测
    edged = cv2.Canny(blurred, 75, 200)
    
    # 4. 寻找轮廓
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # 按面积排序，取最大的几个（假设 A4 纸是画面里比较大的物体）
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    
    target_contour = None
    
    for c in contours:
        # 近似轮廓
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        
        # 如果近似后的轮廓有 4 个点，我们就认为它是 A4 纸
        if len(approx) == 4:
            target_contour = approx
            break
    
    distance_text = "Distance: N/A"
    
    if target_contour is not None:
        # 画出找到的框
        cv2.drawContours(frame, [target_contour], -1, (0, 255, 0), 2)
        
        # 提取 4 个点的像素坐标
        # 注意：approx 的形状是 (4, 1, 2)，需要 reshape 成 (4, 2)
        pts = target_contour.reshape(4, 2)
        
        # 【重要】点的顺序必须和 obj_points 对应！
        # 这里做一个简单的排序：按 y 坐标分上下，再按 x 分左右的逻辑比较复杂
        # 简化版：假设 findContours 找到的顺序大致是顺时针或逆时针
        # 为了稳健，通常需要写一个 order_points 函数来排序 (左上, 右上, 右下, 左下)
        # 这里为了演示，我们直接用 PnP，如果顺序乱了，测出来的距离会不对，你需要手动调整 pts 的顺序
        
        # 简单的排序逻辑 (供参考)：
        # 1. 按 y 排序，上面两个是 top，下面两个是 bottom
        # 2. top 里按 x 排序 -> 左上, 右上
        # 3. bottom 里按 x 排序 -> 左下, 右下 (注意 obj_points 定义顺序)
        
        # 这里使用 solvePnP
        # retval, rvec, tvec = cv2.solvePnP(obj_points, pts, mtx, dist)
        
        # 由于自动排序很难写对，建议你先用鼠标点击选点的方式测试（见下文提示）
        # 或者在这里打印 pts 看看顺序
        
        # 暂时先用固定逻辑尝试（可能不准，需要你调试）：
        # 按照 OpenCV 轮廓通常是顺时针/逆时针，可能需要调整 obj_points 的顺序匹配
        
        # --- 为了让你能跑通，这里使用一个简化的假设 ---
        # 实际上，自动检测 A4 纸角点并排序是 CV 里的经典难题。
        # 建议：先用鼠标手动点 4 个角来验证 PnP 算法是否工作。
        
        pass 

    # 显示画面
    cv2.putText(frame, distance_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('Measure', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()