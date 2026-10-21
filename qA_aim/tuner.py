"""
HSV 阈值调节工具 — 滑动条实时调节
使用方法: python tuner.py
"""
import cv2
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DATA_DIR


def nothing(x):
    pass


def main():
    # 视频选择
    video_path = DATA_DIR / "装甲板_蓝.mp4"
    if not video_path.exists():
        print("请将视频文件放在 data/armor/ 目录下")
        return
    
    cap = cv2.VideoCapture(str(video_path))
    
    # 创建窗口
    window_name = "HSV Tuner - 装甲板识别"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1200, 800)
    
    # 创建滑动条
    # 红色区间1
    cv2.createTrackbar("H1_low", window_name, 0, 180, nothing)
    cv2.createTrackbar("H1_high", window_name, 15, 180, nothing)
    cv2.createTrackbar("S1_low", window_name, 100, 255, nothing)
    cv2.createTrackbar("S1_high", window_name, 255, 255, nothing)
    cv2.createTrackbar("V1_low", window_name, 120, 255, nothing)
    cv2.createTrackbar("V1_high", window_name, 255, 255, nothing)
    
    # 红色区间2
    cv2.createTrackbar("H2_low", window_name, 170, 180, nothing)
    cv2.createTrackbar("H2_high", window_name, 180, 180, nothing)
    cv2.createTrackbar("S2_low", window_name, 100, 255, nothing)
    cv2.createTrackbar("S2_high", window_name, 255, 255, nothing)
    cv2.createTrackbar("V2_low", window_name, 120, 255, nothing)
    cv2.createTrackbar("V2_high", window_name, 255, 255, nothing)
    
    # 形态学参数
    cv2.createTrackbar("Dilate", window_name, 1, 5, nothing)
    
    # 蓝色模式切换
    cv2.createTrackbar("Mode(0=Red,1=Blue)", window_name, 0, 1, nothing)
    
    paused = False
    frame_idx = 0
    
    print("=" * 50)
    print("HSV 阈值调节工具")
    print("  空格键: 暂停/播放")
    print("  左右箭头: 前后跳帧")
    print("  q: 退出")
    print("  s: 截图保存当前参数")
    print("=" * 50)
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
            frame_idx += 1
        
        # 读取滑动条
        mode = cv2.getTrackbarPos("Mode(0=Red,1=Blue)", window_name)
        
        if mode == 0:  # Red
            h1_low = cv2.getTrackbarPos("H1_low", window_name)
            h1_high = cv2.getTrackbarPos("H1_high", window_name)
            s1_low = cv2.getTrackbarPos("S1_low", window_name)
            s1_high = cv2.getTrackbarPos("S1_high", window_name)
            v1_low = cv2.getTrackbarPos("V1_low", window_name)
            v1_high = cv2.getTrackbarPos("V1_high", window_name)
            
            h2_low = cv2.getTrackbarPos("H2_low", window_name)
            h2_high = cv2.getTrackbarPos("H2_high", window_name)
            s2_low = cv2.getTrackbarPos("S2_low", window_name)
            s2_high = cv2.getTrackbarPos("S2_high", window_name)
            v2_low = cv2.getTrackbarPos("V2_low", window_name)
            v2_high = cv2.getTrackbarPos("V2_high", window_name)
            
            lower1 = np.array([h1_low, s1_low, v1_low])
            upper1 = np.array([h1_high, s1_high, v1_high])
            lower2 = np.array([h2_low, s2_low, v2_low])
            upper2 = np.array([h2_high, s2_high, v2_high])
        else:  # Blue
            # 简化：复用红色滑动条位置作为蓝色参数
            h1_low = cv2.getTrackbarPos("H1_low", window_name)
            h1_high = cv2.getTrackbarPos("H1_high", window_name)
            s1_low = cv2.getTrackbarPos("S1_low", window_name)
            v1_low = cv2.getTrackbarPos("V1_low", window_name)
            
            lower1 = np.array([h1_low + 90, s1_low, v1_low])
            upper1 = np.array([h1_high + 90, 255, 255])
            lower2 = None
            upper2 = None
        
        dilate_iter = cv2.getTrackbarPos("Dilate", window_name)
        
        # 处理
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        mask1 = cv2.inRange(hsv, lower1, upper1)
        if lower2 is not None:
            mask2 = cv2.inRange(hsv, lower2, upper2)
            binary = cv2.bitwise_or(mask1, mask2)
        else:
            binary = mask1
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.dilate(binary, kernel, iterations=dilate_iter)
        
        # 显示
        vis = np.hstack([
            frame,
            cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR),
        ])
        
        # 信息
        cv2.putText(vis, f"Frame: {frame_idx}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(vis, f"Mode: {'Blue' if mode else 'Red'}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        cv2.imshow(window_name, vis)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('s'):
            # 截图
            cv2.imwrite("tuner_screenshot.png", vis)
            print(f"截图已保存: tuner_screenshot.png")
            if mode == 0:
                print(f"当前红色阈值:")
                print(f"  lower1 = np.array([{h1_low}, {s1_low}, {v1_low}])")
                print(f"  upper1 = np.array([{h1_high}, {s1_high}, {v1_high}])")
                print(f"  lower2 = np.array([{h2_low}, {s2_low}, {v2_low}])")
                print(f"  upper2 = np.array([{h2_high}, {s2_high}, {v2_high}])")
        elif key == 81:  # 左箭头
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx - 10))
            frame_idx = max(0, frame_idx - 10)
        elif key == 83:  # 右箭头
            frame_idx += 10
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
