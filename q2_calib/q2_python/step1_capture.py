import cv2
import os

# 创建一个文件夹存照片
save_dir = 'q2_calib/calib_images'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 打开摄像头 (0 代表默认摄像头)
cap = cv2.VideoCapture(0)

print("=== 拍照模式 ===")
print("按 [空格键] 拍照")
print("按 [q] 退出")

count = 0
while True:
    ret, frame = cap.read() # 读取一帧画面
    if not ret:
        break

    # 显示画面
    cv2.imshow('Camera', frame)

    # 等待按键
    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '): # 如果按下空格
        count += 1
        filename = os.path.join(save_dir, f'calib_{count}.jpg')
        cv2.imwrite(filename, frame)
        print(f"已保存第 {count} 张照片: {filename}")
    
    elif key == ord('q'): # 如果按下 q
        break

cap.release()
cv2.destroyAllWindows()
print("采集结束！")