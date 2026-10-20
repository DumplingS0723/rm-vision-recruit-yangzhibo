"""
装甲板识别与运动预测 — 配置文件
"""
import json
import numpy as np
from pathlib import Path

# ==================== 路径配置 ====================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data" / "armor"
OUTPUT_DIR = BASE_DIR / "outputs"
FIGURES_DIR = BASE_DIR / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ==================== 装甲板物理参数 ====================
# 小装甲板：灯条长 55mm，两灯条中心距 135mm
LIGHT_BAR_LENGTH = 0.055       # 灯条长度 (m)
LIGHT_BAR_CENTER_DIST = 0.135  # 两灯条中心间距 (m)
LIGHT_BAR_WIDTH = 0.010        # 灯条宽度估约值 (m)，用于 PnP 3D 模型

# ==================== 相机参数 ====================
def load_camera_params(camera_info_path=None):
    """加载相机内参和畸变系数"""
    if camera_info_path is None:
        camera_info_path = DATA_DIR / "camera_info.json"
    
    with open(camera_info_path, "r") as f:
        info = json.load(f)
    
    K = np.array(info["camera_matrix"]).reshape(3, 3)        # 相机矩阵 (3x3)
    dist = np.array(info["distortion_coefficients"])            # 畸变系数 [k1, k2, p1, p2, k3]
    resolution = [info.get("image_width", 1440), info.get("image_height", 1080)]
    
    return K, dist, resolution

# ==================== 视频配置 ====================
VIDEO_CONFIGS = {
    "远距离平移": {
        "fps": 150,
        "color": "red",
        "has_pose": True,
        "gyro": False,
    },
    "中距离陀螺": {
        "fps": 150,
        "color": "red",
        "has_pose": True,
        "gyro": True,
    },
    "近距离陀螺": {
        "fps": 150,
        "color": "red",
        "has_pose": True,
        "gyro": True,
    },
    "装甲板_蓝": {
        "fps": 24,
        "color": "blue",
        "has_pose": False,
        "gyro": False,
    },
}

# ==================== HSV 颜色阈值 ====================
# 红色在 HSV 中跨越 0/180 边界，需要两个区间
# 区间1: H ∈ [0, 15]    — 偏橙红
# 区间2: H ∈ [170, 180] — 偏紫红
HSV_THRESHOLDS = {
    "red": {
        "lower1": np.array([0, 44, 221]),
        "upper1": np.array([59, 255, 255]),
        "lower2": np.array([170, 43, 106]),
        "upper2": np.array([180, 255, 255]),
    },
    "blue": {
        "lower": np.array([100, 100, 80]),
        "upper": np.array([135, 255, 255]),
    },
}

# ==================== 灯条筛选参数 ====================
LIGHT_BAR_FILTER = {
    "min_area": 50,           # 最小轮廓面积 (px²)
    "max_area": 5000,         # 最大轮廓面积 (px²)
    "min_aspect_ratio": 2.0,  # 最小长宽比（细长条）
    "max_aspect_ratio": 8.0,  # 最大长宽比
    "min_solidity": 0.4,      # 最小凸性（轮廓面积/凸包面积）
    "min_rectangularity": 0.5, # 最小矩形度
}

# ==================== 灯条配对约束 ====================
PAIRING_CONSTRAINTS = {
    "max_angle_diff": 10.0,                 # 两灯条最大角度差 (度)
    "max_center_dist_ratio": 5.0,           # 中心距/灯条平均长度 最大值
    "min_center_dist_ratio": 0.5,
    "max_perpendicular_angle": 15.0,        # 中心连线与灯条垂直方向最大偏差 (度)
    "max_height_ratio": 1.5,                # 两灯条长度比最大值
}

# ==================== PnP 解算参数 ====================
PNP_CONFIG = {
    "method": "iterative",  # cv2.SOLVEPNP_ITERATIVE
    "use_extrinsic_guess": False,
    "max_reproj_error": 8.0,  # 最大允许重投影误差 (px)
}

# ==================== 卡尔曼滤波器参数 ====================
# 普通 KF（用于平移目标预测）
# 状态: [x, y, z, vx, vy, vz, ax, ay, az]
KF_CONFIG = {
    "dt": 1.0 / 150,  # 时间步长
    "Q_position": 1e-4,    # 位置过程噪声
    "Q_velocity": 1e-2,    # 速度过程噪声
    "Q_acceleration": 1e-1, # 加速度过程噪声
    "R_position": 1e-3,    # 位置观测噪声
    "predict_time": 0.25,   # 预测时间 (s)
}

# EKF（用于小陀螺目标估计）
# 状态: [cx, cy, cz, omega, theta, r, vz]
#   cx,cy,cz: 车体中心 (相机坐标系)
#   omega:    旋转角速度 (rad/s)
#   theta:    当前相位角 (rad)，装甲板相对于车体中心的角度
#   r:        旋转半径 (m)，装甲板到车体中心的距离
#   vz:       车体z方向速度 (m/s)，理论上小陀螺为0
EKF_CONFIG = {
    "dt": 1.0 / 150,
    # 初始状态不确定性
    "init_P": np.diag([0.1, 0.1, 0.5, 0.5, 0.5, 0.05, 0.1]),
    # 过程噪声协方差
    "Q": np.diag([1e-6, 1e-6, 1e-6,   # cx,cy,cz 应近乎不变
                   1e-3,               # omega 允许缓慢变化
                   1e-4,               # theta 随旋转快速变化
                   1e-6,               # r 不变
                   1e-6]),             # vz 近乎0
    # 观测噪声协方差
    "R": np.diag([1e-4, 1e-4, 1e-3]),  # x,y,z 观测
    "predict_time": 0.25,
}

# ==================== 可视化配置 ====================
VIS_CONFIG = {
    "output_fps": 30,       # 输出视频帧率
    "figure_dpi": 150,
    "keypoint_colors": [(0, 255, 0), (255, 255, 0), (0, 165, 255), (255, 0, 255)],
    "keypoint_order_names": ["LT", "LB", "RB", "RT"],
}
