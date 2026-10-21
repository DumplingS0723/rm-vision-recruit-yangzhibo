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
LIGHT_BAR_LENGTH = 0.055       # 灯条长度 (m)
LIGHT_BAR_CENTER_DIST = 0.135  # 两灯条中心间距 (m)
LIGHT_BAR_WIDTH = 0.010        # 灯条宽度估约值 (m)

# ==================== 相机参数 ====================
def load_camera_params(camera_info_path=None):
    if camera_info_path is None:
        camera_info_path = DATA_DIR / "camera_info.json"
    
    with open(camera_info_path, "r") as f:
        info = json.load(f)
    
    K = np.array(info["camera_matrix"]).reshape(3, 3)
    dist = np.array(info["distortion_coefficients"])
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
HSV_THRESHOLDS = {
    "red": {
        "lower1": np.array([0, 100, 120]),
        "upper1": np.array([15, 255, 255]),
        "lower2": np.array([170, 100, 120]),
        "upper2": np.array([180, 255, 255]),
    },
    "blue": {
        # 极宽松蓝色阈值，适配极暗视频（V均值仅11.2）
        "lower": np.array([0, 50, 86]),
        "upper": np.array([10, 0, 0]),
    },
}

# ==================== 灯条筛选参数（红色/默认） ====================
LIGHT_BAR_FILTER = {
    "min_area": 50,
    "max_area": 5000,
    "min_aspect_ratio": 2.0,
    "max_aspect_ratio": 8.0,
    "min_solidity": 0.4,
    "min_rectangularity": 0.5,
}

# ==================== 灯条筛选参数（蓝色专用，极暗视频） ====================
LIGHT_BAR_FILTER_BLUE = {
    "min_area": 10,
    "max_area": 5000,
    "min_aspect_ratio": 1.2,
    "max_aspect_ratio": 12.0,
    "min_solidity": 0.25,
    "min_rectangularity": 0.25,
}

# ==================== 灯条配对约束（红色/默认） ====================
PAIRING_CONSTRAINTS = {
    "max_angle_diff": 15.0,
    "max_center_dist_ratio": 5.0,
    "min_center_dist_ratio": 0.3,
    "max_perpendicular_angle": 30.0,
    "max_height_ratio": 2.0,
}

# ==================== 灯条配对约束（蓝色专用） ====================
PAIRING_CONSTRAINTS_BLUE = {
    "max_angle_diff": 15.0,
    "max_center_dist_ratio": 8.0,
    "min_center_dist_ratio": 0.3,
    "max_perpendicular_angle": 25.0,
    "max_height_ratio": 2.0,
}

# ==================== PnP 解算参数 ====================
PNP_CONFIG = {
    "method": "iterative",
    "use_extrinsic_guess": False,
    "max_reproj_error": 8.0,
}

# ==================== 卡尔曼滤波器参数 ====================
KF_CONFIG = {
    "dt": 1.0 / 150,
    "Q_position": 1e-4,
    "Q_velocity": 1e-2,
    "Q_acceleration": 1e-1,
    "R_position": 1e-3,
    "predict_time": 0.25,
}

EKF_CONFIG = {
    "dt": 1.0 / 150,
    "init_P": np.diag([0.1, 0.1, 0.5, 0.5, 0.5, 0.05, 0.1]),
    "Q": np.diag([1e-6, 1e-6, 1e-6,
                   1e-3,
                   1e-4,
                   1e-6,
                   1e-6]),
    "R": np.diag([1e-4, 1e-4, 1e-3]),
    "predict_time": 0.25,
}

# ==================== 可视化配置 ====================
VIS_CONFIG = {
    "output_fps": 30,
    "figure_dpi": 150,
    "keypoint_colors": [(0, 255, 0), (255, 255, 0), (0, 165, 255), (255, 0, 255)],
    "keypoint_order_names": ["LT", "LB", "RB", "RT"],
}
