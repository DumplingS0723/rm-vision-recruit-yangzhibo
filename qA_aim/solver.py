"""
PnP 位姿解算器
"""
import cv2
import numpy as np
from config import LIGHT_BAR_LENGTH, LIGHT_BAR_CENTER_DIST, PNP_CONFIG


class PnPSolver:
    """基于装甲板4个角点求解位姿"""
    
    def __init__(self, K, dist=None):
        """
        Args:
            K: 相机矩阵 (3x3)
            dist: 畸变系数 [k1,k2,p1,p2,k3] 或 None
        """
        self.K = K
        self.dist = dist
        
        # 3D 装甲板模型点（在装甲板坐标系中）
        # 原点 = 装甲板中心
        # x轴向右, y轴向下, z轴向前
        half_w = LIGHT_BAR_CENTER_DIST / 2.0  # 67.5 mm
        half_h = LIGHT_BAR_LENGTH / 2.0       # 27.5 mm
        
        self.object_points = np.array([
            [-half_w, -half_h, 0],  # LT: 左上
            [-half_w,  half_h, 0],  # LB: 左下
            [ half_w,  half_h, 0],  # RB: 右下
            [ half_w, -half_h, 0],  # RT: 右上
        ], dtype=np.float32)
        
        # 方法映射
        method_map = {
            "iterative": cv2.SOLVEPNP_ITERATIVE,
            "epnp": cv2.SOLVEPNP_EPNP,
            "p3p": cv2.SOLVEPNP_P3P,
            "ap3p": cv2.SOLVEPNP_AP3P,
        }
        self.method = method_map.get(PNP_CONFIG["method"], cv2.SOLVEPNP_ITERATIVE)
        
    def solve(self, image_points, use_distortion=True):
        """
        PnP 解算
        
        Args:
            image_points: 4个角点像素坐标 [lt, lb, rb, rt], shape (4,2)
            use_distortion: 是否使用畸变系数
        
        Returns:
            dict: {"position": (x,y,z), "yaw": float, "success": bool,
                   "rvec": ..., "tvec": ..., "reproj_error": float}
                   失败返回 None
        """
        if len(image_points) != 4:
            return None
        
        image_points = np.array(image_points, dtype=np.float32).reshape(4, 2)
        
        dist = self.dist if use_distortion else None
        
        try:
            success, rvec, tvec = cv2.solvePnP(
                self.object_points,
                image_points,
                self.K,
                dist,
                flags=self.method,
                useExtrinsicGuess=PNP_CONFIG["use_extrinsic_guess"],
            )
            
            if not success:
                return None
            
            # 计算重投影误差
            reproj, _ = cv2.projectPoints(
                self.object_points, rvec, tvec, self.K, dist
            )
            reproj_error = np.linalg.norm(
                image_points.reshape(-1, 2) - reproj.reshape(-1, 2), axis=1
            ).mean()
            
            if reproj_error > PNP_CONFIG["max_reproj_error"]:
                return None
            
            # 提取 yaw（绕 y 轴的旋转角，因为相机坐标系中 y 向下）
            R, _ = cv2.Rodrigues(rvec)
            yaw = self._extract_yaw(R)
            
            return {
                "position": tuple(tvec.flatten()),  # (x, y, z) in meters
                "yaw": yaw,                         # radians
                "rvec": rvec,
                "tvec": tvec,
                "R": R,
                "reproj_error": reproj_error,
                "success": True,
            }
            
        except cv2.error:
            return None
    
    def solve_compare_distortion(self, image_points):
        """
        对比使用/不使用畸变系数的结果差异
        
        Returns:
            dict with both results and difference
        """
        result_with = self.solve(image_points, use_distortion=True)
        result_without = self.solve(image_points, use_distortion=False)
        
        if result_with is None or result_without is None:
            return None
        
        pos_diff = np.linalg.norm(
            np.array(result_with["position"]) - np.array(result_without["position"])
        )
        yaw_diff = abs(result_with["yaw"] - result_without["yaw"])
        
        # 计算图像点距离主点的平均距离（判断在中心还是边缘）
        principal_point = np.array([self.K[0, 2], self.K[1, 2]])
        mean_dist_to_center = np.linalg.norm(
            image_points.reshape(-1, 2) - principal_point, axis=1
        ).mean()
        
        return {
            "with_distortion": result_with,
            "without_distortion": result_without,
            "pos_diff_m": pos_diff,
            "yaw_diff_rad": yaw_diff,
            "mean_dist_to_center_px": mean_dist_to_center,
        }
    
    def _extract_yaw(self, R):
        """
        从旋转矩阵提取 yaw 角（绕相机坐标系y轴旋转）
        R: 3x3 旋转矩阵，世界→相机
        yaw = atan2(R[2,0], R[0,0])
        
        注意：这里需要根据实际坐标系约定调整。
        装甲板自身z轴朝前（离开装甲板平面），相机z轴也朝前。
        yaw 表示装甲板法向量在 xz 平面上的旋转。
        """
        yaw = np.arctan2(R[2, 0], R[0, 0])
        return float(yaw)