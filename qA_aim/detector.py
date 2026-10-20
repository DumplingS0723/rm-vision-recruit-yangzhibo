"""
装甲板检测器 — 基于颜色阈值 + 灯条配对
"""
import cv2
import numpy as np
from config import (
    HSV_THRESHOLDS, LIGHT_BAR_FILTER, PAIRING_CONSTRAINTS,
    LIGHT_BAR_CENTER_DIST, LIGHT_BAR_LENGTH,
)


class ArmorDetector:
    """红色/蓝色装甲板检测器"""
    
    def __init__(self, color="red", hsv_thresholds=None):
        """
        Args:
            color: "red" 或 "blue"
            hsv_thresholds: 自定义阈值字典（用于调参工具）
        """
        self.color = color
        self.thresholds = hsv_thresholds or HSV_THRESHOLDS[color]
        self.filter = LIGHT_BAR_FILTER
        self.pairing = PAIRING_CONSTRAINTS
        
    def preprocess(self, bgr_image):
        """
        预处理：高斯模糊 → HSV转换 → 颜色阈值 → 膨胀
        
        Returns:
            binary: 二值掩码
        """
        blurred = cv2.GaussianBlur(bgr_image, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        if self.color == "red":
            # 红色需要合并两个区间
            mask1 = cv2.inRange(hsv, self.thresholds["lower1"], self.thresholds["upper1"])
            mask2 = cv2.inRange(hsv, self.thresholds["lower2"], self.thresholds["upper2"])
            binary = cv2.bitwise_or(mask1, mask2)
        else:
            binary = cv2.inRange(hsv, self.thresholds["lower"], self.thresholds["upper"])
        
        # 膨胀闭合小断裂（运动模糊导致的灯条断裂）
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.dilate(binary, kernel, iterations=1)
        
        return binary
    
    def find_light_bars(self, binary):
        """
        从二值图中提取灯条候选
        
        Returns:
            light_bars: list of dict, 每个 dict 包含:
                contour, center, length, width, angle, 
                top_pt, bottom_pt, rect, area
        """
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        light_bars = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.filter["min_area"] or area > self.filter["max_area"]:
                continue
            
            # 最小外接旋转矩形
            rect = cv2.minAreaRect(cnt)
            (cx, cy), (w, h), angle = rect
            
            # 确保 w 为长边（长度），h 为短边（宽度）
            if w < h:
                w, h = h, w
                angle += 90
            
            # 宽高比筛选
            aspect_ratio = w / max(h, 1.0)
            if not (self.filter["min_aspect_ratio"] <= aspect_ratio <= self.filter["max_aspect_ratio"]):
                continue
            
            # 凸性筛选
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / max(hull_area, 1.0)
            if solidity < self.filter["min_solidity"]:
                continue
            
            # 矩形度筛选
            rect_area = w * h
            rectangularity = area / max(rect_area, 1.0)
            if rectangularity < self.filter["min_rectangularity"]:
                continue
            
            # 提取端点
            top_pt, bottom_pt = self._extract_endpoints(rect, angle)
            
            light_bars.append({
                "contour": cnt,
                "center": np.array([cx, cy]),
                "length": w,
                "width": h,
                "angle": angle,          # 在 OpenCV 坐标系中（x右y下）
                "top_pt": top_pt,
                "bottom_pt": bottom_pt,
                "rect": rect,
                "area": area,
            })
        
        return light_bars
    
    def _extract_endpoints(self, rect, angle):
        """从旋转矩形提取上下端点"""
        box = cv2.boxPoints(rect)  # 4 个角点，顺序不定
        
        # 在 OpenCV 图像坐标系（y向下）中：
        # "上" = y 较小，"下" = y 较大
        # 对于近乎垂直的灯条，用y区分；对于水平的，用x区分
        if 45 < abs(angle) < 135:
            # 灯条接近垂直 → 用 y 区分上下
            top_idx = np.argmin(box[:, 1])
            bottom_idx = np.argmax(box[:, 1])
        else:
            # 灯条接近水平 → 用 x 区分上下（此时上=左或右，具体看角度符号）
            # 但实际装甲板灯条都是近似垂直的，这里做容错
            top_idx = np.argmin(box[:, 1])
            bottom_idx = np.argmax(box[:, 1])
        
        return box[top_idx], box[bottom_idx]
    
    def match_light_bars(self, light_bars, image_shape=None):
        """
        灯条配对：两两匹配，筛选出装甲板候选
        
        Returns:
            armors: list of dict, 每个包含:
                left_bar, right_bar, corners (lt,lb,rb,rt), score
        """
        if len(light_bars) < 2:
            return []
        
        candidates = []
        n = len(light_bars)
        
        for i in range(n):
            for j in range(i + 1, n):
                bar_a = light_bars[i]
                bar_b = light_bars[j]
                
                score, is_valid = self._evaluate_pair(bar_a, bar_b)
                if is_valid:
                    candidates.append({
                        "bar_a": bar_a,
                        "bar_b": bar_b,
                        "score": score,
                    })
        
        if not candidates:
            return []
        
        # 按评分排序，贪心选取互不冲突的配对
        candidates.sort(key=lambda x: x["score"], reverse=True)
        
        used_indices = set()
        armors = []
        
        # 为灯条建立索引（用id标识）
        for i, bar in enumerate(light_bars):
            bar["_id"] = i
        
        for cand in candidates:
            id_a = cand["bar_a"]["_id"]
            id_b = cand["bar_b"]["_id"]
            if id_a in used_indices or id_b in used_indices:
                continue
            used_indices.add(id_a)
            used_indices.add(id_b)
            
            # 确定左右灯条（左灯条 x 较小）
            if cand["bar_a"]["center"][0] < cand["bar_b"]["center"][0]:
                left_bar, right_bar = cand["bar_a"], cand["bar_b"]
            else:
                left_bar, right_bar = cand["bar_b"], cand["bar_a"]
            
            # 提取四个角点并按 lt, lb, rb, rt 排序
            corners = self._sort_corners(left_bar, right_bar)
            
            armors.append({
                "left_bar": left_bar,
                "right_bar": right_bar,
                "corners": corners,
                "score": cand["score"],
            })
        
        return armors
    
    def _evaluate_pair(self, bar_a, bar_b):
        """评估两个灯条的配对得分，返回 (score, is_valid)"""
        score = 0.0
        
        # 1. 角度一致性
        angle_diff = abs(bar_a["angle"] - bar_b["angle"])
        # 处理角度环绕（如 5° 和 175° 实际差 10°）
        if angle_diff > 90:
            angle_diff = 180 - angle_diff
        if angle_diff > self.pairing["max_angle_diff"]:
            return 0.0, False
        score += (1.0 - angle_diff / self.pairing["max_angle_diff"]) * 3.0
        
        # 2. 间距合理性
        center_dist = np.linalg.norm(bar_a["center"] - bar_b["center"])
        avg_length = (bar_a["length"] + bar_b["length"]) / 2.0
        dist_ratio = center_dist / max(avg_length, 1.0)
        
        min_ratio = self.pairing["min_center_dist_ratio"]
        max_ratio = self.pairing["max_center_dist_ratio"]
        if not (min_ratio <= dist_ratio <= max_ratio):
            return 0.0, False
        
        # 距离越接近理想值（135/55 ≈ 2.45）得分越高
        ideal_ratio = LIGHT_BAR_CENTER_DIST / LIGHT_BAR_LENGTH  # ≈ 2.45
        ratio_score = 1.0 - abs(dist_ratio - ideal_ratio) / max_ratio
        score += ratio_score * 2.0
        
        # 3. 中心连线与灯条方向的垂直性
        center_vec = bar_b["center"] - bar_a["center"]
        center_angle = np.degrees(np.arctan2(center_vec[1], center_vec[0]))
        
        # 灯条方向角（取两灯条均值）
        avg_angle = (bar_a["angle"] + bar_b["angle"]) / 2.0
        
        # 中心连线应垂直于灯条方向
        perpendicular_diff = abs(abs(center_angle - avg_angle) - 90)
        if perpendicular_diff > 90:
            perpendicular_diff = 180 - perpendicular_diff
        if perpendicular_diff > self.pairing["max_perpendicular_angle"]:
            return 0.0, False
        score += (1.0 - perpendicular_diff / self.pairing["max_perpendicular_angle"]) * 2.0
        
        # 4. 高度比
        height_ratio = max(bar_a["length"], bar_b["length"]) / max(min(bar_a["length"], bar_b["length"]), 1.0)
        if height_ratio > self.pairing["max_height_ratio"]:
            return 0.0, False
        score += (1.0 - (height_ratio - 1) / (self.pairing["max_height_ratio"] - 1)) * 1.0
        
        return score, True
    
    def _sort_corners(self, left_bar, right_bar):
        """
        排序四个角点为 lt, lb, rb, rt：
        - 左灯条的上端点 = lt，下端点 = lb
        - 右灯条的下端点 = rb，上端点 = rt
        """
        # 左灯条：区分上下
        if left_bar["top_pt"][1] <= left_bar["bottom_pt"][1]:
            lt = left_bar["top_pt"]
            lb = left_bar["bottom_pt"]
        else:
            lt = left_bar["bottom_pt"]
            lb = left_bar["top_pt"]
        
        # 右灯条：区分上下
        if right_bar["top_pt"][1] <= right_bar["bottom_pt"][1]:
            rt = right_bar["top_pt"]
            rb = right_bar["bottom_pt"]
        else:
            rt = right_bar["bottom_pt"]
            rb = right_bar["top_pt"]
        
        return np.array([lt, lb, rb, rt], dtype=np.float32)
    
    def detect(self, bgr_image):
        """
        完整检测流程
        
        Returns:
            armors: list of dict
            binary: 预处理后的二值图（用于可视化）
        """
        binary = self.preprocess(bgr_image)
        light_bars = self.find_light_bars(binary)
        armors = self.match_light_bars(light_bars, bgr_image.shape[:2])
        return armors, binary
    
    def draw_results(self, image, armors):
        """绘制检测结果"""
        vis = image.copy()
        colors = [(0, 255, 0), (255, 255, 0), (0, 165, 255), (255, 0, 255)]
        labels = ["LT", "LB", "RB", "RT"]
        
        for armor in armors:
            corners = armor["corners"].astype(np.int32)
            
            # 绘制四边形
            cv2.polylines(vis, [corners], True, (0, 255, 0), 2)
            
            # 绘制角点和标签
            for i, (pt, color, label) in enumerate(zip(corners, colors, labels)):
                cv2.circle(vis, tuple(pt), 5, color, -1)
                cv2.putText(vis, f"{label}", (pt[0] + 8, pt[1] - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # 绘制左右灯条标记
            left_center = tuple(armor["left_bar"]["center"].astype(np.int32))
            right_center = tuple(armor["right_bar"]["center"].astype(np.int32))
            cv2.circle(vis, left_center, 3, (255, 0, 0), -1)
            cv2.circle(vis, right_center, 3, (0, 0, 255), -1)
        
        return vis