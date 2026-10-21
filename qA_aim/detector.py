"""
装甲板检测器 — 基于颜色阈值 + 灯条配对
"""
import cv2
import numpy as np
from config import (
    HSV_THRESHOLDS, LIGHT_BAR_FILTER, LIGHT_BAR_FILTER_BLUE,
    PAIRING_CONSTRAINTS, PAIRING_CONSTRAINTS_BLUE,
    LIGHT_BAR_CENTER_DIST, LIGHT_BAR_LENGTH,
)


class ArmorDetector:
    """红色/蓝色装甲板检测器"""
    
    def __init__(self, color="red", hsv_thresholds=None, is_blue_video=False):
        """
        Args:
            color: "red" 或 "blue"
            hsv_thresholds: 自定义阈值字典（用于调参工具）
            is_blue_video: 蓝色装甲板视频 — 使用极宽松参数适配极暗画面
        """
        self.color = color
        self.thresholds = hsv_thresholds or HSV_THRESHOLDS[color]
        
        # 蓝色极暗视频：使用专属宽松参数
        if is_blue_video and color == "blue":
            self.filter = LIGHT_BAR_FILTER_BLUE
            self.pairing = PAIRING_CONSTRAINTS_BLUE
        else:
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
            mask1 = cv2.inRange(hsv, self.thresholds["lower1"], self.thresholds["upper1"])
            mask2 = cv2.inRange(hsv, self.thresholds["lower2"], self.thresholds["upper2"])
            binary = cv2.bitwise_or(mask1, mask2)
        else:
            binary = cv2.inRange(hsv, self.thresholds["lower"], self.thresholds["upper"])
        
        # 膨胀闭合小断裂（运动模糊导致的灯条断裂）
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.dilate(binary, kernel, iterations=1)

        if self.color == "blue":
        # 蓝色视频：更强的形态学处理，连接断裂灯条
            kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_large)  # 闭运算填洞
            binary = cv2.dilate(binary, kernel_large, iterations=1)           # 膨胀
        else:
        # 红色视频：原有处理
            binary = cv2.erode(binary, kernel, iterations=1)
            binary = cv2.dilate(binary, kernel, iterations=1)
        
        return binary
    
    def find_light_bars(self, binary):
        """
        从二值图中提取灯条候选
        
        Returns:
            light_bars: list of dict
        """
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        light_bars = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.filter["min_area"] or area > self.filter["max_area"]:
                continue
            
            rect = cv2.minAreaRect(cnt)
            (cx, cy), (w, h), angle = rect
            
            if w < h:
                w, h = h, w
                angle += 90
            
            aspect_ratio = w / max(h, 1.0)
            if not (self.filter["min_aspect_ratio"] <= aspect_ratio <= self.filter["max_aspect_ratio"]):
                continue
            
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / max(hull_area, 1.0)
            if solidity < self.filter["min_solidity"]:
                continue
            
            rect_area = w * h
            rectangularity = area / max(rect_area, 1.0)
            if rectangularity < self.filter["min_rectangularity"]:
                continue
            
            top_pt, bottom_pt = self._extract_endpoints(rect, angle)
            
            light_bars.append({
                "contour": cnt,
                "center": np.array([cx, cy]),
                "length": w,
                "width": h,
                "angle": angle,
                "top_pt": top_pt,
                "bottom_pt": bottom_pt,
                "rect": rect,
                "area": area,
            })
        
        return light_bars
    
    def _extract_endpoints(self, rect, angle):
        """从旋转矩形提取上下端点"""
        box = cv2.boxPoints(rect)
        
        if 45 < abs(angle) < 135:
            top_idx = np.argmin(box[:, 1])
            bottom_idx = np.argmax(box[:, 1])
        else:
            top_idx = np.argmin(box[:, 1])
            bottom_idx = np.argmax(box[:, 1])
        
        return box[top_idx], box[bottom_idx]
    
    def match_light_bars(self, light_bars, image_shape=None):
        """灯条配对：两两匹配，筛选出装甲板候选"""
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
        
        candidates.sort(key=lambda x: x["score"], reverse=True)
        
        used_indices = set()
        armors = []
        
        for i, bar in enumerate(light_bars):
            bar["_id"] = i
        
        for cand in candidates:
            id_a = cand["bar_a"]["_id"]
            id_b = cand["bar_b"]["_id"]
            if id_a in used_indices or id_b in used_indices:
                continue
            used_indices.add(id_a)
            used_indices.add(id_b)
            
            if cand["bar_a"]["center"][0] < cand["bar_b"]["center"][0]:
                left_bar, right_bar = cand["bar_a"], cand["bar_b"]
            else:
                left_bar, right_bar = cand["bar_b"], cand["bar_a"]
            
            corners = self._sort_corners(left_bar, right_bar)
            
            armors.append({
                "left_bar": left_bar,
                "right_bar": right_bar,
                "corners": corners,
                "score": cand["score"],
            })
        
        return armors
    
    def _evaluate_pair(self, bar_a, bar_b):
        """评估两个灯条的配对得分"""
        score = 0.0
        
        angle_diff = abs(bar_a["angle"] - bar_b["angle"])
        if angle_diff > 90:
            angle_diff = 180 - angle_diff
        if angle_diff > self.pairing["max_angle_diff"]:
            return 0.0, False
        score += (1.0 - angle_diff / self.pairing["max_angle_diff"]) * 3.0
        
        center_dist = np.linalg.norm(bar_a["center"] - bar_b["center"])
        avg_length = (bar_a["length"] + bar_b["length"]) / 2.0
        dist_ratio = center_dist / max(avg_length, 1.0)
        
        min_ratio = self.pairing["min_center_dist_ratio"]
        max_ratio = self.pairing["max_center_dist_ratio"]
        if not (min_ratio <= dist_ratio <= max_ratio):
            return 0.0, False
        
        ideal_ratio = LIGHT_BAR_CENTER_DIST / LIGHT_BAR_LENGTH
        ratio_score = 1.0 - abs(dist_ratio - ideal_ratio) / max_ratio
        score += ratio_score * 2.0
        
        center_vec = bar_b["center"] - bar_a["center"]
        center_angle = np.degrees(np.arctan2(center_vec[1], center_vec[0]))
        
        avg_angle = (bar_a["angle"] + bar_b["angle"]) / 2.0
        
        perpendicular_diff = abs(abs(center_angle - avg_angle) - 90)
        if perpendicular_diff > 90:
            perpendicular_diff = 180 - perpendicular_diff
        if perpendicular_diff > self.pairing["max_perpendicular_angle"]:
            return 0.0, False
        score += (1.0 - perpendicular_diff / self.pairing["max_perpendicular_angle"]) * 2.0
        
        height_ratio = max(bar_a["length"], bar_b["length"]) / max(min(bar_a["length"], bar_b["length"]), 1.0)
        if height_ratio > self.pairing["max_height_ratio"]:
            return 0.0, False
        score += (1.0 - (height_ratio - 1) / (self.pairing["max_height_ratio"] - 1)) * 1.0
        
        return score, True
    
    def _sort_corners(self, left_bar, right_bar):
        """排序四个角点为 lt, lb, rb, rt"""
        if left_bar["top_pt"][1] <= left_bar["bottom_pt"][1]:
            lt = left_bar["top_pt"]
            lb = left_bar["bottom_pt"]
        else:
            lt = left_bar["bottom_pt"]
            lb = left_bar["top_pt"]
        
        if right_bar["top_pt"][1] <= right_bar["bottom_pt"][1]:
            rt = right_bar["top_pt"]
            rb = right_bar["bottom_pt"]
        else:
            rt = right_bar["bottom_pt"]
            rb = right_bar["top_pt"]
        
        return np.array([lt, lb, rb, rt], dtype=np.float32)
    
    def detect(self, bgr_image):
        """完整检测流程"""
        binary = self.preprocess(bgr_image)
        light_bars = self.find_light_bars(binary)
        
    # 调试：打印灯条信息
        if self.color == "blue" and len(light_bars) > 0:
            print(f"[BLUE DEBUG] 找到 {len(light_bars)} 个灯条:")
            for i, bar in enumerate(light_bars):
                print(f"  灯条{i}: area={bar['area']:.1f}, "
                      f"aspect={bar['length']/max(bar['width'],1):.1f}, "
                      f"angle={bar['angle']:.1f}, "
                      f"center=({bar['center'][0]:.0f},{bar['center'][1]:.0f})")


#
        armors = self.match_light_bars(light_bars, bgr_image.shape[:2])
        return armors, binary
    
    def draw_results(self, image, armors):
        """绘制检测结果"""
        vis = image.copy()
        colors = [(0, 255, 0), (255, 255, 0), (0, 165, 255), (255, 0, 255)]
        labels = ["LT", "LB", "RB", "RT"]
        
        for armor in armors:
            corners = armor["corners"].astype(np.int32)
            
            cv2.polylines(vis, [corners], True, (0, 255, 0), 2)
            
            for i, (pt, color, label) in enumerate(zip(corners, colors, labels)):
                cv2.circle(vis, tuple(pt), 5, color, -1)
                cv2.putText(vis, f"{label}", (pt[0] + 8, pt[1] - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            left_center = tuple(armor["left_bar"]["center"].astype(np.int32))
            right_center = tuple(armor["right_bar"]["center"].astype(np.int32))
            cv2.circle(vis, left_center, 3, (255, 0, 0), -1)
            cv2.circle(vis, right_center, 3, (0, 0, 255), -1)
        
        return vis
