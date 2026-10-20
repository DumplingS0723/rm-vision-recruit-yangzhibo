"""
运动跟踪器 — 普通KF（平移）+ EKF（小陀螺）
"""
import numpy as np
from config import KF_CONFIG, EKF_CONFIG


class LinearKalmanFilter:
    """
    恒定加速度 (CA) 卡尔曼滤波器
    状态: [x, y, z, vx, vy, vz, ax, ay, az]  (9维)
    观测: [x, y, z]  (3维)
    """
    
    def __init__(self, dt=None, Q_config=None, R_config=None):
        self.dt = dt or KF_CONFIG["dt"]
        
        # 状态维度
        self.state_dim = 9
        self.obs_dim = 3
        
        # 初始化状态
        self.x = np.zeros((self.state_dim, 1))
        self.P = np.eye(self.state_dim) * 100.0  # 初始不确定性大
        self.initialized = False
        
        # 状态转移矩阵 (CA模型)
        dt = self.dt
        self.F = np.eye(self.state_dim)
        self.F[0, 3] = dt;  self.F[0, 6] = 0.5 * dt**2  # x = x + vx*dt + 0.5*ax*dt²
        self.F[1, 4] = dt;  self.F[1, 7] = 0.5 * dt**2
        self.F[2, 5] = dt;  self.F[2, 8] = 0.5 * dt**2
        self.F[3, 6] = dt   # vx = vx + ax*dt
        self.F[4, 7] = dt
        self.F[5, 8] = dt
        # ax,ay,az 保持不变 (随机游走)
        
        # 观测矩阵
        self.H = np.zeros((self.obs_dim, self.state_dim))
        self.H[0, 0] = 1; self.H[1, 1] = 1; self.H[2, 2] = 1
        
        # 过程噪声协方差
        q_p = Q_config.get("Q_position", KF_CONFIG["Q_position"]) if Q_config else KF_CONFIG["Q_position"]
        q_v = Q_config.get("Q_velocity", KF_CONFIG["Q_velocity"]) if Q_config else KF_CONFIG["Q_velocity"]
        q_a = Q_config.get("Q_acceleration", KF_CONFIG["Q_acceleration"]) if Q_config else KF_CONFIG["Q_acceleration"]
        
        self.Q = np.diag([q_p, q_p, q_p, q_v, q_v, q_v, q_a, q_a, q_a])
        
        # 观测噪声协方差
        r_p = R_config.get("R_position", KF_CONFIG["R_position"]) if R_config else KF_CONFIG["R_position"]
        self.R = np.eye(self.obs_dim) * r_p
        
        # 历史记录
        self.history = []
    
    def initialize(self, pos):
        """用初始观测初始化状态"""
        self.x[0:3, 0] = pos
        self.x[3:6, 0] = 0
        self.x[6:9, 0] = 0
        self.initialized = True
    
    def predict(self, dt=None):
        """预测 dt 秒后的状态"""
        if not self.initialized:
            return None
        
        if dt is None:
            dt = self.dt
        
        # 构建 dt 对应的状态转移矩阵
        F_dt = np.eye(self.state_dim)
        F_dt[0, 3] = dt;  F_dt[0, 6] = 0.5 * dt**2
        F_dt[1, 4] = dt;  F_dt[1, 7] = 0.5 * dt**2
        F_dt[2, 5] = dt;  F_dt[2, 8] = 0.5 * dt**2
        F_dt[3, 6] = dt
        F_dt[4, 7] = dt
        F_dt[5, 8] = dt
        
        # 标准KF预测步骤
        x_pred = F_dt @ self.x
        P_pred = F_dt @ self.P @ F_dt.T + self.Q
        
        # 不更新内部状态，只返回预测值
        return x_pred[0:3].flatten()
    
    def update(self, obs):
        """标准KF更新步骤"""
        obs = np.array(obs, dtype=np.float64).reshape(self.obs_dim, 1)
        
        if not self.initialized:
            self.initialize(obs.flatten())
            return self.x[0:3].flatten()
        
        # 预测
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q
        
        # 更新
        y = obs - self.H @ x_pred  # 残差
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)
        
        self.x = x_pred + K @ y
        self.P = (np.eye(self.state_dim) - K @ self.H) @ P_pred
        
        self.history.append({
            "state": self.x.copy(),
            "filtered_pos": self.x[0:3].flatten().copy(),
        })
        
        return self.x[0:3].flatten()
    
    def get_filtered_position(self):
        return self.x[0:3].flatten()
    
    def get_velocity(self):
        return self.x[3:6].flatten()


class GyroEKF:
    """
    小陀螺目标 EKF
    状态: [cx, cy, cz, omega, theta, r, vz]
      cx,cy,cz: 车体中心在相机坐标系中的位置
      omega:    旋转角速度 (rad/s) — 绕世界y轴，正方向
      theta:    装甲板当前相位角 (rad)
      r:        装甲板到车体中心的旋转半径 (m)
      vz:       车体z方向速度 (m/s)，小陀螺应为~0
    
    过程模型（假设绕y轴旋转，在xz平面内做圆周运动）：
      装甲板位置:
        x_armor = cx + r * cos(theta)
        z_armor = cz + r * sin(theta)
        y_armor = cy  (高度不变)
      
      状态转移:
        cx,k+1 = cx,k
        cy,k+1 = cy,k
        cz,k+1 = cz,k + vz * dt
        omega,k+1 = omega,k
        theta,k+1 = theta,k + omega * dt
        r,k+1 = r,k
        vz,k+1 = vz,k
    """
    
    def __init__(self, dt=None, config=None):
        self.dt = dt or EKF_CONFIG["dt"]
        cfg = config or EKF_CONFIG
        
        self.state_dim = 7
        self.obs_dim = 3
        
        # 状态: [cx, cy, cz, omega, theta, r, vz]
        self.x = np.zeros((self.state_dim, 1))
        self.P = cfg["init_P"].copy()
        self.Q = cfg["Q"].copy()
        self.R = cfg["R"].copy()
        
        self.initialized = False
        self.history = []
    
    def initialize(self, first_pos):
        """
        用第一次观测初始化。
        first_pos: (x, y, z) 装甲板第一次观测位置
        初始假设：车体中心在装甲板附近的某个位置，theta=0, r=0.1
        """
        px, py, pz = first_pos
        self.x[0, 0] = px       # cx 先设为当前x
        self.x[1, 0] = py       # cy
        self.x[2, 0] = pz       # cz
        self.x[3, 0] = 2.0 * np.pi  # omega 初始猜测 ~1 rev/s ≈ 6.28 rad/s
        self.x[4, 0] = 0.0      # theta
        self.x[5, 0] = 0.1      # r 初始猜测 0.1m
        self.x[6, 0] = 0.0      # vz
        self.initialized = True
    
    def _state_transition(self, x, dt):
        """非线性状态转移函数 f(x)"""
        cx, cy, cz, omega, theta, r, vz = x.flatten()
        
        cx_new = cx
        cy_new = cy
        cz_new = cz + vz * dt
        omega_new = omega
        theta_new = theta + omega * dt
        r_new = r
        vz_new = vz
        
        return np.array([cx_new, cy_new, cz_new, omega_new, theta_new, r_new, vz_new]).reshape(-1, 1)
    
    def _jacobian_F(self, x, dt):
        """状态转移函数的雅可比矩阵"""
        cx, cy, cz, omega, theta, r, vz = x.flatten()
        
        F = np.eye(self.state_dim)
        # cz 对 vz 的偏导
        F[2, 6] = dt
        # theta 对 omega 的偏导
        F[4, 3] = dt
        # theta 对 theta 的偏导 = 1（已设）
        
        return F
    
    def _measurement_model(self, x):
        """观测模型 h(x): 从状态计算装甲板位置"""
        cx, cy, cz, omega, theta, r, vz = x.flatten()
        
        x_armor = cx + r * np.cos(theta)
        y_armor = cy
        z_armor = cz + r * np.sin(theta)
        
        return np.array([x_armor, y_armor, z_armor]).reshape(-1, 1)
    
    def _jacobian_H(self, x):
        """观测模型的雅可比矩阵"""
        cx, cy, cz, omega, theta, r, vz = x.flatten()
        
        H = np.zeros((self.obs_dim, self.state_dim))
        # x_armor = cx + r*cos(theta)
        H[0, 0] = 1                                      # ∂/∂cx
        H[0, 4] = -r * np.sin(theta)                     # ∂/∂theta
        H[0, 5] = np.cos(theta)                          # ∂/∂r
        
        # y_armor = cy
        H[1, 1] = 1                                      # ∂/∂cy
        
        # z_armor = cz + r*sin(theta)
        H[2, 2] = 1                                      # ∂/∂cz
        H[2, 4] = r * np.cos(theta)                      # ∂/∂theta
        H[2, 5] = np.sin(theta)                          # ∂/∂r
        
        return H
    
    def predict(self, dt=None):
        """EKF 预测步骤"""
        if not self.initialized:
            return None
        
        if dt is None:
            dt = self.dt
        
        F = self._jacobian_F(self.x, dt)
        self.x = self._state_transition(self.x, dt)
        self.P = F @ self.P @ F.T + self.Q
        
        return self._measurement_model(self.x).flatten()
    
    def update(self, obs):
        """EKF 更新步骤"""
        obs = np.array(obs, dtype=np.float64).reshape(self.obs_dim, 1)
        
        if not self.initialized:
            self.initialize(obs.flatten())
            return self.get_car_center(), self.get_omega()
        
        # 先预测
        F = self._jacobian_F(self.x, self.dt)
        x_pred = self._state_transition(self.x, self.dt)
        P_pred = F @ self.P @ F.T + self.Q
        
        # 更新
        H = self._jacobian_H(x_pred)
        z_pred = self._measurement_model(x_pred)
        
        y = obs - z_pred  # 残差
        S = H @ P_pred @ H.T + self.R
        K = P_pred @ H.T @ np.linalg.inv(S)
        
        self.x = x_pred + K @ y
        self.P = (np.eye(self.state_dim) - K @ H) @ P_pred
        
        # 归一化 theta 到 [-pi, pi]
        self.x[4, 0] = np.arctan2(np.sin(self.x[4, 0]), np.cos(self.x[4, 0]))
        
        self.history.append({
            "state": self.x.copy(),
            "car_center": self.get_car_center(),
            "omega": self.get_omega(),
        })
        
        return self.get_car_center(), self.get_omega()
    
    def get_car_center(self):
        """获取车体中心位置"""
        return self.x[0:3].flatten()
    
    def get_omega(self):
        """获取旋转角速度 (rad/s)"""
        return float(self.x[3, 0])
    
    def get_armor_position(self):
        """获取当前估计的装甲板位置"""
        return self._measurement_model(self.x).flatten()
    
    def predict_armor_position(self, dt=0.25):
        """预测 dt 秒后的装甲板位置"""
        if not self.initialized:
            return None
        
        # 前向传播状态
        x_future = self._state_transition(self.x, dt)
        return self._measurement_model(x_future).flatten()