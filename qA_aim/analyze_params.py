"""
Q、R 参数对比分析
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import FIGURES_DIR, DATA_DIR, load_camera_params, VIDEO_CONFIGS
from detector import ArmorDetector
from solver import PnPSolver
from tracker import LinearKalmanFilter, GyroEKF


plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def analyze_kf_qr_params():
    """
    分析 Q、R 对平移目标预测的影响
    """
    import cv2
    
    K, dist, _ = load_camera_params()
    video_path = DATA_DIR / "远距离平移.mp4"
    cap = cv2.VideoCapture(str(video_path))
    fps = 150
    
    detector = ArmorDetector(color="red")
    solver = PnPSolver(K, dist)
    
    # 收集观测数据
    observations = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        armors, _ = detector.detect(frame)
        if armors:
            result = solver.solve(armors[0]["corners"])
            if result:
                observations.append((frame_idx / fps, *result["position"]))
        frame_idx += 1
    cap.release()
    
    if len(observations) < 50:
        print("观测数据不足")
        return
    
    observations = np.array(observations)  # (N, 4)
    
    # 测试参数组合
    Q_positions = [1e-6, 1e-4, 1e-2, 1.0]
    R_positions = [1e-5, 1e-3, 1e-1, 10.0]
    
    fig, axes = plt.subplots(len(Q_positions), len(R_positions), 
                              figsize=(4*len(R_positions), 3*len(Q_positions)))
    
    for i, q_p in enumerate(Q_positions):
        for j, r_p in enumerate(R_positions):
            # 创建 KF
            kf = LinearKalmanFilter(
                dt=1.0/fps,
                Q_config={"Q_position": q_p, "Q_velocity": q_p*100, "Q_acceleration": q_p*1000},
                R_config={"R_position": r_p},
            )
            
            filtered = []
            predicted = []
            future_obs = []
            
            for k, (t, x, y, z) in enumerate(observations):
                obs = np.array([x, y, z])
                filt = kf.update(obs)
                filtered.append(filt)
                
                # 预测
                pred = kf.predict(dt=0.25)
                if pred is not None:
                    predicted.append((t, *pred))
                    
                    # 找 0.25s 后的观测
                    fut_k = k + int(0.25 * fps)
                    if fut_k < len(observations):
                        future_obs.append(observations[fut_k, 1:])
                    else:
                        future_obs.append(None)
            
            # 计算误差
            errors = []
            for pred_val, fut_val in zip(predicted, future_obs):
                if fut_val is not None:
                    errors.append(np.linalg.norm(np.array(pred_val[1:]) - fut_val))
            
            mean_error = np.mean(errors) if errors else float('inf')
            
            ax = axes[i, j] if len(Q_positions) > 1 else axes[j]
            ax.plot(observations[:, 0], observations[:, 1], 'b.', markersize=1, alpha=0.3, label='观测')
            filtered_np = np.array(filtered)
            ax.plot(observations[:len(filtered_np), 0], filtered_np[:, 0], 'r-', linewidth=1, label='滤波')
            ax.set_title(f'Q_pos={q_p:.0e}, R_pos={r_p:.0e}\nMeanErr={mean_error:.4f}m')
            ax.grid(True, alpha=0.3)
            if i == len(Q_positions)-1:
                ax.set_xlabel('时间(s)')
            if j == 0:
                ax.set_ylabel('x (m)')
    
    plt.suptitle('Q、R 参数对滤波效果的影响\nQ大=信任模型, R大=信任观测', fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "Q_R对比图.png", dpi=150)
    plt.close()
    print(f"Q/R对比图已保存: {FIGURES_DIR / 'Q_R对比图.png'}")
    
    # 打印结论
    print("\n=== Q、R 参数分析结论 ===")
    print("Q (过程噪声) 越大 → 滤波器更信任观测值 → 滤波曲线更贴近观测，但噪声保留更多")
    print("Q 越小 → 滤波器更信任模型 → 滤波曲线更平滑，但响应变慢")
    print("R (观测噪声) 越大 → 滤波器更不信任观测 → 滤波曲线更平滑")
    print("R 越小 → 滤波器更信任观测 → 滤波曲线更贴近原始数据")
    print("对于平移目标：适中的 Q 和小 R 效果较好，平衡平滑度和响应速度")


def analyze_ekf_qr_params():
    """
    分析 EKF Q、R 对小陀螺估计的影响
    """
    import cv2
    
    K, dist, _ = load_camera_params()
    video_path = DATA_DIR / "中距离陀螺.mp4"
    cap = cv2.VideoCapture(str(video_path))
    fps = 150
    
    detector = ArmorDetector(color="red")
    solver = PnPSolver(K, dist)
    
    observations = []
    frame_idx = 0
    max_frames = 500  # 只取前500帧做对比
    
    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        armors, _ = detector.detect(frame)
        if armors:
            result = solver.solve(armors[0]["corners"])
            if result:
                observations.append((frame_idx / fps, *result["position"]))
        frame_idx += 1
    cap.release()
    
    if len(observations) < 50:
        return
    
    observations = np.array(observations)
    
    # 不同 Q_omega
    omega_Q_values = [1e-5, 1e-3, 1e-1]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, q_omega in enumerate(omega_Q_values):
        ekf = GyroEKF(dt=1.0/fps)
        ekf.Q[3, 3] = q_omega  # omega 的过程噪声
        
        car_centers = []
        omegas = []
        times = []
        
        for t, x, y, z in observations:
            cc, omega = ekf.update(np.array([x, y, z]))
            times.append(t)
            car_centers.append(cc)
            omegas.append(omega)
        
        car_centers = np.array(car_centers)
        omegas = np.array(omegas)
        
        ax = axes[idx]
        ax.plot(observations[:, 0], observations[:, 3], 'b.', markersize=1, alpha=0.3, label='装甲板观测z')
        ax.plot(times, car_centers[:, 2], 'r-', linewidth=1.5, label='车体中心cz')
        
        std_cz = np.std(car_centers[:, 2])
        ax.set_title(f'Q_omega={q_omega:.0e}\ncz std={std_cz:.4f}m')
        ax.set_xlabel('时间 (s)')
        ax.set_ylabel('z (m)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('EKF: Q_omega 对车体中心估计的影响\nQ_omega 越大 → 越信任观测 → 车体中心波动越大', fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "Q_R对比图_陀螺.png", dpi=150)
    plt.close()
    print(f"陀螺Q/R对比图已保存: {FIGURES_DIR / 'Q_R对比图_陀螺.png'}")


def main():
    print("分析 Q、R 参数对预测效果的影响...")
    analyze_kf_qr_params()
    analyze_ekf_qr_params()


if __name__ == "__main__":
    main()