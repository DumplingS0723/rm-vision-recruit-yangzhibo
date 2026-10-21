"""
Q、R 参数对比分析
========================
分析卡尔曼滤波器参数 Q（过程噪声协方差）和 R（观测噪声协方差）
对滤波效果和预测精度的影响。

使用方法: python analyze_params.py
"""
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    FIGURES_DIR, DATA_DIR, load_camera_params, VIDEO_CONFIGS,
    KF_CONFIG, EKF_CONFIG,
)
from detector import ArmorDetector
from solver import PnPSolver
from tracker import LinearKalmanFilter, GyroEKF


plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "DejaVu Sans", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# Part 1: 普通 KF（平移目标）— Q、R 参数分析
# ============================================================

def collect_translation_observations(max_frames=None):
    """从 '远距离平移.mp4' 收集观测数据"""
    K, dist, _ = load_camera_params()
    video_path = DATA_DIR / "远距离平移.mp4"
    
    if not video_path.exists():
        print(f"视频文件不存在: {video_path}")
        return None
    
    cap = cv2.VideoCapture(str(video_path))
    fps = 150
    
    detector = ArmorDetector(color="red")
    solver = PnPSolver(K, dist)
    
    observations = []
    frame_idx = 0
    
    print("收集平移视频观测数据...")
    while True:
        if max_frames and frame_idx >= max_frames:
            break
        ret, frame = cap.read()
        if not ret:
            break
        
        armors, _ = detector.detect(frame)
        if armors:
            result = solver.solve(armors[0]["corners"])
            if result and result["reproj_error"] < 5.0:
                observations.append((
                    frame_idx / fps,
                    result["position"][0],
                    result["position"][1],
                    result["position"][2],
                ))
        
        frame_idx += 1
    
    cap.release()
    observations = np.array(observations)
    print(f"  收集到 {len(observations)} 个有效观测")
    return observations, fps


def run_kf_with_params(observations, fps, Q_pos, Q_vel, Q_acc, R_pos):
    """
    使用指定参数运行 KF，返回滤波结果和预测误差
    
    Returns:
        filtered: 滤波后位置 (N, 4) [t, x, y, z]
        predicted: 预测值列表 [(t, x, y, z), ...]
        errors: 预测误差列表 [e1, e2, ...]
        smoothness: 滤波平滑度（相邻帧位置变化的标准差，越小越平滑）
    """
    kf = LinearKalmanFilter(
        dt=1.0 / fps,
        Q_config={
            "Q_position": Q_pos,
            "Q_velocity": Q_vel,
            "Q_acceleration": Q_acc,
        },
        R_config={"R_position": R_pos},
    )
    
    filtered = []
    predicted = []
    
    for i, (t, x, y, z) in enumerate(observations):
        obs = np.array([x, y, z])
        filt = kf.update(obs)
        filtered.append((t, filt[0], filt[1], filt[2]))
        
        # 预测 0.25s 后
        pred = kf.predict(dt=0.25)
        if pred is not None:
            predicted.append((t, pred[0], pred[1], pred[2]))
    
    filtered = np.array(filtered)
    
    # 计算预测误差：匹配预测值和 0.25s 后的观测
    errors = []
    pred_frames = int(0.25 * fps)  # 0.25s 对应的帧数
    
    for pred_row in predicted:
        t_pred = pred_row[0]
        pred_pos = np.array(pred_row[1:])
        
        # 找 t_pred + 0.25s 最近的观测
        target_t = t_pred + 0.25
        time_diffs = np.abs(observations[:, 0] - target_t)
        closest_idx = np.argmin(time_diffs)
        
        if time_diffs[closest_idx] < 0.02:  # 容差约 3 帧
            obs_pos = observations[closest_idx, 1:]
            errors.append(np.linalg.norm(pred_pos - obs_pos))
    
    # 计算平滑度：滤波位置相邻帧的变化量
    if len(filtered) > 1:
        diffs = np.diff(filtered[:, 1:], axis=0)
        smoothness = np.mean(np.linalg.norm(diffs, axis=1))
    else:
        smoothness = float('inf')
    
    mean_error = np.mean(errors) if errors else float('inf')
    rmse = np.sqrt(np.mean(np.array(errors)**2)) if errors else float('inf')
    
    return filtered, predicted, errors, smoothness, mean_error, rmse


def analyze_kf_qr(observations, fps):
    """全面的 Q、R 参数分析"""
    
    # ============ 参数网格 ============
    # Q_position: 位置过程噪声 — 反映模型对状态变化的预期
    # R_position: 观测噪声 — 反映对传感器精度的信任程度
    
    Q_pos_values = [1e-6, 1e-4, 1e-3, 1e-2, 1e-1]
    R_pos_values = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    
    # ============ 图1: 热力图 — 预测误差随 Q、R 变化 ============
    error_matrix = np.zeros((len(Q_pos_values), len(R_pos_values)))
    smoothness_matrix = np.zeros((len(Q_pos_values), len(R_pos_values)))
    
    print("\n扫描 Q、R 参数空间...")
    for i, q_pos in enumerate(Q_pos_values):
        for j, r_pos in enumerate(R_pos_values):
            _, _, _, smoothness, mean_error, _ = run_kf_with_params(
                observations, fps,
                Q_pos=q_pos,
                Q_vel=q_pos * 10,      # 速度噪声 = Q_pos * 10
                Q_acc=q_pos * 100,     # 加速度噪声 = Q_pos * 100
                R_pos=r_pos,
            )
            error_matrix[i, j] = mean_error
            smoothness_matrix[i, j] = smoothness
            print(f"  Q={q_pos:.0e}, R={r_pos:.0e} → 误差={mean_error:.4f}m, 平滑度={smoothness:.4f}")
    
    # ============ 绘制大图 ============
    fig = plt.figure(figsize=(20, 24))
    gs = GridSpec(4, 2, figure=fig, hspace=0.35, wspace=0.3)
    
    # ---- (A) 热力图：预测误差 ----
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(error_matrix, cmap='RdYlGn_r', aspect='auto', origin='lower')
    ax.set_xticks(range(len(R_pos_values)))
    ax.set_xticklabels([f'{v:.0e}' for v in R_pos_values], rotation=45)
    ax.set_yticks(range(len(Q_pos_values)))
    ax.set_yticklabels([f'{v:.0e}' for v in Q_pos_values])
    ax.set_xlabel('R (观测噪声)', fontsize=11)
    ax.set_ylabel('Q_pos (过程噪声)', fontsize=11)
    ax.set_title('(A) 预测误差热力图 (m)\n绿色=低误差  |  红色=高误差', fontsize=12, fontweight='bold')
    
    # 标注数值
    for i in range(len(Q_pos_values)):
        for j in range(len(R_pos_values)):
            color = 'white' if error_matrix[i, j] > np.median(error_matrix) else 'black'
            ax.text(j, i, f'{error_matrix[i,j]:.3f}', ha='center', va='center',
                   color=color, fontsize=8, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8, label='Mean Prediction Error (m)')
    
    # ---- (B) 热力图：平滑度 ----
    ax = fig.add_subplot(gs[0, 1])
    im2 = ax.imshow(smoothness_matrix, cmap='YlOrRd', aspect='auto', origin='lower')
    ax.set_xticks(range(len(R_pos_values)))
    ax.set_xticklabels([f'{v:.0e}' for v in R_pos_values], rotation=45)
    ax.set_yticks(range(len(Q_pos_values)))
    ax.set_yticklabels([f'{v:.0e}' for v in Q_pos_values])
    ax.set_xlabel('R (观测噪声)', fontsize=11)
    ax.set_ylabel('Q_pos (过程噪声)', fontsize=11)
    ax.set_title('(B) 滤波平滑度热力图\n绿色=平滑  |  红色=抖动', fontsize=12, fontweight='bold')
    plt.colorbar(im2, ax=ax, shrink=0.8, label='Smoothness (m/frame)')
    
    # ---- (C) 固定 R，变化 Q：时间序列对比 ----
    ax = fig.add_subplot(gs[1, :])
    
    r_fixed = 1e-4  # 固定 R
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(Q_pos_values)))
    
    # 只画前 200 个观测点
    n_show = min(200, len(observations))
    
    ax.plot(observations[:n_show, 0], observations[:n_show, 1],
            'k.', markersize=1.5, alpha=0.4, label='观测 (raw)')
    
    for idx, q_pos in enumerate(Q_pos_values):
        filtered, _, _, _, _, _ = run_kf_with_params(
            observations, fps,
            Q_pos=q_pos, Q_vel=q_pos*10, Q_acc=q_pos*100,
            R_pos=r_fixed,
        )
        n = min(n_show, len(filtered))
        ax.plot(filtered[:n, 0], filtered[:n, 1],
                color=colors[idx], linewidth=1.5,
                label=f'Q={q_pos:.0e}')
    
    ax.set_xlabel('时间 (s)', fontsize=11)
    ax.set_ylabel('x (m)', fontsize=11)
    ax.set_title(f'(C) 固定 R={r_fixed:.0e}，不同 Q 的滤波效果\n'
                 f'Q越小 → 越平滑但响应慢 | Q越大 → 越贴近观测但噪声多',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # ---- (D) 固定 Q，变化 R：时间序列对比 ----
    ax = fig.add_subplot(gs[2, :])
    
    q_fixed = 1e-4  # 固定 Q
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(R_pos_values)))
    
    ax.plot(observations[:n_show, 0], observations[:n_show, 1],
            'k.', markersize=1.5, alpha=0.4, label='观测 (raw)')
    
    for idx, r_pos in enumerate(R_pos_values):
        filtered, _, _, _, _, _ = run_kf_with_params(
            observations, fps,
            Q_pos=q_fixed, Q_vel=q_fixed*10, Q_acc=q_fixed*100,
            R_pos=r_pos,
        )
        n = min(n_show, len(filtered))
        ax.plot(filtered[:n, 0], filtered[:n, 1],
                color=colors[idx], linewidth=1.5,
                label=f'R={r_pos:.0e}')
    
    ax.set_xlabel('时间 (s)', fontsize=11)
    ax.set_ylabel('x (m)', fontsize=11)
    ax.set_title(f'(D) 固定 Q={q_fixed:.0e}，不同 R 的滤波效果\n'
                 f'R越小 → 越信任观测 → 噪声多 | R越大 → 越平滑但可能滞后',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # ---- (E) 误差 vs Q/R 曲线 ----
    ax = fig.add_subplot(gs[3, 0])
    
    # 固定 R，变化 Q
    r_mid = R_pos_values[len(R_pos_values)//2]
    errors_vs_q = []
    for q_pos in Q_pos_values:
        _, _, _, _, mean_e, rmse = run_kf_with_params(
            observations, fps,
            Q_pos=q_pos, Q_vel=q_pos*10, Q_acc=q_pos*100,
            R_pos=r_mid,
        )
        errors_vs_q.append((mean_e, rmse))
    errors_vs_q = np.array(errors_vs_q)
    
    ax.semilogx(Q_pos_values, errors_vs_q[:, 0], 'b-o', linewidth=2, markersize=8, label='Mean Error')
    ax.semilogx(Q_pos_values, errors_vs_q[:, 1], 'r-s', linewidth=2, markersize=8, label='RMSE')
    ax.set_xlabel('Q_pos (过程噪声)', fontsize=11)
    ax.set_ylabel('预测误差 (m)', fontsize=11)
    ax.set_title(f'(E) 预测误差 vs Q (R={r_mid:.0e}固定)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # ---- (F) 误差 vs R 曲线 ----
    ax = fig.add_subplot(gs[3, 1])
    
    q_mid = Q_pos_values[len(Q_pos_values)//2]
    errors_vs_r = []
    for r_pos in R_pos_values:
        _, _, _, _, mean_e, rmse = run_kf_with_params(
            observations, fps,
            Q_pos=q_mid, Q_vel=q_mid*10, Q_acc=q_mid*100,
            R_pos=r_pos,
        )
        errors_vs_r.append((mean_e, rmse))
    errors_vs_r = np.array(errors_vs_r)
    
    ax.semilogx(R_pos_values, errors_vs_r[:, 0], 'b-o', linewidth=2, markersize=8, label='Mean Error')
    ax.semilogx(R_pos_values, errors_vs_r[:, 1], 'r-s', linewidth=2, markersize=8, label='RMSE')
    ax.set_xlabel('R_pos (观测噪声)', fontsize=11)
    ax.set_ylabel('预测误差 (m)', fontsize=11)
    ax.set_title(f'(F) 预测误差 vs R (Q={q_mid:.0e}固定)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # ---- 总标题 ----
    fig.suptitle('卡尔曼滤波器 Q、R 参数对平移目标预测效果的影响\n'
                 'Q = 过程噪声协方差（信任模型程度）  |  R = 观测噪声协方差（信任观测程度）',
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(FIGURES_DIR / "Q_R对比图.png", dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n✅ Q/R对比图已保存: {FIGURES_DIR / 'Q_R对比图.png'}")
    
    # 返回最优参数
    best_idx = np.unravel_index(np.argmin(error_matrix), error_matrix.shape)
    best_Q = Q_pos_values[best_idx[0]]
    best_R = R_pos_values[best_idx[1]]
    best_error = error_matrix[best_idx]
    
    return best_Q, best_R, best_error


# ============================================================
# Part 2: EKF（小陀螺目标）— Q、R 参数分析
# ============================================================

def collect_gyro_observations(video_name="中距离陀螺", max_frames=800):
    """从小陀螺视频收集观测数据"""
    K, dist, _ = load_camera_params()
    video_path = DATA_DIR / f"{video_name}.mp4"
    
    if not video_path.exists():
        print(f"视频文件不存在: {video_path}")
        return None, None
    
    cap = cv2.VideoCapture(str(video_path))
    fps = 150
    
    detector = ArmorDetector(color="red")
    solver = PnPSolver(K, dist)
    
    observations = []
    frame_idx = 0
    
    print(f"收集 {video_name} 观测数据...")
    while True:
        if max_frames and frame_idx >= max_frames:
            break
        ret, frame = cap.read()
        if not ret:
            break
        
        armors, _ = detector.detect(frame)
        if armors:
            result = solver.solve(armors[0]["corners"])
            if result and result["reproj_error"] < 5.0:
                observations.append((
                    frame_idx / fps,
                    result["position"][0],
                    result["position"][1],
                    result["position"][2],
                ))
        
        frame_idx += 1
    
    cap.release()
    observations = np.array(observations)
    print(f"  收集到 {len(observations)} 个有效观测")
    return observations, fps


def run_ekf_with_q(observations, fps, Q_omega, Q_others=1e-6):
    """
    使用指定 Q_omega 运行 EKF
    
    Returns:
        car_centers: (N, 4) [t, cx, cy, cz]
        omegas: (N, 2) [t, omega]
        cz_std: 车体中心 z 的标准差（衡量车体中心的稳定性）
        omega_std: 角速度的标准差
    """
    ekf = GyroEKF(dt=1.0 / fps)
    
    # 修改 Q 矩阵
    ekf.Q = np.diag([
        Q_others,   # cx
        Q_others,   # cy
        Q_others,   # cz
        Q_omega,    # omega ← 重点变量
        1e-4,       # theta
        1e-6,       # r
        Q_others,   # vz
    ])
    
    car_centers = []
    omegas = []
    
    for t, x, y, z in observations:
        obs = np.array([x, y, z])
        cc, omega = ekf.update(obs)
        car_centers.append((t, cc[0], cc[1], cc[2]))
        omegas.append((t, omega))
    
    car_centers = np.array(car_centers)
    omegas = np.array(omegas)
    
    # 稳定性指标
    # 排除初始收敛阶段（前 50 个样本）
    if len(car_centers) > 100:
        cz_stable = car_centers[50:, 3]
        omega_stable = omegas[50:, 1]
    else:
        cz_stable = car_centers[:, 3]
        omega_stable = omegas[:, 1]
    
    cz_std = np.std(cz_stable)
    omega_std = np.std(omega_stable)
    omega_mean = np.mean(omega_stable)
    omega_rps = omega_mean / (2 * np.pi)
    
    return car_centers, omegas, cz_std, omega_std, omega_mean, omega_rps


def analyze_ekf_qr(observations, fps):
    """EKF Q 参数分析（重点分析 omega 的过程噪声）"""
    
    Q_omega_values = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    R_pos_values = [1e-5, 1e-4, 1e-3, 1e-2]
    
    # ============ 图2: EKF 参数分析 ============
    fig = plt.figure(figsize=(20, 20))
    gs = GridSpec(4, 2, figure=fig, hspace=0.35, wspace=0.3)
    
    # ---- (A) 不同 Q_omega 下的车体中心 z 轨迹 ----
    ax = fig.add_subplot(gs[0, :])
    
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(Q_omega_values)))
    
    # 画装甲板观测 z
    n_show = min(500, len(observations))
    ax.plot(observations[:n_show, 0], observations[:n_show, 3],
            'k.', markersize=1, alpha=0.3, label='装甲板观测 z')
    
    for idx, q_omega in enumerate(Q_omega_values):
        car_centers, omegas, cz_std, _, omega_mean, omega_rps = run_ekf_with_q(
            observations, fps, Q_omega=q_omega
        )
        n = min(n_show, len(car_centers))
        ax.plot(car_centers[:n, 0], car_centers[:n, 3],
                color=colors[idx], linewidth=1.5,
                label=f'Q_ω={q_omega:.0e} (std={cz_std:.4f}m, ω={omega_rps:.2f} rev/s)')
    
    ax.set_xlabel('时间 (s)', fontsize=11)
    ax.set_ylabel('z (m)', fontsize=11)
    ax.set_title('(G) 不同 Q_omega 下车体中心 z 的估计轨迹\n'
                 '车体中心应不变 → cz_std 越小越好 | ω 应稳定',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # ---- (B) cz 稳定性 vs Q_omega ----
    ax = fig.add_subplot(gs[1, 0])
    
    cz_stds = []
    omega_stds = []
    omega_means = []
    
    for q_omega in Q_omega_values:
        _, _, cz_std, omega_std, omega_mean, _ = run_ekf_with_q(
            observations, fps, Q_omega=q_omega
        )
        cz_stds.append(cz_std)
        omega_stds.append(omega_std)
        omega_means.append(omega_mean)
    
    ax.semilogx(Q_omega_values, cz_stds, 'b-o', linewidth=2, markersize=8, label='cz 标准差')
    ax.set_xlabel('Q_omega (omega 过程噪声)', fontsize=11)
    ax.set_ylabel('车体中心 z 标准差 (m)', fontsize=11, color='b')
    ax.tick_params(axis='y', labelcolor='b')
    ax.set_title('(H) 车体中心稳定性 vs Q_omega\n标准差越小 = 车体中心估计越稳定',
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 双 y 轴：角速度标准差
    ax2 = ax.twinx()
    ax2.semilogx(Q_omega_values, omega_stds, 'r-s', linewidth=2, markersize=8, label='ω 标准差')
    ax2.set_ylabel('角速度标准差 (rad/s)', fontsize=11, color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    # 合并图例
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
    
    # ---- (C) 角速度估计 vs Q_omega ----
    ax = fig.add_subplot(gs[1, 1])
    
    for idx, q_omega in enumerate(Q_omega_values[:4]):  # 只画前4个，避免太密
        _, omegas, _, _, _, _ = run_ekf_with_q(observations, fps, Q_omega=q_omega)
        n = min(n_show, len(omegas))
        ax.plot(omegas[:n, 0], omegas[:n, 1],
                color=colors[idx], linewidth=1.2,
                label=f'Q_ω={q_omega:.0e}')
    
    ax.set_xlabel('时间 (s)', fontsize=11)
    ax.set_ylabel('角速度 (rad/s)', fontsize=11)
    ax.set_title('(I) 不同 Q_omega 下角速度估计曲线\n'
                 'Q_ω 小 → 角速度稳定但收敛慢 | Q_ω 大 → 响应快但波动大',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # ---- (D) 角速度收敛分析 ----
    ax = fig.add_subplot(gs[2, :])
    
    q_omega_good = Q_omega_values[np.argmin(cz_stds)]  # 选 cz 最稳定的
    _, omegas_good, _, _, omega_mean, omega_rps = run_ekf_with_q(
        observations, fps, Q_omega=q_omega_good
    )
    
    ax.plot(omegas_good[:, 0], omegas_good[:, 1],
            'b-', linewidth=1.5, alpha=0.8, label=f'Q_ω={q_omega_good:.0e}')
    ax.axhline(omega_mean, color='r', linestyle='--', linewidth=2,
               label=f'均值 = {omega_mean:.2f} rad/s = {omega_rps:.2f} rev/s')
    
    # ±1σ 区间
    omega_std_good = np.std(omegas_good[50:, 1])
    ax.fill_between(omegas_good[:, 0],
                    omega_mean - omega_std_good,
                    omega_mean + omega_std_good,
                    alpha=0.2, color='r',
                    label=f'±1σ = ±{omega_std_good:.2f} rad/s')
    
    ax.set_xlabel('时间 (s)', fontsize=11)
    ax.set_ylabel('角速度 (rad/s)', fontsize=11)
    ax.set_title(f'(J) 最优 Q_omega={q_omega_good:.0e} 下的角速度估计\n'
                 f'测试台旋转角速度约 {omega_rps:.2f} rev/s ({omega_mean:.1f} rad/s)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # ---- (E) R 参数对 EKF 的影响 ----
    ax = fig.add_subplot(gs[3, 0])
    
    cz_stds_vs_r = []
    for r_pos in R_pos_values:
        ekf = GyroEKF(dt=1.0/fps)
        ekf.Q[3, 3] = q_omega_good
        ekf.R = np.diag([r_pos, r_pos, r_pos])
        
        car_centers_r = []
        for t, x, y, z in observations:
            obs = np.array([x, y, z])
            cc, _ = ekf.update(obs)
            car_centers_r.append((t, cc[0], cc[1], cc[2]))
        car_centers_r = np.array(car_centers_r)
        cz_stds_vs_r.append(np.std(car_centers_r[50:, 3]))
    
    ax.semilogx(R_pos_values, cz_stds_vs_r, 'g-D', linewidth=2, markersize=8)
    ax.set_xlabel('R_pos (观测噪声)', fontsize=11)
    ax.set_ylabel('车体中心 z 标准差 (m)', fontsize=11)
    ax.set_title(f'(K) 车体中心稳定性 vs R (Q_ω={q_omega_good:.0e})\n'
                 f'R 越大 → 越不信任观测 → 车体中心越稳定',
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # ---- (F) 不同 R 下的车体中心轨迹对比 ----
    ax = fig.add_subplot(gs[3, 1])
    
    colors_r = plt.cm.cool(np.linspace(0.1, 0.9, len(R_pos_values)))
    
    ax.plot(observations[:n_show, 0], observations[:n_show, 3],
            'k.', markersize=1, alpha=0.2, label='装甲板观测 z')
    
    for idx, r_pos in enumerate(R_pos_values):
        ekf = GyroEKF(dt=1.0/fps)
        ekf.Q[3, 3] = q_omega_good
        ekf.R = np.diag([r_pos, r_pos, r_pos])
        
        car_centers_r = []
        for t, x, y, z in observations:
            obs = np.array([x, y, z])
            cc, _ = ekf.update(obs)
            car_centers_r.append((t, cc[0], cc[1], cc[2]))
        car_centers_r = np.array(car_centers_r)
        
        n_r = min(n_show, len(car_centers_r))
        ax.plot(car_centers_r[:n_r, 0], car_centers_r[:n_r, 3],
                color=colors_r[idx], linewidth=1.5,
                label=f'R={r_pos:.0e}')
    
    ax.set_xlabel('时间 (s)', fontsize=11)
    ax.set_ylabel('z (m)', fontsize=11)
    ax.set_title(f'(L) 不同 R 下车体中心 z 轨迹\n'
                 f'R 大=平滑但可能偏离 | R 小=贴近观测但波动',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # ---- 总标题 ----
    fig.suptitle('EKF Q、R 参数对小陀螺目标（车体中心 + 角速度）估计的影响\n'
                 '核心参数: Q_omega（角速度过程噪声）决定角速度跟踪的灵活度',
                 fontsize=16, fontweight='bold', y=0.99)
    
    plt.savefig(FIGURES_DIR / "Q_R对比图_陀螺.png", dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n✅ 陀螺Q/R对比图已保存: {FIGURES_DIR / 'Q_R对比图_陀螺.png'}")
    
    return q_omega_good, omega_mean, omega_rps


# ============================================================
# Part 3: 综合分析 + 文字结论
# ============================================================

def print_conclusions(best_Q, best_R, best_error, best_Q_gyro, omega_mean, omega_rps):
    """打印分析结论"""
    print("\n" + "=" * 70)
    print("                     Q、R 参数分析结论")
    print("=" * 70)
    
    print("""
┌─────────────────────────────────────────────────────────────────────┐
│ 1. Q（过程噪声协方差）对滤波的影响                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Q 反映我们对"状态变化速度"的预期：                                   │
│                                                                     │
│  Q 大 → 模型预期状态变化大 → 滤波器更"信任"新观测                     │
│         ✓ 响应快，能跟上目标机动                                     │
│         ✗ 噪声保留多，滤波曲线不平滑                                  │
│                                                                     │
│  Q 小 → 模型预期状态变化小 → 滤波器更"信任"自己的预测                 │
│         ✓ 滤波曲线平滑                                               │
│         ✗ 响应慢，目标机动时产生滞后甚至发散                          │
│                                                                     │
│  物理意义：Q 的选择取决于目标运动特性。                                │
│  - 匀速运动 → Q 小时可平滑噪声                                       │
│  - 机动目标 → 需要较大 Q 才能跟上                                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ 2. R（观测噪声协方差）对滤波的影响                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  R 反映我们对传感器（PnP 解算）精度的信任程度：                        │
│                                                                     │
│  R 大 → 认为观测不可靠 → 滤波器更依赖模型预测                         │
│         ✓ 能有效抑制观测噪声和野值                                   │
│         ✗ 真实信号也可能被衰减                                       │
│                                                                     │
│  R 小 → 认为观测很准确 → 滤波结果贴近原始观测                         │
│         ✓ 能捕捉细微变化                                             │
│         ✗ 噪声被保留，异常值影响大                                   │
│                                                                     │
│  物理意义：R 应匹配实际传感器的噪声水平。                              │
│  - PnP 解算噪声约 1-5cm 级别（良好条件下）                            │
│  - 远距离或运动模糊时噪声增大，可适当增大 R                            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
""")
    
    print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│ 3. 平移目标最优参数                                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  最优 Q_pos = {best_Q:.0e}                                            │
│  最优 R_pos = {best_R:.0e}                                            │
│  对应预测误差 = {best_error:.4f} m                                    │
│                                                                     │
│  解读：                                                               │
│  - 平移目标运动较规律（导轨往复），中等 Q 即可平衡平滑度和响应速度      │
│  - PnP 解算精度在 cm 级别，R 取 1e-4 量级合理                        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ 4. 小陀螺目标 EKF 参数                                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  最优 Q_omega = {best_Q_gyro:.0e}                                      │
│  估计角速度   = {omega_mean:.2f} rad/s ≈ {omega_rps:.2f} rev/s        │
│                                                                     │
│  解读：                                                               │
│  - 小陀螺转速约 {omega_rps:.1f} 转/秒                                     │
│  - Q_omega 控制角速度估计的自适应速度                                 │
│  - Q_omega 太小 → 角速度收敛慢，初始偏差持续很久                       │
│  - Q_omega 太大 → 角速度估计抖动，车体中心不稳定                       │
│  - 推荐取中间值，既能快速收敛又保持稳定                               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ 5. 普通 KF 不适用于小陀螺的原因（回答第⑦问）                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  普通 KF 假设系统为线性模型（CV/CA），状态转移为：                      │
│    x(k+1) = F·x(k) + w                                               │
│                                                                     │
│  但小陀螺的装甲板做匀速圆周运动：                                      │
│    x(t) = cx + r·cos(ω·t + θ₀)                                       │
│    z(t) = cz + r·sin(ω·t + θ₀)                                       │
│                                                                     │
│  这是高度非线性的！用线性模型近似会导致：                               │
│  ① 位置预测有系统性偏差（直线 vs 圆弧）                               │
│  ② 速度方向持续变化，恒定速度假设不成立                               │
│  ③ 无法估计车体中心和角速度                                          │
│                                                                     │
│  解决方案：EKF/UKF，将 ω 和车体中心纳入状态向量，                        │
│  用非线性状态转移模型描述圆周运动。                                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
""")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("        Q、R 参数对比分析")
    print("=" * 70)
    
    # --- Part 1: 平移 KF ---
    print("\n" + "-" * 50)
    print("Part 1: 普通 KF（平移目标）Q、R 参数分析")
    print("-" * 50)
    
    obs_trans, fps_trans = collect_translation_observations(max_frames=600)
    
    if obs_trans is not None and len(obs_trans) > 50:
        best_Q, best_R, best_error = analyze_kf_qr(obs_trans, fps_trans)
    else:
        print("⚠️ 平移观测数据不足，跳过 KF 分析")
        best_Q, best_R, best_error = KF_CONFIG["Q_position"], KF_CONFIG["R_position"], 0.0
    
    # --- Part 2: 陀螺 EKF ---
    print("\n" + "-" * 50)
    print("Part 2: EKF（小陀螺目标）Q、R 参数分析")
    print("-" * 50)
    
    obs_gyro, fps_gyro = collect_gyro_observations("中距离陀螺", max_frames=800)
    
    if obs_gyro is not None and len(obs_gyro) > 50:
        best_Q_gyro, omega_mean, omega_rps = analyze_ekf_qr(obs_gyro, fps_gyro)
    else:
        print("⚠️ 陀螺观测数据不足，跳过 EKF 分析")
        best_Q_gyro = EKF_CONFIG["Q"][3, 3]
        omega_mean = 0.0
        omega_rps = 0.0
    
    # --- Part 3: 结论 ---
    print_conclusions(best_Q, best_R, best_error, best_Q_gyro, omega_mean, omega_rps)
    
    print("\n" + "=" * 70)
    print("分析完成！图表已保存至 qA_aim/figures/ 目录")
    print("=" * 70)


if __name__ == "__main__":
    main()
