"""
主流程 — 装甲板识别、位姿解算、运动预测
"""
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    DATA_DIR, OUTPUT_DIR, FIGURES_DIR, VIDEO_CONFIGS,
    load_camera_params, VIS_CONFIG,
    LIGHT_BAR_CENTER_DIST, LIGHT_BAR_LENGTH,
)
from detector import ArmorDetector
from solver import PnPSolver
from tracker import LinearKalmanFilter, GyroEKF


# 中文字体设置
plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def process_video(video_name, config, K=None, dist=None):
    """
    处理单段视频
    
    Returns:
        detections_df, predictions_df, frames_for_video
    """
    video_path = DATA_DIR / f"{video_name}.mp4"
    if not video_path.exists():
        print(f"⚠️ 视频文件不存在: {video_path}")
        return None, None, None
    
    cap = cv2.VideoCapture(str(video_path))
    fps = config["fps"]
    color = config["color"]
    has_pose = config["has_pose"]
    is_gyro = config.get("gyro", False)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\n{'='*60}")
    print(f"处理: {video_name}")
    print(f"  分辨率: {width}x{height}, 帧率: {fps}, 总帧数: {total_frames}")
    print(f"  颜色: {color}, 位姿: {has_pose}, 陀螺: {is_gyro}")
    print(f"{'='*60}")
    
    # 初始化
    is_blue = (color == "blue")
    detector = ArmorDetector(color=color, is_blue_video=is_blue)
    solver = PnPSolver(K, dist) if (has_pose and K is not None) else None
    
    if is_gyro:
        tracker = GyroEKF(dt=1.0/fps)
    else:
        tracker = LinearKalmanFilter(dt=1.0/fps)
    
    # 结果存储
    detections_rows = []
    predictions_rows = []
    video_frames = []
    
    # 用于可视化的数据缓存
    obs_positions = []          # (t, x, y, z)
    filtered_positions = []     # (t, x, y, z)
    predicted_positions = []    # (t, x_pred, y_pred, z_pred)
    future_obs_map = {}         # frame_idx → 观测（用于"预测vs实际"对比）
    car_centers = []            # (t, cx, cy, cz)
    omega_estimates = []        # (t, omega)
    
    frame_idx = 0
    pbar = tqdm(total=total_frames, desc=video_name)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        t = frame_idx / fps
        
        # 检测
        armors, binary = detector.detect(frame)
        
        if armors:
            armor = armors[0]  # 取最高分装甲板
            corners = armor["corners"]
            
            # 记录检测结果
            row = {
                "frame": frame_idx,
                "lt_x": corners[0][0], "lt_y": corners[0][1],
                "lb_x": corners[1][0], "lb_y": corners[1][1],
                "rb_x": corners[2][0], "rb_y": corners[2][1],
                "rt_x": corners[3][0], "rt_y": corners[3][1],
            }
            
            # 位姿解算
            if solver is not None:
                pnp_result = solver.solve(corners, use_distortion=True)
                if pnp_result is not None:
                    pos = pnp_result["position"]
                    yaw = pnp_result["yaw"]
                    row.update({"x": pos[0], "y": pos[1], "z": pos[2], "yaw": yaw})
                    
                    obs_positions.append((t, pos[0], pos[1], pos[2]))
                    
                    # 跟踪更新
                    if is_gyro:
                        car_center, omega = tracker.update(pos)
                        if car_center is not None:
                            car_centers.append((t, car_center[0], car_center[1], car_center[2]))
                            omega_estimates.append((t, omega))
                            # 预测
                            pred_pos = tracker.predict_armor_position(dt=0.25)
                            if pred_pos is not None:
                                predictions_rows.append({
                                    "t": t,
                                    "cx": car_center[0], "cy": car_center[1], "cz": car_center[2],
                                    "omega": omega,
                                })
                    else:
                        filt_pos = tracker.update(pos)
                        if filt_pos is not None:
                            filtered_positions.append((t, filt_pos[0], filt_pos[1], filt_pos[2]))
                            # 预测 0.25s 后
                            pred_pos = tracker.predict(dt=0.25)
                            if pred_pos is not None:
                                predicted_positions.append((t, pred_pos[0], pred_pos[1], pred_pos[2]))
                                predictions_rows.append({
                                    "t": t,
                                    "x": pred_pos[0], "y": pred_pos[1], "z": pred_pos[2],
                                })
                else:
                    row.update({"x": "", "y": "", "z": "", "yaw": ""})
            else:
                row.update({"x": "", "y": "", "z": "", "yaw": ""})
            
            detections_rows.append(row)
            
            # 缓存观测值用于后续帧的预测对比
            future_obs_map[frame_idx] = pos if (solver and pnp_result) else None
        
        # 可视化帧
        vis_frame = detector.draw_results(frame, armors)
        
        # 在帧上叠加信息
        info_lines = [f"Frame: {frame_idx}", f"Time: {t:.3f}s"]
        if armors and solver and pnp_result:
            info_lines.append(f"Pos: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})m")
        if armors and is_gyro and tracker.initialized:
            cc = tracker.get_car_center()
            info_lines.append(f"Car: ({cc[0]:.2f}, {cc[1]:.2f}, {cc[2]:.2f})m")
            info_lines.append(f"Omega: {tracker.get_omega():.2f} rad/s")
        
        for i, line in enumerate(info_lines):
            cv2.putText(vis_frame, line, (10, 30 + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        video_frames.append(vis_frame)
        
        frame_idx += 1
        pbar.update(1)
    
    cap.release()
    pbar.close()
    
    # 保存检测结果 CSV
    detections_df = pd.DataFrame(detections_rows)
    if not detections_df.empty:
        detections_df.to_csv(OUTPUT_DIR / f"detections_{video_name}.csv", index=False)
    
    # 保存预测结果 CSV
    if predictions_rows:
        pred_df = pd.DataFrame(predictions_rows)
        if is_gyro:
            pred_df.to_csv(OUTPUT_DIR / f"pred_{video_name}.csv", index=False)
        else:
            pred_df.to_csv(OUTPUT_DIR / f"pred_{video_name}.csv", index=False)
    
    return detections_df, predictions_rows, video_frames, {
        "obs": obs_positions,
        "filtered": filtered_positions,
        "predicted": predicted_positions,
        "future_obs": future_obs_map,
        "car_centers": car_centers,
        "omega": omega_estimates,
        "fps": fps,
        "is_gyro": is_gyro,
    }


def save_recognition_video(video_frames, video_name, fps=30):
    """保存识别结果视频"""
    if not video_frames:
        return
    
    out_path = FIGURES_DIR / f"{video_name}_识别.mp4"
    h, w = video_frames[0].shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*"avc1")  # H.264 for macOS
    out = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    
    for frame in video_frames:
        out.write(frame)
    out.release()
    print(f"  识别视频已保存: {out_path}")


def visualize_translation(results, video_name="远距离平移"):
    """可视化平移视频的结果"""
    data = results["data"]
    obs = data["obs"]
    filtered = data["filtered"]
    predicted = data["predicted"]
    future_obs = data["future_obs"]
    fps = data["fps"]
    
    if not obs:
        print("  无观测数据，跳过可视化")
        return
    
    obs = np.array(obs)          # (N, 4): t, x, y, z
    filtered = np.array(filtered) if filtered else None
    predicted = np.array(predicted) if predicted else None
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # ===== 图1: 位置随时间变化 =====
    ax = axes[0, 0]
    ax.plot(obs[:, 0], obs[:, 1], 'b.', markersize=2, alpha=0.5, label='观测 x')
    if filtered is not None:
        ax.plot(filtered[:, 0], filtered[:, 1], 'r-', linewidth=1, label='滤波 x')
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('x (m)')
    ax.set_title('X 位置')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.plot(obs[:, 0], obs[:, 2], 'b.', markersize=2, alpha=0.5, label='观测 y')
    if filtered is not None:
        ax.plot(filtered[:, 0], filtered[:, 2], 'r-', linewidth=1, label='滤波 y')
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('y (m)')
    ax.set_title('Y 位置')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 2]
    ax.plot(obs[:, 0], obs[:, 3], 'b.', markersize=2, alpha=0.5, label='观测 z')
    if filtered is not None:
        ax.plot(filtered[:, 0], filtered[:, 3], 'r-', linewidth=1, label='滤波 z')
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('z (m)')
    ax.set_title('Z 位置（深度）')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # ===== 图2: 俯视轨迹 (x-z平面) =====
    ax = axes[1, 0]
    ax.plot(obs[:, 1], obs[:, 3], 'b.', markersize=2, alpha=0.5, label='观测')
    if filtered is not None:
        ax.plot(filtered[:, 1], filtered[:, 3], 'r-', linewidth=1, label='滤波')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('z (m)')
    ax.set_title('俯视轨迹 (x-z)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # ===== 图3: 预测 vs 0.25s后观测 =====
    if predicted is not None and len(predicted) > 0:
        ax = axes[1, 1]
        
        # 匹配预测值和0.25s后的观测
        matched_pred = []
        matched_obs = []
        match_times = []
        
        for pred_row in predicted:
            t_pred = pred_row[0]
            # 找到 t_pred + 0.25s 附近的观测
            target_t = t_pred + 0.25
            frame_approx = int(target_t * fps)
            
            if frame_approx in future_obs and future_obs[frame_approx] is not None:
                obs_at_target = future_obs[frame_approx]
                matched_pred.append(pred_row[1:])  # x,y,z
                matched_obs.append(obs_at_target)
                match_times.append(t_pred)
        
        if matched_pred:
            matched_pred = np.array(matched_pred)
            matched_obs = np.array(matched_obs)
            
            errors = np.linalg.norm(matched_pred - matched_obs, axis=1)
            
            ax.plot(match_times, matched_pred[:, 0], 'r-', linewidth=1, label='预测 x')
            ax.plot(match_times, matched_obs[:, 0], 'b--', linewidth=1, label='实际 x')
            ax.set_xlabel('时间 (s)')
            ax.set_ylabel('x (m)')
            ax.set_title('预测值 vs 0.25s后观测值')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 误差统计
            ax = axes[1, 2]
            ax.hist(errors, bins=30, alpha=0.7, edgecolor='black')
            ax.axvline(np.mean(errors), color='r', linestyle='--', 
                      label=f'均值: {np.mean(errors):.3f}m')
            ax.axvline(np.median(errors), color='g', linestyle='--', 
                      label=f'中位数: {np.median(errors):.3f}m')
            ax.set_xlabel('误差 (m)')
            ax.set_ylabel('频次')
            ax.set_title(f'预测误差分布\n均值={np.mean(errors):.4f}m, '
                        f'RMSE={np.sqrt(np.mean(errors**2)):.4f}m, '
                        f'最大={np.max(errors):.4f}m')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{video_name}_分析.png", dpi=VIS_CONFIG["figure_dpi"])
    plt.close()
    print(f"  分析图表已保存: {FIGURES_DIR / f'{video_name}_分析.png'}")


def visualize_gyro(results, video_name):
    """可视化小陀螺结果"""
    data = results["data"]
    car_centers = np.array(data["car_centers"]) if data["car_centers"] else None
    omega = np.array(data["omega"]) if data["omega"] else None
    obs = np.array(data["obs"]) if data["obs"] else None
    
    if car_centers is None or len(car_centers) == 0:
        print("  无车体中心数据")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 图1: 装甲板观测位置 vs 车体中心估计
    ax = axes[0]
    if obs is not None:
        ax.plot(obs[:, 0], obs[:, 3], 'b.', markersize=1, alpha=0.3, label='装甲板观测 (x-z)')
    ax.plot(car_centers[:, 0], car_centers[:, 2], 'r.', markersize=3, alpha=0.7, label='车体中心估计')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('z (m)')
    ax.set_title(f'{video_name}: 装甲板轨迹与车体中心')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # 图2: 角速度估计
    ax = axes[1]
    if omega is not None:
        ax.plot(omega[:, 0], omega[:, 1], 'g-', linewidth=1)
        ax.axhline(np.mean(omega[:, 1]), color='r', linestyle='--', 
                   label=f'均值: {np.mean(omega[:, 1]):.2f} rad/s = {np.mean(omega[:, 1])/(2*np.pi):.2f} rev/s')
        ax.set_xlabel('时间 (s)')
        ax.set_ylabel('角速度 (rad/s)')
        ax.set_title(f'旋转角速度估计')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{video_name}_陀螺分析.png", dpi=VIS_CONFIG["figure_dpi"])
    plt.close()
    print(f"  陀螺分析图表已保存: {FIGURES_DIR / f'{video_name}_陀螺分析.png'}")


def main():
    """主入口"""
    print("=" * 60)
    print("装甲板识别与运动预测系统")
    print("=" * 60)
    
    # 加载相机参数
    try:
        K, dist, resolution = load_camera_params()
        print(f"相机参数已加载: {K[0,0]:.1f}, {K[1,1]:.1f}, "
              f"分辨率 {resolution[0]}x{resolution[1]}")
    except Exception as e:
        print(f"⚠️ 相机参数加载失败: {e}")
        print("将仅进行识别，不进行位姿解算")
        K, dist = None, None
    
    all_results = {}
    
    # 处理各视频
    video_order = ["远距离平移", "中距离陀螺", "近距离陀螺", "装甲板_蓝"]#"远距离平移", "中距离陀螺", "近距离陀螺", "装甲板_蓝"
    
    for video_name in video_order:
        config = VIDEO_CONFIGS[video_name]
        has_pose = config["has_pose"]
        
        result = process_video(
            video_name, config,
            K=K if has_pose else None,
            dist=dist if has_pose else None,
        )
        
        if result[0] is not None:
            detections_df, predictions_rows, video_frames, data = result
            
            all_results[video_name] = {
                "detections_df": detections_df,
                "predictions_rows": predictions_rows,
                "video_frames": video_frames,
                "data": data,
            }
            
            # 保存识别视频
            save_recognition_video(video_frames, video_name, fps=VIS_CONFIG["output_fps"])
            
            # 可视化
            if config.get("gyro", False):
                visualize_gyro({"data": data}, video_name)
            elif video_name == "远距离平移":
                visualize_translation({"data": data}, video_name)
    
    print("\n" + "=" * 60)
    print("处理完成！")
    print(f"结果保存在: {OUTPUT_DIR}")
    print(f"图表保存在: {FIGURES_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
