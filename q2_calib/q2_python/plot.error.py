import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. 填入你的实验数据
# ==========================================
# 真实距离 (单位: cm)，例如 50cm, 100cm ... 300cm
real_distances = [50, 100, 150, 200, 250, 300]

# 算法测量距离 (单位: cm)，对应上面的真实距离
# 注意：solvePnP 算出来通常是 mm，记得除以 10 换算成 cm，或者统一单位
measured_distances = [52.3, 98.5, 155.2, 210.8, 245.1, 315.6] 

# ==========================================
# 2. 计算误差
# ==========================================
# 计算绝对误差 (Absolute Error)
abs_errors = [abs(m - r) for m, r in zip(measured_distances, real_distances)]

# 计算相对误差 (Relative Error %) - 可选，通常分析这个更有意义
rel_errors = [(abs(m - r) / r) * 100 for m, r in zip(measured_distances, real_distances)]

# ==========================================
# 3. 绘制曲线
# ==========================================
plt.figure(figsize=(10, 6)) # 设置画布大小

# 绘制绝对误差曲线
plt.plot(real_distances, abs_errors, marker='o', linestyle='-', color='b', label='Absolute Error (cm)')

# 如果你想同时展示相对误差，可以加一条线（使用双Y轴或者归一化，这里简单起见只画绝对误差，或者你可以注释掉上面画下面这个）
# plt.plot(real_distances, rel_errors, marker='s', linestyle='--', color='r', label='Relative Error (%)')

# 添加图表元素
plt.title('Distance Measurement Error vs. Real Distance', fontsize=14)
plt.xlabel('Real Distance (cm)', fontsize=12)
plt.ylabel('Absolute Error (cm)', fontsize=12)
plt.grid(True, which='both', linestyle='--', alpha=0.7) # 添加网格，方便读数
plt.legend()

# 显示每个点的具体数值
for i, txt in enumerate(abs_errors):
    plt.annotate(f'{txt:.1f}', (real_distances[i], abs_errors[i]), textcoords="offset points", xytext=(0,10), ha='center')

# 保存图片到 q2_calib 文件夹
plt.savefig('q2_calib/error_curve.png', dpi=300) # 保存高清图片
plt.show() # 在屏幕上显示窗口

print("图表已保存至 q2_calib/error_curve.png")