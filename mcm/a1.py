import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# 全局参数
C_max = 12000  # kg/ha
W_max = 8000  # kg/ha
r_p = 0.15  # 害虫日增长率
K_p = 120  # 害虫承载力 (个体/m²)
alpha = 0.002  # 蝙蝠捕食效率
gamma_bat = 0.0012  # 蝙蝠繁殖率
mu_B0 = 0.0005  # 基础死亡率
mu_Bd = 0.00001  # 密度依赖系数
nu = 0.0012  # 杀虫剂毒性系数
h_s = 10  # 功能响应半饱和常数
nu_pesticide = 0.0012  # 杀虫剂毒性系数
nu_herbicide = 0.0005  # 除草剂毒性系数
# 修正参数
W_max = 4000  # 杂草最大生物量 (kg/ha)
gamma_herbicide = 0.008  # 除草剂杀灭效率
crop_suppression = 0.0008  # 作物抑制系数


def weed_dynamics(t, W, C, h):
    # 季节生长率
    mu_W = 0.008 + 0.003 * np.cos(2 * np.pi * (t - 60) / 365)

    # 生长项
    growth = mu_W * W * (1 - W / W_max)

    # 控制项
    herbicide_effect = gamma_herbicide * h * W
    crop_effect = crop_suppression * (C ** 0.7) * W

    dWdt = growth - herbicide_effect - crop_effect
    return dWdt






def agricultural_model(t, y):
    C, W, H, B, S = y
    day = t

    # ========== 季节性驱动函数 ==========
    # 除草剂 (春季)
    h = 0.85 * np.exp(-8 * (day - 110) ** 2 / 1800) if 90 <= day <= 130 else 0

    # 杀虫剂 (夏季)
    p = 0.92 * np.exp(-8 * (day - 185) ** 2 / 1800) if 165 <= day <= 205 else 0

    # 蝙蝠繁殖季节调节 (5-9月)
    M = 0.5 + 0.5 * np.cos(2 * np.pi * (day - 180) / 365)
    M = np.clip(M, 0, 1)  # 限制在[0,1]

    # ========== 各模块动态方程 ==========


   # == == 除草剂双阶段施用 == ==
    if 60 <= day <= 100:  # 春季芽前除草
        h = 1.2 * np.exp(-10 * (day - 80) ** 2 / 1600)
    elif 150 <= day <= 190:  # 夏季芽后除草
        h = 0.8 * np.exp(-10 * (day - 170) ** 2 / 1600)
    else:
        h = 0

    # ==== 作物动态 ====
    if 60 <= day < 270:
        mu_C = 0.018 * (1 - 0.3 * np.cos(2 * np.pi * (day - 75) / 365))
        dCdt = mu_C * C * (1 - (C + 0.8 * W) / 12000) - 0.0008 * H * C - 0.002 * h * C
    else:
        dCdt = 0

    # ==== 杂草动态 ====
    mu_W = 0.008 + 0.004 * np.cos(2 * np.pi * (day - 60) / 365)
    dWdt = mu_W * W * (1 - W / 8000) - 0.003 * h * W - 0.0005 * (C ** 1.2) * W
    # --- 害虫 ---
    dHdt = r_p * H * (1 - H / K_p) - alpha * B * H - 0.08 * p * H

    # 蝙蝠动态（整合双农药影响）
    mortality = (0.0005 + 0.00001 * B + nu_pesticide * p + nu_herbicide * h) * B
    dBdt = 0.0001 * (H / (H + 10)) * B * M - mortality

    # --- 土壤 ---
    dSdt = 0.00004 * W - 0.0001 * (h + p) * S - 0.00002 * S

    # 防止负值
    B = max(B, 0)
    H = max(H, 0)


    return [dCdt, dWdt, dHdt, dBdt, dSdt]


# 初始条件
y0 = [300, 150, 8, 50, 1.8]  # C=150kg/ha, W=300kg/ha, H=8/m², B=50/km², S=1.8%


# 求解配置
def harvest_event(t, y):
    return t - 270


harvest_event.terminal = True

sol = solve_ivp(agricultural_model, [0, 365], y0, events=harvest_event,
                max_step=1, method='LSODA')

# 收割后处理
if sol.t_events[0].size > 0:
    harvest_idx = np.where(sol.t >= sol.t_events[0][0])[0][0]
    sol.y[0][harvest_idx:] = 0  # 农作物清零

# ==================== 结果可视化 ====================
# 绘制综合动态图
fig, axs = plt.subplots(3, 1, figsize=(12, 12))

# 农作物与杂草
axs[0].plot(sol.t, sol.y[0], 'g-', label='Crops')
axs[0].plot(sol.t, sol.y[1], 'brown', label='Weeds')
axs[0].set_ylabel('Biomass (kg/ha)')
axs[0].legend()
axs[0].grid(True)

# 害虫与蝙蝠
ax1 = axs[1]
ax1.plot(sol.t, sol.y[2], 'r-', label='Pests')
ax1.set_ylabel('Pest Density (ind/m²)', color='r')
ax1.tick_params(axis='y', labelcolor='r')
ax2 = ax1.twinx()
ax2.plot(sol.t, sol.y[3], 'k--', label='Bats')
ax2.set_ylabel('Bat Density (ind/km²)', color='k')
ax2.tick_params(axis='y', labelcolor='k')
axs[1].grid(True)

# 土壤健康
axs[2].plot(sol.t, sol.y[4], 'blue', label='Soil Health')
axs[2].set_ylabel('Organic Matter (%)')
axs[2].set_xlabel('Day of Year')
axs[2].grid(True)

plt.tight_layout()
plt.show()

# ==================== 结果分析 ====================
print("关键指标统计：")
print(f"作物峰值产量：{np.max(sol.y[0]):.0f} kg/ha")
print(f"最大杂草生物量：{np.max(sol.y[1]):.0f} kg/ha")
print(f"害虫峰值密度：{np.max(sol.y[2]):.1f} 个体/m²")
print(f"蝙蝠年存活率：{(sol.y[3][-1]/y0[3]*100):.1f}%")
print(f"土壤有机质年变化：{(sol.y[4][-1]-y0[4]):.2f}%")