import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# 全局参数
C_max = 12000
W_max = 8000
r_p = 0.15
alpha = 0.002
nu_p = 0.0012
nu_h = 0.0005


def agricultural_model(t, y):
    C, W, H, B, S = y
    day = t

    # 农药驱动函数
    h = 0.85 * np.exp(-8 * (day - 110) ** 2 / 1800) if 90 <= day <= 130 else 0  # 除草剂
    p = 0.92 * np.exp(-8 * (day - 185) ** 2 / 1800) if 165 <= day <= 205 else 0  # 杀虫剂

    # 农作物动态
    if 60 <= day < 270:
        mu_C = 0.015 * (1 - 0.4 * np.cos(2 * np.pi * (day - 80) / 365))
        dCdt = mu_C * C * (1 - (C + 0.5 * W) / C_max) - 0.0008 * H * C - 0.002 * h * C
    else:
        dCdt = 0

    # 杂草动态
    mu_W = 0.012 + 0.008 * np.cos(2 * np.pi * (day - 60) / 365)
    dWdt = mu_W * W * (1 - W / W_max) - 0.0015 * h * W - 0.0003 * (C ** 0.7) * W

    # 害虫动态
    dHdt = r_p * H * (1 - H / 120) - alpha * B * H - 0.08 * p * H

    # 蝙蝠动态（整合双农药影响）
    M = 0.5 + 0.5 * np.cos(2 * np.pi * (day - 180) / 365)  # 繁殖季节调节
    M = np.clip(M, 0, 1)
    functional_response = H / (H + 10)
    mortality = (0.0005 + 0.00001 * B + nu_p * p + nu_h * h) * B
    dBdt = 0.0001 * functional_response * B * M - mortality

    # 土壤动态
    dSdt = 0.00004 * W - 0.0001 * (h + p) * S - 0.00002 * S

    # 防止负值
    B = max(B, 0)
    H = max(H, 0)

    return [dCdt, dWdt, dHdt, dBdt, dSdt]


# 初始条件和求解
y0 = [150, 300, 8, 50, 1.8]  # 初始值


def harvest_event(t, y): return t - 270  # 第270天收割


sol = solve_ivp(agricultural_model, [0, 365], y0, events=harvest_event,
                max_step=1, method='LSODA')

# 收割后处理
if sol.t_events[0].size > 0:
    harvest_idx = np.where(sol.t >= sol.t_events[0][0])[0][0]
    sol.y[0][harvest_idx:] = 0

# 可视化
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(sol.t, sol.y[0], 'g-', label='Crops')
plt.plot(sol.t, sol.y[1], 'brown', label='Weeds')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(sol.t, sol.y[2], 'r-', label='Pests')
plt.twinx().plot(sol.t, sol.y[3], 'k--', label='Bats')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(sol.t, sol.y[4], 'blue', label='Soil Health')
plt.xlabel('Day of Year')
plt.legend()
plt.show()