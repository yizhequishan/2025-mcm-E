import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd

# ================== 参数定义（最终修正版） ==================
params = {
    # 农作物模块
    'C_max': 12500.0,
    'mu_C_base': 0.025,  # 基础生长率 (day⁻¹)
    'alpha_W': 0.5,  # 杂草竞争系数
    'delta_C': 0.00012,  # 害虫破坏系数
    'lambda_C': 0.0000,  # 除草剂损伤系数

    # 杂草模块
    'W_max': 8000.0,
    'mu_W_min': 0.028,  # 降低最小生长率
    'mu_W_amp': 0.007,
    'gamma_W': 0.00,  # 恢复除草剂效率
    'beta_WC': 0.002,  # 作物抑制系数
    'k': 0.7,

    # 害虫模块
    'r_H': 0.15,
    'K_H': 100.0,
    'alpha_H': 0.002,
    'beta_H': 0.18,

    # 蝙蝠模块
    'gamma_B': 0.0001,
    'h_s': 10.0,
    'mu_B0': 0.0005,
    'mu_Bd': 0.00001,
    'nu_p': 0.002,
    'nu_h': 0.0005,

    # 土壤模块
    'omega_S': 0.00004,
    'eta_S': 0.0001,
    'xi_S': 0.00002,

    # 传粉相关参数
    'gamma_P0': 0.35,  # 最大传粉增益率 (无因次)
    'kappa': 0.0025,  # 传粉效率响应系数 (km²/ind)
    'b_s': 45.0,  # 传粉增益半饱和密度 (ind/km²)
    'theta': 0.65,  # 捕食时间分配比例

    # 可选功能调节
    'enable_pollination': True  # 是否启用传粉功能
}


# ================== 辅助函数 ==================
def temperature_factor(day):
    T = 18 + 12 * np.sin(2 * np.pi * (day - 70) / 365)
    return np.clip((T - 8) / 20, 0.0, 1.0)


def pesticide_application(t):
    day = t
    p_spring = 0.6 * np.exp(-8 * (day - 120) ** 2 / 1800) if 100 <= day <= 140 else 0.0
    p_summer = 0.9 * np.exp(-8 * (day - 185) ** 2 / 1800) if 165 <= day <= 205 else 0.0
    p_autumn = 0.4 * np.exp(-8 * (day - 250) ** 2 / 1800) if 230 <= day <= 270 else 0.0
    return p_spring + p_summer + p_autumn


def herbicide_app(t):
    return 0  # 除草剂未启用


# ================== 核心模型 ==================
def agricultural_model(t, y, params):
    C, W, H, B, S, W_seed = y  # 6个状态变量
    day = t

    # 初始化所有导数项
    dCdt, dWdt, dHdt, dBdt, dSdt, dW_seed = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    # 季节驱动参数
    h = herbicide_app(day)
    p = pesticide_application(day)
    M = 0.5 + 0.5 * np.cos(2 * np.pi * (day - 180) / 365)
    M = np.clip(M, 0.0, 1.0)

    # === 杂草种子动态 ===
    dW_seed = -0.05 * W_seed  # 自然衰减

    # 春季萌发 (60-100天)
    if 60 <= day < 100:
        germination = 0.05 * W_seed
        dWdt += germination  # 杂草增加
        dW_seed -= germination  # 种子减少

    # 秋季产种 (200-270天)
    if 200 <= day < 270:
        seed_production = 0.1 * W
        dW_seed += seed_production

    # === 农作物动态 ===
    if 60 <= day < 270:
        # 生长阶段调节
        if day < 110:
            phase_factor = 0.8
        elif 110 <= day < 230:
            phase_factor = 1.8 * (1 - 0.1 * np.cos(2 * np.pi * (day - 170) / 365))
        else:
            phase_factor = 1.0

        mu_C = params['mu_C_base'] * phase_factor * temperature_factor(day)

        # 传粉增益计算
        if params['enable_pollination']:
            poll_effect = params['gamma_P0'] * (1 - np.exp(-params['kappa'] * B))
            poll_saturation = 1 + B / params['b_s']
            pollination_gain = poll_effect / poll_saturation
        else:
            pollination_gain = 0.0

        dCdt = mu_C * C * (1 - (C + params['alpha_W'] * W) / params['C_max']) * (1 + pollination_gain) \
               - params['delta_C'] * H * C - params['lambda_C'] * h * C
    else:
        dCdt = 0.0

    # === 杂草动态 ===
    mu_W = params['mu_W_min'] + params['mu_W_amp'] * np.cos(2 * np.pi * (day - 120) / 365)
    weed_suppression = params['beta_WC'] * (C ** params['k']) * W
    dWdt += mu_W * W * (1 - W / params['W_max']) - params['gamma_W'] * h * W - weed_suppression

    # === 害虫动态 ===
    dHdt = params['r_H'] * H * (1 - H / params['K_H']) - params['alpha_H'] * (B / 1e6) * H - params['beta_H'] * p * H

    # === 蝙蝠动态 ===
    functional_response = params['theta'] * H / (H + params['h_s'])
    reproduction = params['gamma_B'] * functional_response * B * M
    mortality = (params['mu_B0'] + params['mu_Bd'] * B + params['nu_p'] * p + params['nu_h'] * h) * B
    dBdt = reproduction - mortality

    # === 土壤健康 ===
    dSdt = params['omega_S'] * W - params['eta_S'] * (h + p) * S - params['xi_S'] * S

    return [dCdt, dWdt, dHdt, dBdt, dSdt, dW_seed]


# ================== 模拟函数 ==================
def run_simulation(enable_poll=True):
    sim_params = params.copy()
    sim_params.update({
        'enable_pollination': enable_poll,
        'theta': 0.65 if enable_poll else 1.0
    })

    y0 = [500.0, 300.0, 4.0, 80.0, 2.2, 2000.0]

    sol = solve_ivp(
        lambda t, y: agricultural_model(t, y, sim_params),
        [0, 365], y0,
        events=lambda t, y: t - 270,
        max_step=1.0,
        method='LSODA'
    )

    # 收割处理
    if sol.t_events[0].size > 0:
        harvest_idx = np.where(sol.t >= 270)[0][0]
        sol.y[0][harvest_idx:] = 0.0
    return sol


# ================== 结果可视化 ==================
def plot_comparison(sol_with, sol_without):
    plt.figure(figsize=(14, 12))

    # 农作物对比
    plt.subplot(3, 1, 1)
    plt.plot(sol_with.t, sol_with.y[0], 'g-', label='With Pollination')
    plt.plot(sol_without.t, sol_without.y[0], 'r--', label='Without Pollination')
    plt.axvline(270, c='k', ls=':', label='Harvest')
    plt.ylabel('Crop Biomass (kg/ha)')
    plt.legend()
    plt.grid(True)

    # 害虫与蝙蝠对比
    plt.subplot(3, 1, 2)
    plt.plot(sol_with.t, sol_with.y[2], 'm-', label='Pests (With)')
    plt.plot(sol_without.t, sol_without.y[2], 'm--', label='Pests (Without)')
    plt.ylabel('Pest Density (ind/m²)', color='m')
    ax2 = plt.gca().twinx()
    ax2.plot(sol_with.t, sol_with.y[3], 'b-', label='Bats (With)')
    ax2.plot(sol_without.t, sol_without.y[3], 'b--', label='Bats (Without)')
    plt.ylabel('Bat Population (ind/km²)', color='b')
    plt.legend(loc='upper left')
    plt.grid(True)

    # 土壤健康对比
    plt.subplot(3, 1, 3)
    plt.plot(sol_with.t, sol_with.y[4], 'c-', label='With Pollination')
    plt.plot(sol_without.t, sol_without.y[4], 'orange', label='Without Pollination')
    plt.ylabel('Soil Organic Matter (%)')
    plt.xlabel('Day of Year')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# ================== 执行模拟 ==================
if __name__ == "__main__":
    sol_with = run_simulation(enable_poll=True)
    sol_without = run_simulation(enable_poll=False)
    plot_comparison(sol_with, sol_without)