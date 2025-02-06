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
    'r_H': 0.18,
    'K_H': 100.0,
    'alpha_H': 0.003,
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
    'gamma_P0': 0.45,  # 最大传粉增益率 (无因次)
    'kappa': 0.0025,  # 传粉效率响应系数 (km²/ind)
    'b_s': 30.0,  # 传粉增益半饱和密度 (ind/km²)
    'theta': 0.65,  # 捕食时间分配比例

    # 可选功能调节
    'enable_pollination': True,  # 是否启用传粉功能

    # 新增蜜蜂参数
    'gamma_E': 0.0004, 'beta_EW': 0.2, 'phi_s': 500.0,
    'mu_E0': 0.0008, 'mu_Ed': 0.00003, 'nu_Ep': 0.003,
    'gamma_PE': 0.45, 'e_s': 20.0
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
    C, W, H, B, S, W_seed, E = y  # 7个状态变量
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
    # === 蜜蜂动态 ===
    nectar = (C + params['beta_EW'] * W) / (C + params['beta_EW'] * W + params['phi_s'])
    dEdt = params['gamma_E'] * nectar * E - (params['mu_E0'] + params['mu_Ed'] * E + params['nu_Ep'] * p) * E

    return [dCdt, dWdt, dHdt, dBdt, dSdt, dW_seed, dEdt]  # 新增dEdt


# ================== 情景模拟函数 ==================
def run_scenario(scenario='both'):
    """ 运行指定情景的模拟 """
    sim_params = params.copy()

    # 根据情景调整参数
    if scenario == 'both':
        # 启用蝙蝠和蜜蜂传粉
        sim_params.update({
            'enable_pollination': True,
            'gamma_P0': 0.35,  # 蝙蝠传粉
            'gamma_PE': 0.25  # 蜜蜂传粉
        })
    elif scenario == 'bat_only':
        # 仅蝙蝠传粉
        sim_params.update({
            'enable_pollination': True,
            'gamma_PE': 0.0  # 禁用蜜蜂传粉
        })
    elif scenario == 'bee_only':
        # 仅蜜蜂传粉
        sim_params.update({
            'enable_pollination': False,  # 禁用蝙蝠传粉增益
            'gamma_PE': 0.25,
            'theta': 1.0  # 蝙蝠100%时间捕食
        })
    elif scenario == 'none':
        # 无传粉者
        sim_params.update({
            'enable_pollination': False,
            'gamma_PE': 0.0,
            'theta': 1.0
        })

    # 初始条件 [C, W, H, B, S, W_seed, E]
    y0 = [500.0, 300.0, 8.0,100, 1.8, 3000.0, 200.0]

    # 运行模拟
    sol = solve_ivp(
        lambda t, y: agricultural_model(t, y, sim_params),
        [0, 365], y0,
        events=lambda t, y: t - 270,
        max_step=1.0,
        method='LSODA'
    )

    # 处理收割
    if sol.t_events[0].size > 0:
        harvest_idx = np.where(sol.t >= 270)[0][0]
        sol.y[0][harvest_idx:] = 0.0
    return sol


# ================== 执行所有情景模拟 ==================
scenarios = ['both', 'bat_only', 'bee_only', 'none']
results = {s: run_scenario(s) for s in scenarios}

import numpy as np
import matplotlib.pyplot as plt

# ================== 数据准备 ==================
categories = ['产量稳定性', '害虫控制', '传粉冗余度', '土壤健康', '经济收益']
labels = np.array(categories)
num_vars = len(categories)

# 各情景数据 (需替换为实际值)
scenario_data = {
    '蝙蝠+蜜蜂': [85, 92, 0.88, 0.82, 4.2],
    '仅蝙蝠': [78, 85, 0.65, 0.75, 3.8],
    '仅蜜蜂': [72, 68, 0.70, 0.68, 3.5],
    '无传粉者': [60, 45, 0.00, 0.55, 2.1]
}


# 数据标准化（不同量纲需分别处理）
def normalize(data, ranges):
    """ 将各指标缩放到0-1区间 """
    norm_data = []
    for d, (min_val, max_val) in zip(data, ranges):
        norm = (d - min_val) / (max_val - min_val)
        norm_data.append(norm)
    return norm_data


# 定义各指标原始范围（根据实际数据调整）
data_ranges = [
    (50, 100),  # 产量稳定性（假设最小50%，最大100%）
    (40, 100),  # 害虫控制率
    (0, 1),  # 传粉冗余度
    (0.5, 1),  # 土壤健康指数
    (0, 5)  # 经济收益
]

# 标准化所有数据
norm_scenarios = {}
for name, data in scenario_data.items():
    norm_scenarios[name] = normalize(data, data_ranges)

# ================== 雷达图绘制 ==================
# 设置角度（将圆周分为N等分）
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # 闭合图形

# 创建子图并设置极坐标
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# 定义颜色和样式
colors = {
    '蝙蝠+蜜蜂': '#2ca02c',
    '仅蝙蝠': '#9467bd',
    '仅蜜蜂': '#ff7f0e',
    '无传粉者': '#d62728'
}
linestyles = ['-', '--', '-.', ':']

# 绘制各情景
for idx, (scenario, data) in enumerate(norm_scenarios.items()):
    # 闭合数据（首尾重复）
    plot_data = data + data[:1]

    # 绘制线条
    ax.plot(angles, plot_data,
            color=colors[scenario],
            linestyle=linestyles[idx],
            linewidth=2,
            label=scenario)

    # 填充颜色
    ax.fill(angles, plot_data,
            color=colors[scenario],
            alpha=0.1)

# 设置坐标轴标签
ax.set_theta_offset(np.pi / 2)  # 起始角度设为12点钟方向
ax.set_theta_direction(-1)  # 顺时针方向
ax.set_rlabel_position(0)  # 半径标签位置

# 设置圆周标签（指标名称）
plt.xticks(angles[:-1], labels, fontsize=10)

# 设置半径刻度（显示原始值）
r_ticks = [0.2, 0.4, 0.6, 0.8, 1.0]
ax.set_rticks(r_ticks)
ax.set_yticklabels([
    f'{int(tick * (r[1] - r[0]) + r[0])}'
    for tick, r in zip(r_ticks, data_ranges)
], fontsize=8)

# 添加图例和标题
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.title('传粉策略生态系统效益雷达图', y=1.15, fontsize=14)

# 保存高清图片
plt.savefig('pollination_radar.png', dpi=300, bbox_inches='tight')
plt.show()