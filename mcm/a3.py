import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ================== 参数定义 ==================
params = {
    # 农作物模块
    'C_max': 12500.0,  # 最大生物量（含秸秆） (kg/ha)
    'mu_C_base': 0.021,  # 基础生长率 (day⁻¹)
    'alpha_W': 0.2,  # 杂草竞争系数
    'delta_C': 0.00012,  # 害虫破坏系数
    'lambda_C': 0.00005,  # 除草剂损伤系数
    # 杂草模块
    'W_max': 8000.0,  # 最大杂草生物量 (kg/ha)
    'mu_W_min': 0.023,  # 最小生长率 (day⁻¹)
    'mu_W_amp': 0.007,  # 生长率季节振幅 (day⁻¹)
    'gamma_W': 0.00,  # 除草剂杀灭系数 ((kg/ha)⁻¹day⁻¹)
    'beta_WC': 0.002,  # 作物抑制系数 ((kg/ha)⁻⁰·⁷day⁻¹)
    'k': 0.7,  # 抑制指数

    # 害虫模块
    'r_H': 0.15,  # 固有增长率 (day⁻¹)
    'K_H': 100.0,  # 环境承载力 (个体/m²)
    'alpha_H': 0.002,  # 蝙蝠捕食效率 ((个体·km²)/(day·m²))
    'beta_H': 0.18,  # 杀虫剂致死率 (day⁻¹)

    # 蝙蝠模块
    'gamma_B': 0.0001,  # 最大繁殖率 (day⁻¹)
    'h_s': 10.0,  # 功能响应半饱和常数 (个体/m²)
    'mu_B0': 0.0005,  # 基础死亡率 (day⁻¹)
    'mu_Bd': 0.00001,  # 密度依赖死亡率系数 ((个体/km²)⁻¹day⁻¹)
    'nu_p': 0.002,  # 杀虫剂毒性系数 ((kg/ha)⁻¹day⁻¹)
    'nu_h': 0.0005,  # 除草剂毒性系数 ((kg/ha)⁻¹day⁻¹)

    # 土壤模块
    'omega_S': 0.00004,  # 有机物输入率 (%·(kg/ha)⁻¹day⁻¹)
    'eta_S': 0.0001,  # 化学损害系数 (%·(kg/ha)⁻¹day⁻¹)
    'xi_S': 0.00002  # 自然流失率 (%·day⁻¹)
}
def temperature_factor(day):
    """ 温度对生长的影响（10-25℃为适宜范围） """
    T = 15 + 15*np.sin(2*np.pi*(day-80)/365)  # 模拟温度年周期
    return np.clip((T - 10)/15, 0.0, 1.0)

def agricultural_model(t, y):
    C, W, H, B, S = y
    day = t
# 修改后的杀虫剂施用函数
def pesticide_application(t):
    day = t
    # 春季喷洒（第100-140天，峰值在120天）
    p_spring = 0.6 * np.exp(-8*(day-120)**2/1800) if 100 <= day <= 140 else 0.0
    # 夏季喷洒（第165-205天，峰值在185天）
    p_summer = 0.9 * np.exp(-8*(day-185)**2/1800) if 165 <= day <= 205 else 0.0
    # 秋季喷洒（第230-270天，峰值在250天）
    p_autumn = 0.4 * np.exp(-8*(day-250)**2/1800) if 230 <= day <= 270 else 0.0
    return p_spring + p_summer + p_autumn

# 除草剂施用函数修正
def herbicide_app(t):
    # 春季施用（播种后立即控制杂草）
    h_spring = 1.2 * np.exp(-10*(t-70)**2/1800) if 50 <= t <= 90 else 0.0  # 峰值提前至70天，剂量提高
    # 夏季补充施用
    h_summer = 0.8 * np.exp(-10*(t-150)**2/1800) if 130 <= t <= 170 else 0.0  # 提前至150天
    return h_spring + h_summer


# ================== 模型核心 ==================
def agricultural_model(t, y):
    """ 农业生态系统微分方程组 """
    C, W, H, B, S = y
    day = t

    # ------------------ 季节驱动函数 ------------------
    # 除草剂 (春季施用)
    h = herbicide_app(day)
    # 杀虫剂（新增三次喷洒）
    p = pesticide_application(day)  # 调用新定义的喷洒函数
    # 蝙蝠繁殖季节调节因子 (5-9月)
    M = 0.5 + 0.5 * np.cos(2 * np.pi * (day - 180) / 365)
    M = np.clip(M, 0.0, 1.0)  # 限制在[0,1]范围

    # ------------------ 微分方程计算 ------------------
    # 农作物动态
    # 农作物动态（修正播种时间）
    if 60 <= day < 270:
        if day < 100:  # 苗期（延长至100天）
            phase_factor = 0.7
        elif 100 <= day < 220:  # 快速生长期（延长至220天）
            phase_factor = 1.5 * (1 - 0.2 * np.cos(2 * np.pi * (day - 160) / 365))  # 提高振幅
        else:  # 成熟期（缩短衰减期）
            phase_factor = 0.9  # 衰减减少
        mu_C = params['mu_C_base'] * phase_factor * temperature_factor(day)
        # 生长方程
        dCdt = mu_C * C * (1 - (C + params['alpha_W'] * W) / params['C_max']) \
               - params['delta_C'] * H * C \
               - params['lambda_C'] * h * C
    else:
        dCdt = 0.0
    # 杂草动态
    mu_W =  params['mu_W_min'] + params['mu_W_amp'] * np.cos(2 * np.pi * (day - 100) / 365)  # 原为day-60
    if day < 270:
        weed_suppression = params['beta_WC'] * (C ** params['k']) * W
    else:
        weed_suppression = 0.0
    dWdt = mu_W * W * (1 - W / params['W_max']) \
           - params['gamma_W'] * h * W \
           - weed_suppression
    # 害虫动态
    dHdt = (
            params['r_H'] * H * (1 - H / params['K_H'])
            - params['alpha_H'] * (B / 1e6) * H  # 单位修正：B转换为个体/m²
            - params['beta_H'] * p * H
    )
    # 蝙蝠动态
    functional_response = H / (H + params['h_s'])
    reproduction = params['gamma_B'] * functional_response * B * M
    mortality = (params['mu_B0']
                 + params['mu_Bd'] * B
                 + params['nu_p'] * p
                 + params['nu_h'] * h) * B
    dBdt = reproduction - mortality

    # 土壤健康动态
    dSdt = (params['omega_S'] * W
            - params['eta_S'] * (h + p) * S
            - params['xi_S'] * S)

    # 防止负值
    B = max(B, 0.0)
    H = max(H, 0.0)

    return [dCdt, dWdt, dHdt, dBdt, dSdt]


# ================== 模拟设置 ==================
# 初始条件 (春季开始)
y0 = [
    180.0,    # C: 农作物初始生物量 (kg/ha) → 幼苗期
    80.0,   # W: 杂草初始生物量 (kg/ha) → 早期杂草生长较快
    8.0,     # H: 害虫密度 (个体/m²)
    50.0,    # B: 蝙蝠种群 (个体/km²)
    1.8      # S: 土壤有机质 (%)
]


# 定义收割事件（第270天）
def harvest_event(t, y):
    return t - 270.0


harvest_event.terminal = True

# 求解ODE
sol = solve_ivp(agricultural_model,
                t_span=[0, 365],
                y0=y0,
                events=harvest_event,
                max_step=1.0,
                method='LSODA')  # 关键修正：移除vectorized=True

# 处理收割事件
if sol.t_events[0].size > 0:
    harvest_day = sol.t_events[0][0]
    harvest_idx = np.where(sol.t >= harvest_day)[0][0]
    sol.y[0][harvest_idx:] = 0.0  # 农作物生物量清零

# ================== 可视化 ==================
plt.figure(figsize=(14, 10))

# 农作物与杂草
plt.subplot(3, 1, 1)
plt.plot(sol.t, sol.y[0], 'g-', lw=2, label='Crops')
plt.plot(sol.t, sol.y[1], 'brown', ls='--', label='Weeds')
plt.axvline(270, color='k', linestyle=':', label='Harvest')
plt.ylabel('Biomass (kg/ha)')
plt.legend()
plt.grid(True)

# 害虫与蝙蝠
plt.subplot(3, 1, 2)
ax1 = plt.gca()
ax1.plot(sol.t, sol.y[2], 'r-', lw=2, label='Pests')
ax1.set_ylabel('Pest Density (ind/m²)', color='r')
ax1.tick_params(axis='y', labelcolor='r')
ax2 = ax1.twinx()
ax2.plot(sol.t, sol.y[3], 'k--', lw=2, label='Bats')
ax2.set_ylabel('Bat Population (ind/km²)', color='k')
ax2.tick_params(axis='y', labelcolor='k')
plt.grid(True)

# 土壤健康
plt.subplot(3, 1, 3)
plt.plot(sol.t, sol.y[4], 'b-', lw=2, label='Soil Health')
plt.xlabel('Day of Year')
plt.ylabel('Organic Matter (%)')
plt.grid(True)

plt.tight_layout()
plt.show()

# ================== 数据输出 ==================
import pandas as pd

# 生成每日时间点（0到365天，步长1）
daily_days = np.arange(0, 366, 1)

# 线性插值获取每日数据
daily_data = {
    "Day": daily_days,
    "Crop_Biomass (kg/ha)": np.interp(daily_days, sol.t, sol.y[0]),
    "Weed_Biomass (kg/ha)": np.interp(daily_days, sol.t, sol.y[1]),
    "Pest_Density (ind/m²)": np.interp(daily_days, sol.t, sol.y[2]),
    "Bat_Population (ind/km²)": np.interp(daily_days, sol.t, sol.y[3]),
    "Soil_Organic_Matter (%)": np.interp(daily_days, sol.t, sol.y[4])
}

# 强制修正收割后的农作物生物量为0
harvest_mask = daily_days >= 270
daily_data["Crop_Biomass (kg/ha)"][harvest_mask] = 0.0

# 创建DataFrame
df = pd.DataFrame(daily_data)

# 保存为CSV和Excel
df.to_csv("a5.csv", index=False)
df.to_excel("a5.xlsx", index=False)

print("每日数据已保存至 agricultural_simulation_daily.csv 和 .xlsx 文件")