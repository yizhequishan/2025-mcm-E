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


def temperature_factor(day):
    """ 温度响应函数优化 """
    T = 18 + 12 * np.sin(2 * np.pi * (day - 70) / 365)
    return np.clip((T - 8) / 20, 0.0, 1.0)


def pesticide_application(t):
    day = t
    p_spring = 0.6 * np.exp(-8 * (day - 120) ** 2 / 1800) if 100 <= day <= 140 else 0.0
    p_summer = 0.9 * np.exp(-8 * (day - 185) ** 2 / 1800) if 165 <= day <= 205 else 0.0
    p_autumn = 0.4 * np.exp(-8 * (day - 250) ** 2 / 1800) if 230 <= day <= 270 else 0.0
    return p_spring + p_summer + p_autumn


def herbicide_app(t):
    """ 高效除草剂施用 """
    h_spring = 1.2 * np.exp(-10 * (t - 70) ** 2 / 1800) if 50 <= t <= 90 else 0.0
    h_summer = 0.8 * np.exp(-10 * (t - 150) ** 2 / 1800) if 130 <= t <= 170 else 0.0
    return 0


def agricultural_model(t, y):
    C, W, H, B, S, W_seed = y  # 新增杂草种子库W_seed
    # 杂草种子萌发（春季）
    if 60 <= day < 100:
        W_germination = 0.05 * W_seed
        W += W_germination
        W_seed -= W_germination
    # 杂草种子产生（秋季）
    if 200 <= day < 270:
        W_seed += 0.1 * W


def agricultural_model(t, y):
    C, W, H, B, S, W_seed = y  # 6个状态变量
    day = t

    # 季节驱动函数
    h = herbicide_app(day)
    p = pesticide_application(day)
    M = 0.5 + 0.5 * np.cos(2 * np.pi * (day - 180) / 365)
    M = np.clip(M, 0.0, 1.0)
    # 杂草种子动态

    dW_seed = -0.05 * W_seed  # 自然衰减
    if 60 <= day < 100:
        germination = 0.05 * W_seed
        W += germination
        W_seed -= germination
    if 200 <= day < 270:
        seed_production = 0.1 * W
        W_seed += seed_production

    # 农作物动态
    if 60 <= day < 270:
        if day < 110:  # 苗期延长至110天
            phase_factor = 0.8
        elif 110 <= day < 230:  # 快速生长期延长
            phase_factor = 1.8 * (1 - 0.1 * np.cos(2 * np.pi * (day - 170) / 365))  # 提高峰值
        else:  # 成熟期减少衰减
            phase_factor = 1.0
        mu_C = params['mu_C_base'] * phase_factor * temperature_factor(day)
        if params['enable_pollination']:
            # 计算传粉增益
            poll_effect = params['gamma_P0'] * (1 - np.exp(-params['kappa'] * B))
            poll_saturation = 1 + B / params['b_s']  # 防止增益无限增长
            pollination_gain = poll_effect / poll_saturation
        else:
            pollination_gain = 0.0

        dCdt = mu_C * C * (1 - (C + params['alpha_W'] * W) / params['C_max']) * (1 + pollination_gain) \
               - params['delta_C'] * H * C \
               - params['lambda_C'] * h * C
    else:
        dCdt = 0.0

        # 杂草动态
    mu_W = params['mu_W_min'] + params['mu_W_amp'] * np.cos(2 * np.pi * (day - 120) / 365)
    weed_suppression = params['beta_WC'] * (C ** params['k']) * W
    dWdt = mu_W * W * (1 - W / params['W_max']) - params['gamma_W'] * h * W - weed_suppression
    # 害虫与蝙蝠动态
    dHdt = params['r_H'] * H * (1 - H / params['K_H']) - params['alpha_H'] * (B / 1e6) * H - params['beta_H'] * p * H
    functional_response = H / (H + params['h_s'])
    reproduction = params['gamma_B'] * functional_response * B * M
    mortality = (params['mu_B0'] + params['mu_Bd'] * B + params['nu_p'] * p + params['nu_h'] * h) * B
    dBdt = reproduction - mortality

    # 土壤健康
    dSdt = params['omega_S'] * W - params['eta_S'] * (h + p) * S - params['xi_S'] * S

    return [dCdt, dWdt, dHdt, dBdt, dSdt,dW_seed]


# ================== 模拟设置 ==================
y0 = [500.0, 300.0, 4.0, 80.0, 2.2,2000.0]
sol = solve_ivp(agricultural_model, [0, 365], y0, events=lambda t, y: t - 270, max_step=1.0, method='LSODA')

# 处理收割
if sol.t_events[0].size > 0:
    harvest_idx = np.where(sol.t >= 270)[0][0]
    sol.y[0][harvest_idx:] = 0.0

# ================== 可视化与输出 ==================
plt.figure(figsize=(14, 10))
plt.subplot(3, 1, 1)
plt.plot(sol.t, sol.y[0], 'g-', lw=2, label='Crops')
plt.plot(sol.t, sol.y[1], 'brown', ls='--', label='Weeds')
plt.axvline(270, color='k', linestyle=':', label='Harvest')
plt.ylabel('Biomass (kg/ha)')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 2)
ax1 = plt.gca()
ax1.plot(sol.t, sol.y[2], 'r-', lw=2, label='Pests')
ax1.set_ylabel('Pest Density (ind/m²)', color='r')
ax2 = ax1.twinx()
ax2.plot(sol.t, sol.y[3], 'k--', lw=2, label='Bats')
ax2.set_ylabel('Bat Population (ind/km²)', color='k')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(sol.t, sol.y[4], 'b-', lw=2, label='Soil Health')
plt.xlabel('Day of Year')
plt.ylabel('Organic Matter (%)')
plt.grid(True)

plt.tight_layout()
plt.show()

# 数据保存
daily_days = np.arange(0, 366, 1)
daily_data = {
    "Day": daily_days,
    "Crop_Biomass (kg/ha)": np.interp(daily_days, sol.t, sol.y[0]),
    "Weed_Biomass (kg/ha)": np.interp(daily_days, sol.t, sol.y[1]),
    "Pest_Density (ind/m²)": np.interp(daily_days, sol.t, sol.y[2]),
    "Bat_Population (ind/km²)": np.interp(daily_days, sol.t, sol.y[3]),
    "Soil_Organic_Matter (%)": np.interp(daily_days, sol.t, sol.y[4])
}
daily_data["Crop_Biomass (kg/ha)"][daily_days >= 270] = 0.0

pd.DataFrame(daily_data).to_csv("agricultural_simulation_daily.csv", index=False)

# ================== 数据输出与格式化 ==================
import pandas as pd
from datetime import datetime, timedelta

# 生成日期序列(假设起始年为2023年)
start_date = datetime(2023, 1, 1)
date_sequence = [start_date + timedelta(days=int(day)) for day in sol.t]

# 创建包含日期和生物量的DataFrame
df = pd.DataFrame({
    "Date": [d.strftime("%Y-%m-%d") for d in date_sequence],
    "Day_of_Year": sol.t.astype(int),
    "Crop_Biomass_kg_ha": np.round(sol.y[0], 2),
    "Weed_Biomass_kg_ha": np.round(sol.y[1], 2),
    "Pest_Density_ind_m2": np.round(sol.y[2], 1),
    "Bat_Population_ind_km2": np.round(sol.y[3], 0),
    "Soil_Organic_Matter_pct": np.round(sol.y[4], 3)
})

# 处理收割后的数据
harvest_mask = df["Day_of_Year"] >= 270
df.loc[harvest_mask, "Crop_Biomass_kg_ha"] = 0.0

# 生成完整每日数据(插值)
full_days = np.arange(0, 366)
full_dates = [start_date + timedelta(days=int(day)) for day in full_days]

interpolated_data = {
    "Date": [d.strftime("%Y-%m-%d") for d in full_dates],
    "Day_of_Year": full_days,
    "Crop_Biomass_kg_ha": np.round(np.interp(full_days, sol.t, sol.y[0]), 2),
    "Weed_Biomass_kg_ha": np.round(np.interp(full_days, sol.t, sol.y[1]), 2),
    "Pest_Density_ind_m2": np.round(np.interp(full_days, sol.t, sol.y[2]), 1),
    "Bat_Population_ind_km2": np.round(np.interp(full_days, sol.t, sol.y[3]), 0),
    "Soil_Organic_Matter_pct": np.round(np.interp(full_days, sol.t, sol.y[4]), 3)
}

# 应用收割后修正
interpolated_data["Crop_Biomass_kg_ha"][full_days >= 270] = 0.0

# 创建完整数据框
full_df = pd.DataFrame(interpolated_data)

# 添加生长阶段标记
def growth_phase(day):
    if day < 60:
        return "Pre-planting"
    elif 60 <= day < 110:
        return "Seedling"
    elif 110 <= day < 230:
        return "Rapid Growth"
    elif 230 <= day < 270:
        return "Maturation"
    else:
        return "Post-harvest"

full_df["Growth_Phase"] = full_df["Day_of_Year"].apply(growth_phase)

# 数据验证
assert len(full_df) == 366, "数据长度必须包含闰年所有天数"
assert full_df["Crop_Biomass_kg_ha"].max() <= params["C_max"] * 1.05, "农作物生物量超过最大值"
assert full_df["Weed_Biomass_kg_ha"].max() <= params["W_max"] * 1.05, "杂草生物量超过最大值"

# 格式化输出
float_format = lambda x: f"{x:.2f}"
full_df = full_df.astype({
    "Crop_Biomass_kg_ha": float,
    "Weed_Biomass_kg_ha": float,
    "Soil_Organic_Matter_pct": float
})

# 保存文件
output_path = "crop_system_daily_data"
full_df.to_csv(f"{output_path}.csv", index=False, float_format="%.2f")
full_df.to_excel(f"{output_path}.xlsx", index=False, float_format="%.2f")

print(f"数据已保存至 {output_path}.csv 和 {output_path}.xlsx")
print("\n数据样例：")
print(full_df.head(10).to_string(index=False))