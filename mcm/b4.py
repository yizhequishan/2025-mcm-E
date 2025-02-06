import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd

# ================== 参数定义 ==================
params = {
    # 基础模块
    'C_max': 12000.0,  # 农作物最大生物量 (kg/ha)
    'W_max': 8000.0,  # 杂草最大生物量 (kg/ha)
    'K_H': 100.0,  # 害虫环境承载力 (ind/m²)

    # 农作物模块
    'mu_C_base': 0.014,  # 基础生长率 (day⁻¹)
    'alpha_W': 0.3,  # 杂草竞争系数
    'delta_C': 0.000165,  # 害虫破坏系数
    'lambda_C': 0.0001,  # 除草剂损伤系数

    # 杂草模块
    'mu_W_min': 0.02,  # 最小生长率 (day⁻¹)
    'mu_W_amp': 0.008,  # 生长率季节振幅
    'gamma_W': 0.006,  # 除草剂杀灭系数
    'beta_WC': 0.0003,  # 作物抑制系数
    'k': 0.7,  # 抑制指数

    # 害虫模块
    'r_H': 0.15,  # 固有增长率 (day⁻¹)
    'alpha_H': 0.002,  # 蝙蝠捕食效率
    'beta_H': 0.18,  # 杀虫剂致死率

    # 蝙蝠模块
    'gamma_B': 0.0001,  # 最大繁殖率 (day⁻¹)
    'h_s': 10.0,  # 功能响应半饱和
    'mu_B0': 0.0005,  # 基础死亡率
    'mu_Bd': 0.00001,  # 密度依赖死亡率
    'nu_p': 0.002,  # 杀虫剂毒性系数

    # 土壤模块
    'omega_S': 0.00004,  # 有机物输入率
    'eta_S': 0.0001,  # 化学损害系数
    'xi_S': 0.00002,  # 自然流失率

    # 食蚜蝇模块
    'gamma_Hf': 0.00015,  # 最大繁殖率
    'h_s_Hf': 15.0,  # 功能响应半饱和
    'mu_Hf0': 0.0007,  # 基础死亡率
    'mu_Hfd': 0.000015,  # 密度依赖死亡率
    'nu_p_Hf': 0.0025,  # 杀虫剂毒性系数
    'alpha_Ow_Hf': 0.0001,  # 林鸮捕食系数

    # 林鸮模块
    'gamma_Ow': 0.00003,  # 最大繁殖率
    'beta_B': 0.6,  # 蝙蝠营养贡献
    'beta_Hf': 0.4,  # 食蚜蝇营养贡献
    'h_s_Ow': 50.0,  # 功能响应半饱和
    'mu_Ow0': 0.0003,  # 基础死亡率
    'mu_Owd': 0.000008,  # 密度依赖死亡率

    # 相互作用参数
    'alpha_Hf': 0.0015,  # 食蚜蝇捕食效率
    'alpha_Ow_B': 0.00008  # 林鸮捕食蝙蝠
}


# ================== 辅助函数 ==================
def herbicide_app(t):
    """ 除草剂施用函数 """
    h_spring = 0.85 * np.exp(-8 * (t - 100) ** 2 / 1800) if 90 <= t <= 130 else 0.0
    h_summer = 0.6 * np.exp(-8 * (t - 180) ** 2 / 1800) if 160 <= t <= 200 else 0.0
    return h_spring + h_summer


def pesticide_application(t):
    """ 杀虫剂施用函数 """
    p_spring = 0.6 * np.exp(-8 * (t - 120) ** 2 / 1800) if 100 <= t <= 140 else 0.0
    p_summer = 0.9 * np.exp(-8 * (t - 185) ** 2 / 1800) if 165 <= t <= 205 else 0.0
    p_autumn = 0.4 * np.exp(-8 * (t - 250) ** 2 / 1800) if 230 <= t <= 270 else 0.0
    return p_spring + p_summer + p_autumn


# ================== 模型核心 ==================
def agricultural_model(t, y):
    """ 七维生态系统微分方程组 """
    C, W, H, B, S, Hf, Ow = y

    # 环境输入
    h = herbicide_app(t)
    p = pesticide_application(t)

    # 季节调节因子
    M_bat = np.clip(0.5 + 0.5 * np.cos(2 * np.pi * (t - 180) / 365), 0, 1)
    M_hf = np.clip(0.6 + 0.4 * np.sin(2 * np.pi * (t - 150) / 365), 0, 1)
    M_ow = np.clip(0.7 - 0.3 * np.cos(2 * np.pi * (t - 210) / 365), 0, 1)

    # 农作物动态
    if 0 <= t < 270:
        mu_C = params['mu_C_base'] * (1 - 0.4 * np.cos(2 * np.pi * (t - 80) / 365))
        dCdt = mu_C * C * (1 - (C + params['alpha_W'] * W) / params['C_max']) \
               - params['delta_C'] * H * C \
               - params['lambda_C'] * h * C
    else:
        dCdt = 0.0

    # 杂草动态
    mu_W = params['mu_W_min'] + params['mu_W_amp'] * np.cos(2 * np.pi * (t - 60) / 365)
    weed_supp = params['beta_WC'] * (C ** params['k']) * W if t < 270 else 0.0
    dWdt = mu_W * W * (1 - W / params['W_max']) \
           - params['gamma_W'] * h * W \
           - weed_supp

    # 害虫动态
    dHdt = params['r_H'] * H * (1 - H / params['K_H']) \
           - (params['alpha_H'] * B + params['alpha_Hf'] * Hf) * H / 1e6 \
           - params['beta_H'] * p * H

    # 蝙蝠动态
    bat_reprod = params['gamma_B'] * (H / (H + params['h_s'])) * B * M_bat
    bat_mort = (params['mu_B0'] + params['mu_Bd'] * B + params['nu_p'] * p) * B
    bat_pred = params['alpha_Ow_B'] * Ow * B / 1e6
    dBdt = bat_reprod - bat_mort - bat_pred

    # 食蚜蝇动态
    hf_reprod = params['gamma_Hf'] * (H / (H + params['h_s_Hf'])) * Hf * M_hf
    hf_mort = (params['mu_Hf0'] + params['mu_Hfd'] * Hf + params['nu_p_Hf'] * p) * Hf
    hf_pred = params['alpha_Ow_Hf'] * Ow * Hf / 1e6
    dHfdt = hf_reprod - hf_mort - hf_pred

    # 林鸮动态
    prey_biomass = params['beta_B'] * B + params['beta_Hf'] * Hf
    ow_reprod = params['gamma_Ow'] * (prey_biomass / (prey_biomass + params['h_s_Ow'])) * Ow * M_ow
    ow_mort = (params['mu_Ow0'] + params['mu_Owd'] * Ow) * Ow
    dOwdt = ow_reprod - ow_mort

    # 土壤动态
    dSdt = params['omega_S'] * W - params['eta_S'] * (h + p) * S - params['xi_S'] * S

    # 数值稳定性处理
    return [max(x, 0) for x in [dCdt, dWdt, dHdt, dBdt, dSdt, dHfdt, dOwdt]]


# ================== 模拟设置 ==================
def harvest_event(t, y):
    """ 收割事件检测 """
    return t - 270.0


harvest_event.terminal = True

# 初始条件
y0 = [
    500.0,  # C: 农作物
    100.0,  # W: 杂草
    8.0,  # H: 害虫
    50.0,  # B: 蝙蝠
    1.8,  # S: 土壤
    300.0,  # Hf: 食蚜蝇
    5.0  # Ow: 林鸮
]

# ================== 模型求解 ==================
sol = solve_ivp(agricultural_model,
                t_span=[0, 365],
                y0=y0,
                events=harvest_event,
                max_step=1,
                method='LSODA')

# 收割后处理
if sol.t_events[0].size > 0:
    harvest_day = sol.t_events[0][0]
    harvest_idx = np.where(sol.t >= harvest_day)[0][0]
    sol.y[0][harvest_idx:] = 0.0


# ================== 结果分析 ==================
def stability_analysis(data):
    """ 计算生态系统稳定性指数 """
    cv = np.std(data, axis=1) / np.mean(data, axis=1)
    return 1 / np.mean(cv[np.isfinite(cv)])


print(f"系统稳定性指数: {stability_analysis(sol.y):.2f}")

# ================== 可视化 ==================
plt.figure(figsize=(14, 10))

# 生物量动态
plt.subplot(3, 1, 1)
plt.plot(sol.t, sol.y[0], 'g-', label='Crops')
plt.plot(sol.t, sol.y[1], 'brown--', label='Weeds')
plt.axvline(270, color='k', linestyle=':', label='Harvest')
plt.ylabel('Biomass (kg/ha)')
plt.legend()

# 天敌动态
plt.subplot(3, 1, 2)
plt.plot(sol.t, sol.y[3], 'b-', label='Bats')
plt.plot(sol.t, sol.y[5], 'm--', label='Hoverflies')
plt.plot(sol.t, sol.y[6], 'k-.', label='Owls')
plt.ylabel('Population (ind/km²)')
plt.legend()

# 害虫与土壤
plt.subplot(3, 1, 3)
ax1 = plt.gca()
ax1.plot(sol.t, sol.y[2], 'r-', label='Pests')
ax1.set_ylabel('Pest Density (ind/m²)', color='r')
ax1.tick_params(axis='y', labelcolor='r')
ax2 = ax1.twinx()
ax2.plot(sol.t, sol.y[4], 'g--', label='Soil Health')
ax2.set_ylabel('Organic Matter (%)', color='g')
ax2.tick_params(axis='y', labelcolor='g')

plt.tight_layout()
plt.show()


# ================== 可行性验证 ==================
def feasibility_check(results):
    """ 执行生态合理性检查 """
    valid = True
    reasons = []

    # 检查负值
    if np.any(results.y < -1e-6):
        valid = False
        reasons.append("Negative population detected")

    # 检查种群崩溃
    extinction_threshold = 1e-3
    if np.any(results.y[3:] < extinction_threshold):
        valid = False
        reasons.append("Species extinction detected")

    # 检查害虫爆发
    if np.max(results.y[2]) > 150:
        valid = False
        reasons.append("Pest outbreak exceeding threshold")

    # 检查土壤退化
    if results.y[4, -1] < 0.5 * results.y[4, 0]:
        valid = False
        reasons.append("Severe soil degradation")

    return valid, reasons


is_valid, issues = feasibility_check(sol)
print(f"模型可行性: {'通过' if is_valid else '未通过'}")
if not is_valid:
    print("存在问题:")
    for issue in issues:
        print(f"- {issue}")
else:
    print("所有生态合理性检查通过")

# ================== 数据导出 ==================
df = pd.DataFrame({
    'Day': sol.t,
    'Crop': sol.y[0],
    'Weed': sol.y[1],
    'Pest': sol.y[2],
    'Bat': sol.y[3],
    'Soil': sol.y[4],
    'Hoverfly': sol.y[5],
    'Owl': sol.y[6]
})
df.to_csv('agricultural_model_results.csv', index=False)