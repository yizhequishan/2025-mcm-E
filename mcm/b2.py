import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt


# ================== 参数类定义 ==================
class EcoParameters:
    def __init__(self):
        # 空间参数
        self.grid_size = 20
        self.days = 730

        # 物种参数
        self.syrphid_disperse = 0.15
        self.owl_edge_maturity_rate = 0.002
        self.syrphid_habitat_threshold = 0.6
        self.pesticide_decay = 0.95

        # 农业参数
        self.crop_max = 12000.0
        self.weed_growth = 0.025
        np.random.seed(42)


# ================== 农业网格类定义 ==================
class AgriculturalGrid:
    def __init__(self, params):
        self.size = params.grid_size
        # 状态变量: [作物, 杂草, 害虫, 食蚜蝇, 土壤健康, 蝙蝠]
        self.grid = np.zeros((self.size, self.size, 6))

        # 初始化空间异质性
        self.grid[:, :, 0] = np.random.lognormal(7, 0.3, (self.size, self.size))  # 作物
        self.grid[:, :, 1] = np.random.poisson(50, (self.size, self.size))  # 杂草
        self.grid[:, :, 2] = np.random.exponential(5, (self.size, self.size))  # 害虫
        self.grid[:, :, 3] = np.random.uniform(10, 30, (self.size, self.size))  # 食蚜蝇
        self.grid[:, :, 4] = 1.8 + np.random.rand(self.size, self.size) * 0.5  # 土壤
        self.grid[:, :, 5] = np.random.poisson(30, (self.size, self.size))  # 蝙蝠

    def apply_harvest(self):
        """ 收割事件应用 """
        self.grid[:, :, 0] = 0  # 清除作物
        self.grid[:, :, 2] *= 0.3  # 减少害虫


# ================== 栖息地系统 ==================
class HabitatSystem:
    def __init__(self, params):
        self.params = params
        self.edge_maturity = np.zeros((params.grid_size, params.grid_size))
        self.pesticide_residue = np.zeros((params.grid_size, params.grid_size))

        # 初始化边缘带
        edge_mask = (np.random.rand(params.grid_size, params.grid_size) > 0.7)
        self.edge_maturity[edge_mask] = np.random.uniform(0, 0.5, edge_mask.sum())

    def update_edges(self):
        self.edge_maturity += self.params.owl_edge_maturity_rate
        self.edge_maturity = np.clip(self.edge_maturity, 0, 1)

    def update_pesticide(self, day):
        self.pesticide_residue *= self.params.pesticide_decay
        self.pesticide_residue += pesticide_app(day)

    def get_syrphid_habitat(self):
        suitability = (1 - self.pesticide_residue) * (self.edge_maturity ** 0.5)
        return np.clip(suitability, 0, 1)


# ================== 物种模型 ==================
class EnhancedSyrphidModel:
    def __init__(self, params):
        self.params = params
        self.dispersion_kernel = np.array([[0.05, 0.2, 0.05],
                                           [0.2, 0.1, 0.2],
                                           [0.05, 0.2, 0.05]])

    def diffuse(self, grid, habitat):
        new_pop = ndimage.convolve(grid[:, :, 3], self.dispersion_kernel, mode='reflect')
        suitability_factor = habitat * (1 + 0.5 * np.sin(np.pi * grid[:, :, 3] / 50))
        return grid[:, :, 3] * 0.5 + new_pop * self.params.syrphid_disperse * suitability_factor


class DynamicOwlModel:
    def __init__(self, params):
        self.params = params
        self.habitat_patches = []

    def detect_habitat(self, edge_maturity):
        threshold = np.percentile(edge_maturity, 75)
        labeled, num = ndimage.label(edge_maturity > threshold)

        self.habitat_patches = []
        for i in range(1, num + 1):
            cells = np.argwhere(labeled == i)
            quality = np.mean(edge_maturity[cells[:, 0], cells[:, 1]])
            if len(cells) >= 4:
                self.habitat_patches.append({
                    'cells': cells,
                    'quality': quality,
                    'occupancy': min(quality * 0.8, 1)
                })

    def update_occupancy(self, pest_density):
        for patch in self.habitat_patches:
            colonization = patch['quality'] * (pest_density[patch['cells'][:, 0],
            patch['cells'][:, 1]].mean() / 100)
            extinction = 0.2 * (1 - patch['quality'])
            patch['occupancy'] = (1 - extinction) * patch['occupancy'] + colonization * (1 - patch['occupancy'])
            patch['occupancy'] = np.clip(patch['occupancy'], 0, 1)


# ================== 农药施用函数 ==================
def pesticide_app(day):
    if 100 <= day <= 140: return 0.6
    if 165 <= day <= 205: return 0.9
    if 230 <= day <= 270: return 0.4
    return 0.0


def herbicide_app(day):
    if 50 <= day <= 90: return 1.2
    if 130 <= day <= 170: return 0.8
    return 0.0


# ================== 主模拟循环 ==================
def enhanced_simulation():
    params = EcoParameters()
    habitat_sys = HabitatSystem(params)
    syrphid_model = EnhancedSyrphidModel(params)
    owl_model = DynamicOwlModel(params)

    # 初始化农业网格
    ag_grid = AgriculturalGrid(params)
    grid = ag_grid.grid

    plt.figure(figsize=(18, 12))

    for day in range(params.days):
        # 环境更新
        habitat_sys.update_edges()
        habitat_sys.update_pesticide(day)

        # 物种交互
        habitat = habitat_sys.get_syrphid_habitat()
        grid[:, :, 3] = syrphid_model.diffuse(grid, habitat)
        owl_model.detect_habitat(habitat_sys.edge_maturity)
        owl_model.update_occupancy(grid[:, :, 2])

        # 收割事件
        if day == 270:
            ag_grid.apply_harvest()

        # 可视化
        if day % 30 == 0:
            plot_enhanced(grid, habitat_sys, owl_model, day)

    return grid


# ================== 可视化函数 ==================
def plot_enhanced(grid, habitat_sys, owl_model, day):
    plt.clf()

    # 食蚜蝇栖息地
    plt.subplot(231)
    plt.imshow(habitat_sys.get_syrphid_habitat(), cmap='YlGn', vmin=0, vmax=1)
    plt.title(f'Day {day} Syrphid Habitat')

    # 林鸮占据率
    owl_map = np.zeros_like(grid[:, :, 0])
    for patch in owl_model.habitat_patches:
        for (i, j) in patch['cells']:
            owl_map[i, j] = patch['occupancy']
    plt.subplot(232)
    plt.imshow(owl_map, cmap='Purples', vmin=0, vmax=1)
    plt.title('Owl Occupancy')

    # 边缘带成熟度
    plt.subplot(233)
    plt.imshow(habitat_sys.edge_maturity, cmap='YlOrBr', vmin=0, vmax=1)
    plt.title('Edge Maturity')

    # 原始变量
    plt.subplot(234)
    plt.imshow(grid[:, :, 0], cmap='YlGn', vmin=0, vmax=12000)
    plt.title('Crop Biomass')

    plt.subplot(235)
    plt.imshow(grid[:, :, 2], cmap='Reds', vmin=0, vmax=100)
    plt.title('Pest Density')

    plt.subplot(236)
    plt.imshow(grid[:, :, 3], cmap='Greens', vmin=0, vmax=50)
    plt.title('Syrphid Density')

    plt.tight_layout()
    plt.pause(0.1)


# ================== 运行程序 ==================
if __name__ == '__main__':
    print("启动生态系统模拟...")
    final_grid = enhanced_simulation()
    plt.show()