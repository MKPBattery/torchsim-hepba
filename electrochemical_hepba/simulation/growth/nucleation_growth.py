import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

class NucleationGrowth:
    """成核生长模拟类"""
    
    def __init__(self, temperature=303.15):
        """
        初始化成核生长模拟
        
        Args:
            temperature: 温度（K）
        """
        self.temperature = temperature
        self.R = 8.314  # 气体常数
        
        # 成核参数
        self.surface_energy = 0.1  # 表面能 (J/m²)
        self.molar_volume = 1e-4   # 摩尔体积 (m³/mol)
        self.diffusion_coeff = 1e-9  # 扩散系数 (m²/s)
        
        # 晶格参数
        self.lattice_constant = 1e-9  # 晶格常数 (m)
        
    def calculate_critical_radius(self, oversaturation):
        """
        计算临界核半径
        
        Args:
            oversaturation: 过饱和度
            
        Returns:
            临界核半径 (m)
        """
        return 2 * self.surface_energy * self.molar_volume / (self.R * self.temperature * np.log(oversaturation))
    
    def nucleation_rate(self, oversaturation, surface_sites=1e15):
        """
        计算成核速率
        
        Args:
            oversaturation: 过饱和度
            surface_sites: 表面活性位点数量
            
        Returns:
            成核速率 (个/s)
        """
        # 计算成核功
        delta_G = 16 * np.pi * self.surface_energy**3 * self.molar_volume**2 / \
                 (3 * (self.R * self.temperature * np.log(oversaturation))**2)
        
        # 计算成核频率因子
        kT = self.R * self.temperature / 6.022e23  # 每个原子的热能
        frequency_factor = kT / 6.626e-34  # 玻尔兹曼常数/普朗克常数
        
        # 计算成核速率
        rate = surface_sites * frequency_factor * np.exp(-delta_G / (kT))
        return rate
    
    def growth_rate(self, oversaturation, radius):
        """
        计算晶体生长速率
        
        Args:
            oversaturation: 过饱和度
            radius: 晶体半径 (m)
            
        Returns:
            生长速率 (m/s)
        """
        # 扩散控制的生长速率
        concentration_gradient = (oversaturation - 1) * 0.2  # mol/m³
        rate = self.diffusion_coeff * concentration_gradient / radius
        return rate
    
    def simulate_growth(self, time, oversaturation, initial_radius=1e-9):
        """
        模拟晶体生长过程
        
        Args:
            time: 时间数组 (s)
            oversaturation: 过饱和度
            initial_radius: 初始半径 (m)
            
        Returns:
            半径随时间的变化
        """
        def dr_dt(r, t, oversaturation):
            return self.growth_rate(oversaturation, r)
        
        radius = odeint(dr_dt, initial_radius, time, args=(oversaturation,))
        return radius.flatten()
    
    def simulate_film_formation(self, time_max=3600, n_steps=1000):
        """
        模拟薄膜形成过程
        
        Args:
            time_max: 最大时间 (s)
            n_steps: 时间步数
            
        Returns:
            时间和覆盖度数据
        """
        time = np.linspace(0, time_max, n_steps)
        oversaturation = 2.0  # 假设的过饱和度
        
        # 计算成核速率
        J = self.nucleation_rate(oversaturation)
        
        # 计算生长速率
        r_crit = self.calculate_critical_radius(oversaturation)
        v = self.growth_rate(oversaturation, r_crit)
        
        # 计算覆盖度（Avrami方程）
        theta = 1 - np.exp(-np.pi * J * v**2 * time**3 / 3)
        
        return time, theta
    
    def plot_results(self, time, coverage):
        """绘制薄膜生长曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(time/60, coverage*100, 'b-')
        plt.xlabel('时间 (min)')
        plt.ylabel('覆盖度 (%)')
        plt.title('薄膜生长曲线')
        plt.grid(True)
        plt.savefig('film_growth.png')
        plt.close()
        
        # 绘制生长速率
        plt.figure(figsize=(10, 6))
        growth_rate = np.gradient(coverage, time)
        plt.plot(time/60, growth_rate*100, 'r-')
        plt.xlabel('时间 (min)')
        plt.ylabel('生长速率 (%/s)')
        plt.title('薄膜生长速率')
        plt.grid(True)
        plt.savefig('growth_rate.png')
        plt.close()

if __name__ == "__main__":
    # 测试代码
    growth = NucleationGrowth()
    
    # 模拟薄膜形成
    t, coverage = growth.simulate_film_formation()
    growth.plot_results(t, coverage)
    
    # 输出关键参数
    oversaturation = 2.0
    r_crit = growth.calculate_critical_radius(oversaturation)
    J = growth.nucleation_rate(oversaturation)
    
    print(f"临界核半径: {r_crit*1e9:.2f} nm")
    print(f"成核速率: {J:.2e} 个/s")
    print(f"最终覆盖度: {coverage[-1]*100:.1f}%") 