import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

class ElectrodeKinetics:
    """电极界面动力学模拟类"""
    
    def __init__(self, area=1.0, temperature=303.15):
        """
        初始化电极界面系统
        
        Args:
            area: 电极面积（cm²）
            temperature: 温度（K）
        """
        self.area = area
        self.temperature = temperature
        self.F = 96485  # 法拉第常数
        self.R = 8.314  # 气体常数
        
        # 电化学参数
        self.E0 = {  # 标准电极电位 (V vs. Ag/AgCl)
            'Cu2+': 0.115,
            'Fe3+': 0.530,
            'Mn2+': -1.185,
            'Ni2+': -0.257,
            'Co2+': -0.277
        }
        
        self.k0 = {  # 标准电子转移速率常数 (cm/s)
            'Cu2+': 1e-2,
            'Fe3+': 5e-3,
            'Mn2+': 1e-3,
            'Ni2+': 2e-3,
            'Co2+': 2e-3
        }
        
        self.alpha = 0.5  # 传递系数
        
    def butler_volmer_current(self, E, C, metal_ion):
        """
        计算Butler-Volmer电流
        
        Args:
            E: 电极电位 (V)
            C: 金属离子浓度 (mol/cm³)
            metal_ion: 金属离子类型
            
        Returns:
            电流密度 (A/cm²)
        """
        eta = E - self.E0[metal_ion]  # 过电位
        f = self.F / (self.R * self.temperature)
        
        # 阳极和阴极电流
        ia = self.k0[metal_ion] * C * np.exp(self.alpha * f * eta)
        ic = -self.k0[metal_ion] * C * np.exp(-(1-self.alpha) * f * eta)
        
        return self.F * (ia + ic)
    
    def simulate_cv(self, scan_rate=0.02, E_start=-0.4, E_end=1.3, cycles=1):
        """
        模拟循环伏安过程
        
        Args:
            scan_rate: 扫描速率 (V/s)
            E_start: 起始电位 (V)
            E_end: 终止电位 (V)
            cycles: 循环次数
            
        Returns:
            电位和电流数据
        """
        # 时间点
        t_cycle = 2 * (E_end - E_start) / scan_rate
        t = np.linspace(0, t_cycle * cycles, cycles * 1000)
        
        # 电位计算
        E = np.zeros_like(t)
        for i in range(len(t)):
            cycle_time = t[i] % t_cycle
            if cycle_time < t_cycle/2:
                E[i] = E_start + scan_rate * cycle_time
            else:
                E[i] = E_end - scan_rate * (cycle_time - t_cycle/2)
        
        # 计算总电流
        I_total = np.zeros_like(t)
        C = 0.2e-3  # 浓度转换为mol/cm³
        
        for metal_ion in self.E0.keys():
            I = self.butler_volmer_current(E, C, metal_ion)
            I_total += I
        
        return E, I_total
    
    def simulate_chronoamperometry(self, current_density=0.02, time=3600):
        """
        模拟恒电流沉积过程
        
        Args:
            current_density: 电流密度 (A/cm²)
            time: 沉积时间 (s)
            
        Returns:
            时间和电位数据
        """
        t = np.linspace(0, time, 1000)
        E = np.zeros_like(t)
        C = 0.2e-3  # 初始浓度 (mol/cm³)
        
        def find_potential(E_guess, I_target, C):
            """求解目标电流对应的电位"""
            I_total = sum(self.butler_volmer_current(E_guess, C, ion) 
                         for ion in self.E0.keys())
            return I_total - I_target
        
        # 计算每个时间点的电位
        for i in range(len(t)):
            # 使用简单的二分法求解
            E_low, E_high = -2.0, 2.0
            while (E_high - E_low) > 1e-3:
                E_mid = (E_low + E_high) / 2
                if find_potential(E_mid, current_density, C) > 0:
                    E_high = E_mid
                else:
                    E_low = E_mid
            E[i] = (E_low + E_high) / 2
            
            # 更新浓度（简化模型）
            C = 0.2e-3 * (1 - t[i]/time)
        
        return t, E
    
    def plot_cv_results(self, E, I):
        """绘制CV曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(E, I*1000, 'b-')
        plt.xlabel('电位 (V vs. Ag/AgCl)')
        plt.ylabel('电流密度 (mA/cm²)')
        plt.title('循环伏安曲线')
        plt.grid(True)
        plt.savefig('cv_curve.png')
        plt.close()
    
    def plot_ca_results(self, t, E):
        """绘制计时电位曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(t/60, E, 'r-')
        plt.xlabel('时间 (min)')
        plt.ylabel('电位 (V vs. Ag/AgCl)')
        plt.title('计时电位曲线')
        plt.grid(True)
        plt.savefig('chronopotentiometry.png')
        plt.close()

if __name__ == "__main__":
    # 测试代码
    kinetics = ElectrodeKinetics()
    
    # 模拟CV
    E_cv, I_cv = kinetics.simulate_cv(scan_rate=0.02, cycles=2)
    kinetics.plot_cv_results(E_cv, I_cv)
    
    # 模拟恒电流
    t_ca, E_ca = kinetics.simulate_chronoamperometry(current_density=0.02)
    kinetics.plot_ca_results(t_ca, E_ca) 