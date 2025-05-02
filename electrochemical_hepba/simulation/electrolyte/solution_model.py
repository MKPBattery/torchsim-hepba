import numpy as np
from scipy.constants import N_A
import pandas as pd

class ElectrolyteSolution:
    """电解液模拟类"""
    
    def __init__(self, volume=1.0, temperature=303.15):
        """
        初始化电解液系统
        
        Args:
            volume: 体积（L）
            temperature: 温度（K）
        """
        self.volume = volume  # L
        self.temperature = temperature  # K
        self.species = {}
        self.complexes = {}
        self.pH = 3.5
        
        # 初始化物种
        self._initialize_species()
        
    def _initialize_species(self):
        """初始化电解液中的物种"""
        # 金属离子
        self.species['Cu2+'] = {'concentration': 0.2, 'charge': 2, 'radius': 0.073}  # nm
        self.species['Fe3+'] = {'concentration': 0.2, 'charge': 3, 'radius': 0.065}
        self.species['Mn2+'] = {'concentration': 0.2, 'charge': 2, 'radius': 0.083}
        self.species['Ni2+'] = {'concentration': 0.2, 'charge': 2, 'radius': 0.069}
        self.species['Co2+'] = {'concentration': 0.2, 'charge': 2, 'radius': 0.075}
        
        # 支持电解质
        self.species['NH4+'] = {'concentration': 0.05, 'charge': 1, 'radius': 0.143}
        self.species['SO42-'] = {'concentration': 0.05, 'charge': -2, 'radius': 0.230}
        
        # 缓冲体系
        self.species['H3BO3'] = {'concentration': 0.162, 'charge': 0, 'radius': 0.244}
        
    def calculate_ionic_strength(self):
        """计算离子强度"""
        I = 0
        for name, info in self.species.items():
            if info['charge'] != 0:
                I += 0.5 * info['concentration'] * (info['charge']**2)
        return I
    
    def calculate_debye_length(self):
        """计算德拜长度（nm）"""
        eps_r = 78.5  # 水的相对介电常数
        eps_0 = 8.854e-12  # 真空介电常数
        kB = 1.380649e-23  # 玻尔兹曼常数
        e = 1.602176634e-19  # 基本电荷
        
        I = self.calculate_ionic_strength() * 1000  # 转换为mol/m³
        
        # 德拜长度计算（nm）
        kappa = np.sqrt((2 * N_A * e**2 * I) / (eps_r * eps_0 * kB * self.temperature))
        return 1 / (kappa * 1e-9)
    
    def calculate_activity_coefficients(self):
        """使用扩展德拜-休克尔方程计算活度系数"""
        I = self.calculate_ionic_strength()
        activity_coeffs = {}
        
        for name, info in self.species.items():
            if info['charge'] != 0:
                # 扩展德拜-休克尔方程
                A = 0.509  # 在25°C水溶液中的常数
                B = 0.328  # nm⁻¹
                z = info['charge']
                r = info['radius']  # nm
                
                log_gamma = -A * z**2 * np.sqrt(I) / (1 + B * r * np.sqrt(I))
                activity_coeffs[name] = 10**log_gamma
            else:
                activity_coeffs[name] = 1.0
                
        return activity_coeffs
    
    def calculate_citrate_complexation(self):
        """计算柠檬酸络合平衡"""
        # 柠檬酸的分步解离常数（pKa值）
        pKa = [3.13, 4.76, 6.40]
        
        # 在pH=3.5时计算各种形态的分布
        alpha = np.zeros(4)  # [H3Cit, H2Cit-, HCit2-, Cit3-]
        
        # 计算分布系数
        h = 10**(-self.pH)
        Ka = [10**(-pk) for pk in pKa]
        
        denominator = (h**3 + Ka[0]*h**2 + Ka[0]*Ka[1]*h + Ka[0]*Ka[1]*Ka[2])
        
        alpha[0] = h**3 / denominator
        alpha[1] = Ka[0]*h**2 / denominator
        alpha[2] = Ka[0]*Ka[1]*h / denominator
        alpha[3] = Ka[0]*Ka[1]*Ka[2] / denominator
        
        return {
            'H3Cit': alpha[0],
            'H2Cit-': alpha[1],
            'HCit2-': alpha[2],
            'Cit3-': alpha[3]
        }
    
    def get_solution_properties(self):
        """获取溶液性质汇总"""
        properties = {
            'ionic_strength': self.calculate_ionic_strength(),
            'debye_length': self.calculate_debye_length(),
            'activity_coefficients': self.calculate_activity_coefficients(),
            'citrate_distribution': self.calculate_citrate_complexation()
        }
        return properties

if __name__ == "__main__":
    # 测试代码
    solution = ElectrolyteSolution()
    properties = solution.get_solution_properties()
    
    print("电解液性质：")
    print(f"离子强度: {properties['ionic_strength']:.3f} M")
    print(f"德拜长度: {properties['debye_length']:.3f} nm")
    print("\n活度系数：")
    for ion, coeff in properties['activity_coefficients'].items():
        print(f"{ion}: {coeff:.3f}")
    print("\n柠檬酸络合物分布：")
    for species, fraction in properties['citrate_distribution'].items():
        print(f"{species}: {fraction:.3f}") 