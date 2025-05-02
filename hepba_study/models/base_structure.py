import numpy as np
from ase import Atoms
from ase.io import write
from ase.optimize import LBFGS
from ase.calculators.lj import LennardJones  # 使用LJ势能
import matplotlib.pyplot as plt
import os

class BaseHEPBA:
    def __init__(self, metal='Cu', lattice_constant=10.2):
        """
        初始化基础HEPBA结构
        
        参数:
            metal: 过渡金属元素 (Cu, Fe, Mn, Ni, Co)
            lattice_constant: 晶格常数 (Å)
        """
        self.metal = metal
        self.lattice_constant = lattice_constant
        self.structure = None
        
        # 定义LJ参数
        self.lj_params = {
            'Cu': {'epsilon': 0.1, 'sigma': 2.5},
            'Fe': {'epsilon': 0.1, 'sigma': 2.5},
            'Mn': {'epsilon': 0.1, 'sigma': 2.5},
            'Ni': {'epsilon': 0.1, 'sigma': 2.5},
            'Co': {'epsilon': 0.1, 'sigma': 2.5}
        }
        
    def create_base_structure(self):
        """创建基础HEPBA结构"""
        # 定义晶胞参数
        a = self.lattice_constant
        cell = np.array([[a, 0, 0],
                        [0, a, 0],
                        [0, 0, a]])
        
        # 定义原子位置（简化模型）
        positions = np.array([
            [0, 0, 0],           # 金属原子位置
            [0.5, 0.5, 0.5],     # 氰根配体位置
            [0.25, 0.25, 0.25],  # 氮原子位置
            [0.75, 0.75, 0.75]   # 碳原子位置
        ])
        
        # 创建原子对象
        symbols = [self.metal, 'C', 'N', 'C']
        self.structure = Atoms(symbols=symbols,
                             positions=positions * a,  # 缩放到实际晶格尺寸
                             cell=cell,
                             pbc=True)
        
        return self.structure
    
    def optimize_structure(self, fmax=0.05):
        """优化结构"""
        if self.structure is None:
            self.create_base_structure()
            
        # 设置LJ计算器
        params = self.lj_params[self.metal]
        calculator = LennardJones(epsilon=params['epsilon'], 
                                sigma=params['sigma'])
        self.structure.set_calculator(calculator)
        
        # 进行结构优化
        optimizer = LBFGS(self.structure)
        optimizer.run(fmax=fmax)
        
        return self.structure
    
    def analyze_structure(self):
        """分析结构性质"""
        if self.structure is None:
            raise ValueError("结构未创建或优化")
            
        # 计算键长
        positions = self.structure.get_positions()
        metal_n_bond = np.linalg.norm(positions[0] - positions[2])
        cn_bond = np.linalg.norm(positions[1] - positions[3])
        
        # 计算晶胞体积
        volume = self.structure.get_volume()
        
        # 计算密度
        mass = sum(self.structure.get_masses())
        density = mass / volume * 1.6605  # 转换为g/cm³
        
        # 计算配位数
        cutoff = 3.0  # 键长阈值
        metal_n_count = 0
        for i in range(1, len(positions)):
            if np.linalg.norm(positions[0] - positions[i]) < cutoff:
                metal_n_count += 1
        
        return {
            'metal_n_bond': metal_n_bond,
            'cn_bond': cn_bond,
            'volume': volume,
            'density': density,
            'energy': self.structure.get_potential_energy(),
            'coordination': metal_n_count
        }
    
    def visualize_structure(self, filename='structure.png'):
        """可视化结构"""
        if self.structure is None:
            raise ValueError("结构未创建或优化")
            
        write(filename, self.structure)
        
    def plot_optimization_results(self, results, filename='optimization_results.png'):
        """绘制优化结果"""
        plt.figure(figsize=(15, 10))
        
        # 键长比较
        plt.subplot(2, 3, 1)
        metals = list(results.keys())
        metal_n_bonds = [results[m]['metal_n_bond'] for m in metals]
        plt.bar(metals, metal_n_bonds)
        plt.xlabel('金属元素')
        plt.ylabel('金属-氮键长 (Å)')
        plt.title('不同金属HEPBA的键长比较')
        
        # 晶胞体积比较
        plt.subplot(2, 3, 2)
        volumes = [results[m]['volume'] for m in metals]
        plt.bar(metals, volumes)
        plt.xlabel('金属元素')
        plt.ylabel('晶胞体积 (Å³)')
        plt.title('不同金属HEPBA的晶胞体积比较')
        
        # 密度比较
        plt.subplot(2, 3, 3)
        densities = [results[m]['density'] for m in metals]
        plt.bar(metals, densities)
        plt.xlabel('金属元素')
        plt.ylabel('密度 (g/cm³)')
        plt.title('不同金属HEPBA的密度比较')
        
        # 能量比较
        plt.subplot(2, 3, 4)
        energies = [results[m]['energy'] for m in metals]
        plt.bar(metals, energies)
        plt.xlabel('金属元素')
        plt.ylabel('能量 (eV)')
        plt.title('不同金属HEPBA的能量比较')
        
        # 配位数比较
        plt.subplot(2, 3, 5)
        coordinations = [results[m]['coordination'] for m in metals]
        plt.bar(metals, coordinations)
        plt.xlabel('金属元素')
        plt.ylabel('配位数')
        plt.title('不同金属HEPBA的配位数比较')
        
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

def main():
    # 测试目标金属的HEPBA结构
    metals = ['Cu', 'Fe', 'Mn', 'Ni', 'Co']
    results = {}
    
    for metal in metals:
        print(f"\n分析 {metal}-HEPBA 结构:")
        hepba = BaseHEPBA(metal=metal)
        
        # 创建并优化结构
        structure = hepba.create_base_structure()
        optimized_structure = hepba.optimize_structure()
        
        # 分析结构性质
        analysis = hepba.analyze_structure()
        results[metal] = analysis
        
        print(f"键长分析:")
        print(f"金属-氮键长: {analysis['metal_n_bond']:.3f} Å")
        print(f"氰根键长: {analysis['cn_bond']:.3f} Å")
        print(f"晶胞体积: {analysis['volume']:.3f} Å³")
        print(f"密度: {analysis['density']:.3f} g/cm³")
        print(f"能量: {analysis['energy']:.3f} eV")
        print(f"配位数: {analysis['coordination']}")
        
        # 保存结构
        os.makedirs('data', exist_ok=True)
        os.makedirs('analysis', exist_ok=True)
        hepba.visualize_structure(f'data/{metal}_hepba.cif')
    
    # 绘制比较图
    hepba.plot_optimization_results(results, 'analysis/base_structure_comparison.png')
    
    # 生成报告
    with open('analysis/hepba_analysis_report.txt', 'w') as f:
        f.write("HEPBA结构分析报告\n")
        f.write("==================\n\n")
        
        for metal in metals:
            f.write(f"{metal}-HEPBA 结构分析:\n")
            f.write(f"-------------------\n")
            f.write(f"金属-氮键长: {results[metal]['metal_n_bond']:.3f} Å\n")
            f.write(f"氰根键长: {results[metal]['cn_bond']:.3f} Å\n")
            f.write(f"晶胞体积: {results[metal]['volume']:.3f} Å³\n")
            f.write(f"密度: {results[metal]['density']:.3f} g/cm³\n")
            f.write(f"能量: {results[metal]['energy']:.3f} eV\n")
            f.write(f"配位数: {results[metal]['coordination']}\n\n")
            
            # 添加结构稳定性分析
            f.write("结构稳定性分析:\n")
            if results[metal]['energy'] < 0:
                f.write("结构相对稳定\n")
            else:
                f.write("结构可能需要进一步优化\n")
            
            # 添加配位环境分析
            f.write("配位环境分析:\n")
            if results[metal]['coordination'] >= 6:
                f.write("配位环境完整，结构稳定\n")
            else:
                f.write("配位环境不完整，可能需要调整\n")
            
            f.write("\n")

if __name__ == "__main__":
    main() 