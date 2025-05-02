import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.visualize import view
from ase.io import write
import matplotlib.pyplot as plt

class HEPBAStructure:
    def __init__(self, metal='Fe', lattice_constant=10.2):
        """
        初始化HEPBA结构
        
        参数:
            metal: 过渡金属元素 (Cu, Fe, Mn, Ni, Co)
            lattice_constant: 晶格常数 (Å)
        """
        self.metal = metal
        self.lattice_constant = lattice_constant
        
    def create_unit_cell(self):
        """创建HEPBA的晶胞结构"""
        # 定义晶胞参数
        a = self.lattice_constant
        cell = np.array([[a, 0, 0],
                        [0, a, 0],
                        [0, 0, a]])
        
        # 定义原子位置
        positions = np.array([
            [0, 0, 0],           # 金属原子位置
            [0.5, 0.5, 0.5],     # 氰根配体位置
            [0.25, 0.25, 0.25],  # 氮原子位置
            [0.75, 0.75, 0.75]   # 碳原子位置
        ])
        
        # 创建原子对象
        symbols = [self.metal, 'C', 'N', 'C']
        self.structure = Atoms(symbols=symbols,
                             positions=positions,
                             cell=cell,
                             pbc=True)
        
        return self.structure
    
    def calculate_bond_lengths(self):
        """计算键长"""
        # 获取原子位置
        positions = self.structure.get_positions()
        
        # 计算金属-氮键长
        metal_n_bond = np.linalg.norm(positions[0] - positions[2])
        
        # 计算氰根键长
        cn_bond = np.linalg.norm(positions[1] - positions[3])
        
        return {
            'metal_n_bond': metal_n_bond,
            'cn_bond': cn_bond
        }
    
    def visualize_structure(self, filename='structure.png'):
        """可视化结构"""
        view(self.structure)
        write(filename, self.structure)
        
    def analyze_structure(self):
        """分析结构性质"""
        # 计算键长
        bond_lengths = self.calculate_bond_lengths()
        
        # 计算晶胞体积
        volume = self.structure.get_volume()
        
        # 计算密度
        mass = sum(self.structure.get_masses())
        density = mass / volume
        
        return {
            'bond_lengths': bond_lengths,
            'volume': volume,
            'density': density
        }

def main():
    # 测试不同金属的HEPBA结构
    metals = ['Cu', 'Fe', 'Mn', 'Ni', 'Co']
    results = {}
    
    for metal in metals:
        print(f"\n分析 {metal}-HEPBA 结构:")
        hepba = HEPBAStructure(metal=metal)
        structure = hepba.create_unit_cell()
        analysis = hepba.analyze_structure()
        
        results[metal] = analysis
        print(f"键长分析:")
        print(f"金属-氮键长: {analysis['bond_lengths']['metal_n_bond']:.3f} Å")
        print(f"氰根键长: {analysis['bond_lengths']['cn_bond']:.3f} Å")
        print(f"晶胞体积: {analysis['volume']:.3f} Å³")
        print(f"密度: {analysis['density']:.3f} g/cm³")
        
        # 保存结构
        hepba.visualize_structure(f'data/{metal}_hepba.cif')
    
    # 绘制比较图
    plt.figure(figsize=(10, 6))
    metals = list(results.keys())
    bond_lengths = [results[m]['bond_lengths']['metal_n_bond'] for m in metals]
    
    plt.bar(metals, bond_lengths)
    plt.xlabel('金属元素')
    plt.ylabel('金属-氮键长 (Å)')
    plt.title('不同金属HEPBA的键长比较')
    plt.savefig('analysis/bond_length_comparison.png')
    plt.close()

if __name__ == "__main__":
    main() 