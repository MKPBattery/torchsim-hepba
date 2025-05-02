import numpy as np
from ase.io import read
from ase.calculators.emt import EMT
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class HEPBAEnergyAnalysis:
    def __init__(self, structure_file):
        """
        初始化HEPBA能量分析类
        
        参数:
            structure_file: CIF文件路径
        """
        self.structure = read(structure_file)
        self.calculator = EMT()  # 使用EMT势能函数进行快速计算
        
    def calculate_total_energy(self):
        """计算总能量"""
        self.structure.set_calculator(self.calculator)
        return self.structure.get_potential_energy()
    
    def calculate_formation_energy(self, reference_energies):
        """
        计算形成能
        
        参数:
            reference_energies: 参考能量字典，包含各元素的参考能量
        """
        total_energy = self.calculate_total_energy()
        formation_energy = total_energy
        
        # 减去参考能量
        for atom in self.structure:
            formation_energy -= reference_energies[atom.symbol]
            
        return formation_energy
    
    def identify_active_sites(self, threshold=0.5):
        """
        识别活性位点
        
        参数:
            threshold: 能量阈值，用于判断活性位点
        """
        active_sites = []
        positions = self.structure.get_positions()
        
        # 计算每个原子的局部能量
        for i, atom in enumerate(self.structure):
            # 计算与周围原子的距离
            distances = np.linalg.norm(positions - positions[i], axis=1)
            neighbors = np.where((distances < 3.0) & (distances > 0.1))[0]
            
            # 计算局部能量
            local_energy = 0
            for j in neighbors:
                local_energy += self.calculator.get_potential_energy(
                    self.structure[[i, j]])
                
            if local_energy > threshold:
                active_sites.append(i)
                
        return active_sites
    
    def calculate_bond_forces(self):
        """计算键力"""
        forces = self.structure.get_forces()
        return np.linalg.norm(forces, axis=1)
    
    def analyze_ion_insertion(self, ion='Li'):
        """
        分析离子嵌入过程
        
        参数:
            ion: 嵌入离子类型
        """
        # 创建离子嵌入结构
        positions = self.structure.get_positions()
        cell = self.structure.get_cell()
        
        # 在晶胞中心添加离子
        center = np.sum(cell, axis=0) / 2
        new_structure = self.structure.copy()
        new_structure.append(ion)
        new_structure.positions[-1] = center
        
        # 计算嵌入能量
        new_structure.set_calculator(self.calculator)
        insertion_energy = new_structure.get_potential_energy() - self.calculate_total_energy()
        
        return insertion_energy

def main():
    # 测试不同金属的HEPBA结构
    metals = ['Cu', 'Fe', 'Mn', 'Ni', 'Co']
    results = {}
    
    # 参考能量（示例值，实际需要从数据库获取）
    reference_energies = {
        'Cu': -3.49,
        'Fe': -5.28,
        'Mn': -4.44,
        'Ni': -4.44,
        'Co': -4.39,
        'C': -1.0,
        'N': -1.0
    }
    
    for metal in metals:
        print(f"\n分析 {metal}-HEPBA 结构:")
        analysis = HEPBAEnergyAnalysis(f'data/{metal}_hepba.cif')
        
        # 计算总能量
        total_energy = analysis.calculate_total_energy()
        print(f"总能量: {total_energy:.3f} eV")
        
        # 计算形成能
        formation_energy = analysis.calculate_formation_energy(reference_energies)
        print(f"形成能: {formation_energy:.3f} eV")
        
        # 识别活性位点
        active_sites = analysis.identify_active_sites()
        print(f"活性位点数量: {len(active_sites)}")
        
        # 计算键力
        bond_forces = analysis.calculate_bond_forces()
        print(f"平均键力: {np.mean(bond_forces):.3f} eV/Å")
        
        # 分析离子嵌入
        insertion_energy = analysis.analyze_ion_insertion()
        print(f"锂离子嵌入能: {insertion_energy:.3f} eV")
        
        results[metal] = {
            'total_energy': total_energy,
            'formation_energy': formation_energy,
            'active_sites': len(active_sites),
            'bond_forces': np.mean(bond_forces),
            'insertion_energy': insertion_energy
        }
    
    # 绘制比较图
    plt.figure(figsize=(15, 10))
    
    # 形成能比较
    plt.subplot(2, 2, 1)
    metals = list(results.keys())
    formation_energies = [results[m]['formation_energy'] for m in metals]
    plt.bar(metals, formation_energies)
    plt.xlabel('金属元素')
    plt.ylabel('形成能 (eV)')
    plt.title('不同金属HEPBA的形成能比较')
    
    # 活性位点比较
    plt.subplot(2, 2, 2)
    active_sites = [results[m]['active_sites'] for m in metals]
    plt.bar(metals, active_sites)
    plt.xlabel('金属元素')
    plt.ylabel('活性位点数量')
    plt.title('不同金属HEPBA的活性位点比较')
    
    # 键力比较
    plt.subplot(2, 2, 3)
    bond_forces = [results[m]['bond_forces'] for m in metals]
    plt.bar(metals, bond_forces)
    plt.xlabel('金属元素')
    plt.ylabel('平均键力 (eV/Å)')
    plt.title('不同金属HEPBA的键力比较')
    
    # 嵌入能比较
    plt.subplot(2, 2, 4)
    insertion_energies = [results[m]['insertion_energy'] for m in metals]
    plt.bar(metals, insertion_energies)
    plt.xlabel('金属元素')
    plt.ylabel('锂离子嵌入能 (eV)')
    plt.title('不同金属HEPBA的锂离子嵌入能比较')
    
    plt.tight_layout()
    plt.savefig('analysis/energy_comparison.png')
    plt.close()

if __name__ == "__main__":
    main() 