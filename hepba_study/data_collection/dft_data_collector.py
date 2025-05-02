import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.calculators.vasp import Vasp
import os
import json
from pathlib import Path
import h5py

class DFTDataCollector:
    def __init__(self, base_dir='data/dft'):
        """
        初始化DFT数据收集器
        
        参数:
            base_dir: 数据存储的基础目录
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # 定义DFT计算参数
        self.vasp_params = {
            'encut': 500,          # 截断能
            'ismear': 0,           # 高斯展宽
            'sigma': 0.05,         # 展宽参数
            'ispin': 2,            # 自旋极化
            'lreal': 'Auto',       # 实空间投影
            'algo': 'Normal',      # 算法
            'ncore': 4,            # 并行核心数
            'kpts': [2, 2, 2],     # k点网格
            'xc': 'PBE'            # 交换关联泛函
        }
        
    def generate_structures(self, metals=['Cu', 'Fe', 'Mn', 'Ni', 'Co']):
        """
        生成不同金属的HEPBA结构
        
        参数:
            metals: 要研究的金属列表
        """
        structures = {}
        for metal in metals:
            # 创建基础结构
            structure = self._create_base_structure(metal)
            structures[metal] = structure
            
            # 保存结构
            struct_dir = self.base_dir / 'structures' / metal
            struct_dir.mkdir(parents=True, exist_ok=True)
            write(struct_dir / 'initial.cif', structure)
            
        return structures
    
    def _create_base_structure(self, metal, lattice_constant=10.2):
        """
        创建基础HEPBA结构
        """
        a = lattice_constant
        cell = np.array([[a, 0, 0],
                        [0, a, 0],
                        [0, 0, a]])
        
        positions = np.array([
            [0, 0, 0],           # 金属原子
            [0.5, 0.5, 0.5],     # 氰根配体
            [0.25, 0.25, 0.25],  # 氮原子
            [0.75, 0.75, 0.75]   # 碳原子
        ])
        
        symbols = [metal, 'C', 'N', 'C']
        return Atoms(symbols=symbols,
                    positions=positions * a,
                    cell=cell,
                    pbc=True)
    
    def run_dft_calculations(self, structures):
        """
        运行DFT计算
        
        参数:
            structures: 结构字典
        """
        results = {}
        for metal, structure in structures.items():
            print(f"\n开始计算 {metal}-HEPBA 结构:")
            
            # 设置计算器
            calc = Vasp(**self.vasp_params)
            structure.set_calculator(calc)
            
            # 运行计算
            try:
                energy = structure.get_potential_energy()
                forces = structure.get_forces()
                stress = structure.get_stress()
                
                # 保存结果
                results[metal] = {
                    'energy': energy,
                    'forces': forces.tolist(),
                    'stress': stress.tolist()
                }
                
                # 保存结构
                struct_dir = self.base_dir / 'structures' / metal
                write(struct_dir / 'optimized.cif', structure)
                
                print(f"{metal}-HEPBA 计算完成:")
                print(f"能量: {energy:.6f} eV")
                print(f"最大力: {np.max(np.abs(forces)):.6f} eV/Å")
                
            except Exception as e:
                print(f"{metal}-HEPBA 计算失败: {str(e)}")
        
        return results
    
    def save_results(self, results):
        """
        保存计算结果
        
        参数:
            results: 计算结果字典
        """
        # 保存为JSON
        with open(self.base_dir / 'dft_results.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        # 保存为HDF5
        with h5py.File(self.base_dir / 'dft_results.h5', 'w') as f:
            for metal, data in results.items():
                group = f.create_group(metal)
                group.create_dataset('energy', data=data['energy'])
                group.create_dataset('forces', data=data['forces'])
                group.create_dataset('stress', data=data['stress'])
    
    def analyze_results(self):
        """
        分析计算结果
        """
        # 读取结果
        with open(self.base_dir / 'dft_results.json', 'r') as f:
            results = json.load(f)
        
        # 分析每个金属的结果
        analysis = {}
        for metal, data in results.items():
            energy = data['energy']
            forces = np.array(data['forces'])
            stress = np.array(data['stress'])
            
            # 计算统计量
            max_force = np.max(np.abs(forces))
            mean_force = np.mean(np.abs(forces))
            max_stress = np.max(np.abs(stress))
            
            analysis[metal] = {
                'energy': energy,
                'max_force': max_force,
                'mean_force': mean_force,
                'max_stress': max_stress
            }
        
        # 保存分析结果
        with open(self.base_dir / 'analysis.json', 'w') as f:
            json.dump(analysis, f, indent=4)
        
        return analysis

def main():
    # 创建数据收集器
    collector = DFTDataCollector()
    
    # 生成结构
    print("生成HEPBA结构...")
    structures = collector.generate_structures()
    
    # 运行DFT计算
    print("\n运行DFT计算...")
    results = collector.run_dft_calculations(structures)
    
    # 保存结果
    print("\n保存计算结果...")
    collector.save_results(results)
    
    # 分析结果
    print("\n分析计算结果...")
    analysis = collector.analyze_results()
    
    print("\n分析完成！")
    for metal, data in analysis.items():
        print(f"\n{metal}-HEPBA 分析结果:")
        print(f"能量: {data['energy']:.6f} eV")
        print(f"最大力: {data['max_force']:.6f} eV/Å")
        print(f"平均力: {data['mean_force']:.6f} eV/Å")
        print(f"最大应力: {data['max_stress']:.6f} eV/Å³")

if __name__ == "__main__":
    main() 