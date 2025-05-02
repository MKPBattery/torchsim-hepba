"""构建高熵普鲁士蓝类似物（HEPBA）结构模型"""

import torch
from ase.build import bulk
from ase.atoms import Atoms
from ase.spacegroup import Spacegroup
from mace.calculators.foundations_models import mace_mp
import torch_sim as ts
from torch_sim.unbatched.models.mace import UnbatchedMaceModel
from torch_sim.optimizers import fire
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from ase.io import write
import json

class HighEntropyHEPBA:
    def __init__(self, 
                 metals_M: List[str] = ['Fe', 'Co', 'Ni', 'Mn', 'Cu'],
                 metal_ratios_M: Optional[List[float]] = None,
                 include_water: bool = True,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 dtype: torch.dtype = torch.float32):
        """初始化高熵HEPBA结构生成器
        
        参数:
            metals_M: M位置的金属元素列表（八面体间隙中心）
            metal_ratios_M: M位置金属元素的比例（如果为None则均匀分布）
            include_water: 是否包含结构水
            device: 计算设备
            dtype: 数据类型
        """
        self.metals_M = metals_M
        
        # 如果未指定比例，则使用均匀分布
        self.metal_ratios_M = metal_ratios_M if metal_ratios_M is not None \
            else [1.0/len(metals_M)] * len(metals_M)
        
        self.include_water = include_water
        self.device = device
        self.dtype = dtype
        
        # 初始化MACE模型
        self._initialize_mace_model()
        
        # 定义晶格参数（典型值）
        self.lattice_constant = 10.2  # Å
        
        # 定义键长参考值
        self.bond_lengths = {
            'M-N': 2.1,    # M-N键长
            'M-O': 2.0,    # M-O键长（结构水）
            'Fe-C': 1.9,   # Fe-C键长（M'位点固定为Fe）
            'C-N': 1.15    # C-N三键
        }

    def _initialize_mace_model(self):
        """初始化MACE模型"""
        mace_checkpoint_url = "https://github.com/ACEsuit/mace-foundations/releases/download/mace_mpa_0/mace-mpa-0-medium.model"
        self.loaded_model = mace_mp(
            model=mace_checkpoint_url,
            return_raw_model=True,
            default_dtype=self.dtype,
            device=self.device,
        )
        
        self.model = UnbatchedMaceModel(
            model=self.loaded_model,
            device=self.device,
            compute_forces=True,
            compute_stress=True,
            dtype=self.dtype,
            enable_cueq=False,
        )

    def create_structure(self, supercell_size: Tuple[int, int, int] = (2, 2, 2)) -> Atoms:
        """创建高熵HEPBA结构
        
        参数:
            supercell_size: 超胞大小
            
        返回:
            ASE Atoms对象
        """
        # 创建基础立方晶格
        a = self.lattice_constant
        cell = np.array([[a, 0, 0],
                        [0, a, 0],
                        [0, 0, a]])
        
        # 定义原子位置
        positions = []
        symbols = []
        
        # 添加M位置的金属离子（八面体间隙中心）
        m_sites = [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5]
        ]
        
        for site in m_sites:
            metal = np.random.choice(self.metals_M, p=self.metal_ratios_M)
            positions.append(site)
            symbols.append(metal)
        
        # 添加[Fe(CN)6]八面体（M'位点固定为Fe）
        mprime_sites = [
            [0.25, 0.25, 0.25],
            [0.75, 0.75, 0.25],
            [0.75, 0.25, 0.75],
            [0.25, 0.75, 0.75]
        ]
        
        for site in mprime_sites:
            # M'位点固定为Fe
            positions.append(site)
            symbols.append('Fe')
            
            # 添加周围的CN配体
            cn_offsets = [
                [0.1, 0, 0], [-0.1, 0, 0],
                [0, 0.1, 0], [0, -0.1, 0],
                [0, 0, 0.1], [0, 0, -0.1]
            ]
            
            for offset in cn_offsets:
                # 添加C原子
                c_pos = np.array(site) + np.array(offset)
                positions.append(c_pos)
                symbols.append('C')
                
                # 添加N原子
                n_pos = c_pos + np.array(offset) * (self.bond_lengths['C-N'] / self.lattice_constant)
                positions.append(n_pos)
                symbols.append('N')
        
        # 添加结构水（如果需要）
        if self.include_water:
            water_sites = [
                [0.25, 0.25, 0.0],
                [0.25, 0.0, 0.25],
                [0.0, 0.25, 0.25],
                [0.75, 0.75, 0.0],
                [0.75, 0.0, 0.75],
                [0.0, 0.75, 0.75]
            ]
            
            for site in water_sites:
                # 添加O原子
                positions.append(site)
                symbols.append('O')
                
                # 添加两个H原子
                h1_pos = np.array(site) + np.array([0.02, 0.02, 0])
                h2_pos = np.array(site) + np.array([-0.02, 0.02, 0])
                positions.extend([h1_pos, h2_pos])
                symbols.extend(['H', 'H'])
        
        # 创建初始结构
        structure = Atoms(
            symbols=symbols,
            positions=np.array(positions) * a,
            cell=cell,
            pbc=True
        )
        
        # 创建超胞
        structure = structure.repeat(supercell_size)
        
        return structure

    def optimize_structure(self, structure: Atoms, max_steps: int = 100) -> Tuple[List[Dict], Atoms]:
        """优化HEPBA结构
        
        参数:
            structure: 初始结构
            max_steps: 最大优化步数
            
        返回:
            优化轨迹和优化后的结构
        """
        # 转换为TorchSim状态
        state = ts.io.atoms_to_state(structure, device=self.device, dtype=self.dtype)
        
        # 使用FIRE优化器
        init_fn, update_fn = fire(
            model=self.model,
            dt_max=0.4,
            dt_start=0.01,
            n_min=5,
            f_inc=1.1,
            f_dec=0.5,
            alpha_start=0.1,
            f_alpha=0.99
        )
        
        # 初始化优化器状态
        state = init_fn(state)
        
        # 运行优化
        trajectory = []
        for step in range(max_steps):
            state = update_fn(state)
            
            trajectory.append({
                'step': step,
                'energy': float(state.energy),
                'max_force': float(torch.max(torch.abs(state.forces))),
                'volume': float(torch.det(state.cell))
            })
            
            if step % 10 == 0:
                print(f"Step {step}: Energy = {float(state.energy):.4f} eV, "
                      f"Max Force = {float(torch.max(torch.abs(state.forces))):.4f} eV/Å")
                
            # 检查收敛
            if float(torch.max(torch.abs(state.forces))) < 0.05:  # 力收敛标准 (eV/Å)
                print("结构已收敛!")
                break
        
        # 转换回ASE结构
        optimized_structure = structure.copy()
        optimized_structure.set_positions(state.positions.detach().cpu().numpy())
        
        return trajectory, optimized_structure

    def analyze_structure(self, structure: Atoms) -> Dict:
        """分析HEPBA结构性质
        
        参数:
            structure: ASE Atoms对象
            
        返回:
            结构性质字典
        """
        # 转换为TorchSim状态
        state = ts.io.atoms_to_state(structure, device=self.device, dtype=self.dtype)
        
        # 运行MACE模型
        results = self.model(state)
        
        # 计算平均键长
        positions = structure.get_positions()
        symbols = structure.get_chemical_symbols()
        bond_lengths = {
            'M-N': [],
            'M-C': [],
            'C-N': []
        }
        
        for i, (pos1, sym1) in enumerate(zip(positions, symbols)):
            for j, (pos2, sym2) in enumerate(zip(positions[i+1:], symbols[i+1:])):
                dist = np.linalg.norm(pos1 - pos2)
                if dist < 3.0:  # 只考虑3Å内的原子对
                    if (sym1 in self.metals_M and sym2 == 'N') or \
                       (sym2 in self.metals_M and sym1 == 'N'):
                        bond_lengths['M-N'].append(dist)
                    elif (sym1 == 'C' and sym2 == 'N') or \
                         (sym2 == 'C' and sym1 == 'N'):
                        bond_lengths['C-N'].append(dist)
        
        # 计算平均键长
        avg_bond_lengths = {
            k: np.mean(v) if v else 0.0 for k, v in bond_lengths.items()
        }
        
        return {
            'energy': float(results['energy']),
            'forces': results['forces'].cpu().numpy(),
            'stress': results['stress'].cpu().numpy(),
            'volume': structure.get_volume(),
            'density': structure.get_masses().sum() / structure.get_volume(),
            'avg_bond_lengths': avg_bond_lengths
        }

    def plot_optimization(self, trajectory: List[Dict], save_path: Optional[str] = None):
        """绘制优化过程
        
        参数:
            trajectory: 优化轨迹
            save_path: 保存路径
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # 绘制能量
        steps = [t['step'] for t in trajectory]
        energies = [t['energy'] for t in trajectory]
        ax1.plot(steps, energies, 'b-', label='Energy')
        ax1.set_ylabel('Energy (eV)')
        ax1.legend()
        
        # 绘制最大力
        max_forces = [t['max_force'] for t in trajectory]
        ax2.plot(steps, max_forces, 'r-', label='Max Force')
        ax2.set_xlabel('Optimization Step')
        ax2.set_ylabel('Max Force (eV/Å)')
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()

def main():
    """主函数：测试所有金属的HEPBA结构并生成对比图表"""
    # 所有金属
    metals = ['Fe', 'Co', 'Ni', 'Mn', 'Cu']
    
    # 存储结果
    results = {}
    
    for metal in metals:
        print(f"\n{'='*50}")
        print(f"分析 {metal}-HEPBA 结构")
        print(f"{'='*50}")
        
        # 创建单金属HEPBA生成器
        print("1. 初始化HEPBA生成器...")
        hepba = HighEntropyHEPBA(
            metals_M=[metal],
            include_water=True
        )
        
        # 创建结构
        print("\n2. 创建HEPBA结构...")
        structure = hepba.create_structure()
        print(f"   - 原子总数: {len(structure)}")
        print(f"   - 晶胞体积: {structure.get_volume():.2f} Å³")
        
        # 保存初始结构
        print("\n3. 保存初始结构...")
        write(f'initial_{metal}_hepba.cif', structure)
        print(f"   - 已保存到: initial_{metal}_hepba.cif")
        
        # 分析初始结构
        print("\n4. 分析初始结构...")
        initial_properties = hepba.analyze_structure(structure)
        print(f"   - 初始能量: {initial_properties['energy']:.4f} eV")
        print(f"   - 初始密度: {initial_properties['density']:.4f} g/cm³")
        print("   - 平均键长:")
        for bond, length in initial_properties['avg_bond_lengths'].items():
            print(f"     * {bond}: {length:.4f} Å")
        
        # 优化结构
        print("\n5. 开始结构优化...")
        trajectory, optimized_structure = hepba.optimize_structure(structure, max_steps=50)
        print(f"   - 优化步数: {len(trajectory)}")
        print(f"   - 最终能量: {trajectory[-1]['energy']:.4f} eV")
        print(f"   - 最大力: {trajectory[-1]['max_force']:.4f} eV/Å")
        
        # 保存优化后的结构
        print("\n6. 保存优化后结构...")
        write(f'optimized_{metal}_hepba.cif', optimized_structure)
        print(f"   - 已保存到: optimized_{metal}_hepba.cif")
        
        # 分析优化后的结构
        print("\n7. 分析优化后结构...")
        optimized_properties = hepba.analyze_structure(optimized_structure)
        print(f"   - 优化后能量: {optimized_properties['energy']:.4f} eV")
        print(f"   - 优化后密度: {optimized_properties['density']:.4f} g/cm³")
        print("   - 优化后平均键长:")
        for bond, length in optimized_properties['avg_bond_lengths'].items():
            print(f"     * {bond}: {length:.4f} Å")
        
        # 保存结果
        results[metal] = {
            'initial': initial_properties,
            'optimized': optimized_properties,
            'trajectory': trajectory
        }
        
        # 绘制优化过程
        print("\n8. 绘制优化过程...")
        hepba.plot_optimization(trajectory, f'optimization_{metal}_trajectory.png')
        print(f"   - 已保存到: optimization_{metal}_trajectory.png")
    
    # 保存结果到文件
    print("\n9. 保存分析结果...")
    import json
    for metal in results:
        with open(f'hepba_analysis_results_{metal}.json', 'w') as f:
            results_json = {
                metal: {
                    'initial': {
                        'energy': float(results[metal]['initial']['energy']),
                        'volume': float(results[metal]['initial']['volume']),
                        'density': float(results[metal]['initial']['density']),
                        'avg_bond_lengths': {k: float(v) for k, v in results[metal]['initial']['avg_bond_lengths'].items()}
                    },
                    'optimized': {
                        'energy': float(results[metal]['optimized']['energy']),
                        'volume': float(results[metal]['optimized']['volume']),
                        'density': float(results[metal]['optimized']['density']),
                        'avg_bond_lengths': {k: float(v) for k, v in results[metal]['optimized']['avg_bond_lengths'].items()}
                    }
                }
            }
            json.dump(results_json, f, indent=4)
        print(f"   - 已保存到: hepba_analysis_results_{metal}.json")
    
    # 生成综合对比表
    print("\n10. 生成综合对比表...")
    print("\n" + "="*100)
    print("HEPBA结构金属对比表")
    print("="*100)
    
    # 表头
    print(f"{'金属':<8} {'初始能量(eV)':<15} {'最终能量(eV)':<15} {'密度(g/cm³)':<15} {'M-N键长(Å)':<15} {'C-N键长(Å)':<15}")
    print("-"*100)
    
    # 按最终能量排序
    sorted_metals = sorted(results.keys(), key=lambda x: results[x]['optimized']['energy'])
    
    for metal in sorted_metals:
        result = results[metal]
        print(f"{metal:<8} "
              f"{result['initial']['energy']:<15.2f} "
              f"{result['optimized']['energy']:<15.2f} "
              f"{result['optimized']['density']:<15.4f} "
              f"{result['optimized']['avg_bond_lengths']['M-N']:<15.4f} "
              f"{result['optimized']['avg_bond_lengths']['C-N']:<15.4f}")
    
    print("="*100)
    
    # 生成对比图表
    print("\n11. 生成对比图表...")
    import matplotlib.pyplot as plt
    import numpy as np
    
    # 准备数据
    metals = sorted_metals
    initial_energies = [results[m]['initial']['energy'] for m in metals]
    optimized_energies = [results[m]['optimized']['energy'] for m in metals]
    densities = [results[m]['optimized']['density'] for m in metals]
    mn_lengths = [results[m]['optimized']['avg_bond_lengths']['M-N'] for m in metals]
    cn_lengths = [results[m]['optimized']['avg_bond_lengths']['C-N'] for m in metals]
    
    # 创建图表
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 能量对比
    ax1.bar(metals, initial_energies, label='初始')
    ax1.bar(metals, optimized_energies, label='优化后')
    ax1.set_ylabel('能量 (eV)')
    ax1.set_title('能量对比')
    ax1.legend()
    
    # 密度对比
    ax2.bar(metals, densities)
    ax2.set_ylabel('密度 (g/cm³)')
    ax2.set_title('密度对比')
    
    # M-N键长对比
    ax3.bar(metals, mn_lengths)
    ax3.set_ylabel('M-N键长 (Å)')
    ax3.set_title('M-N键长对比')
    
    # C-N键长对比
    ax4.bar(metals, cn_lengths)
    ax4.set_ylabel('C-N键长 (Å)')
    ax4.set_title('C-N键长对比')
    
    plt.tight_layout()
    plt.savefig('metal_comparison.png')
    print("   - 已保存到: metal_comparison.png")

if __name__ == "__main__":
    main() 