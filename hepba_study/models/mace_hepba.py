"""使用MACE模型分析HEPBA结构"""

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

class HEPBAMACE:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float32):
        """初始化HEPBA分析器
        
        参数:
            device: 计算设备 (cuda/cpu)
            dtype: 数据类型
        """
        self.device = device
        self.dtype = dtype
        
        # 加载MACE模型
        mace_checkpoint_url = "https://github.com/ACEsuit/mace-foundations/releases/download/mace_mpa_0/mace-mpa-0-medium.model"
        self.loaded_model = mace_mp(
            model=mace_checkpoint_url,
            return_raw_model=True,
            default_dtype=dtype,
            device=device,
        )
        
        # 初始化MACE模型
        self.model = UnbatchedMaceModel(
            model=self.loaded_model,
            device=device,
            compute_forces=True,
            compute_stress=True,
            dtype=dtype,
            enable_cueq=False,
        )
        
        # 定义金属-氰基键长参考值 (Å)
        self.bond_lengths = {
            'Cu': {'M-N': 2.0, 'C-N': 1.15},  # 参考值，需要根据实验数据调整
            'Fe': {'M-N': 2.1, 'C-N': 1.15},
            'Mn': {'M-N': 2.2, 'C-N': 1.15},
            'Ni': {'M-N': 2.0, 'C-N': 1.15},
            'Co': {'M-N': 2.1, 'C-N': 1.15}
        }
        
        # 定义晶格常数参考值 (Å)
        self.lattice_constants = {
            'Cu': 10.0,  # 参考值，需要根据实验数据调整
            'Fe': 10.2,
            'Mn': 10.4,
            'Ni': 10.1,
            'Co': 10.2
        }
        
    def create_hepba_structure(self, metal, lattice_constant=None):
        """创建HEPBA结构
        
        参数:
            metal: 过渡金属元素 (Cu, Fe, Mn, Ni, Co)
            lattice_constant: 晶格常数 (Å)，如果为None则使用参考值
        """
        # 使用参考值或指定的晶格常数
        a = lattice_constant if lattice_constant is not None else self.lattice_constants[metal]
        
        # 创建基础立方晶格
        # 空间群Fm-3m (225)
        spacegroup = Spacegroup(225)
        
        # 定义晶胞中的原子位置
        # 金属原子位置 (0,0,0) 和 (0.5,0.5,0.5)
        # 氰基配体位置 (0.25,0.25,0.25) 和 (0.75,0.75,0.75)
        positions = np.array([
            [0.0, 0.0, 0.0],      # 金属原子1
            [0.5, 0.5, 0.5],      # 金属原子2
            [0.25, 0.25, 0.25],   # C原子
            [0.75, 0.75, 0.75],   # N原子
        ])
        
        # 定义原子类型
        symbols = [metal, metal, 'C', 'N']
        
        # 创建ASE Atoms对象
        cell = np.array([
            [a, 0, 0],
            [0, a, 0],
            [0, 0, a]
        ])
        
        structure = Atoms(
            symbols=symbols,
            positions=positions * a,
            cell=cell,
            pbc=True
        )
        
        # 根据空间群对称性扩展结构
        # 这里我们创建一个2x2x2的超胞
        structure = structure.repeat((2, 2, 2))
        
        # 调整氰基配体的位置
        # 确保M-C≡N-M'配位键的长度符合参考值
        self._adjust_cyanide_positions(structure, metal)
        
        return structure
    
    def _adjust_cyanide_positions(self, structure, metal):
        """调整氰基配体的位置，使其符合参考键长
        
        参数:
            structure: ASE Atoms对象
            metal: 过渡金属元素
        """
        # 获取参考键长
        m_n_length = self.bond_lengths[metal]['M-N']
        c_n_length = self.bond_lengths[metal]['C-N']
        
        # 获取金属和氰基原子的索引
        metal_indices = [i for i, atom in enumerate(structure) if atom.symbol == metal]
        c_indices = [i for i, atom in enumerate(structure) if atom.symbol == 'C']
        n_indices = [i for i, atom in enumerate(structure) if atom.symbol == 'N']
        
        # 调整C和N原子的位置
        for c_idx in c_indices:
            # 找到最近的金属原子
            c_pos = structure.positions[c_idx]
            nearest_metal_idx = min(metal_indices, 
                                  key=lambda i: np.linalg.norm(structure.positions[i] - c_pos))
            metal_pos = structure.positions[nearest_metal_idx]
            
            # 计算新的C原子位置
            direction = c_pos - metal_pos
            direction = direction / np.linalg.norm(direction)
            new_c_pos = metal_pos + direction * (m_n_length - c_n_length)
            structure.positions[c_idx] = new_c_pos
            
            # 找到对应的N原子
            for n_idx in n_indices:
                if np.linalg.norm(structure.positions[n_idx] - c_pos) < 2.0:
                    # 调整N原子位置
                    new_n_pos = new_c_pos + direction * c_n_length
                    structure.positions[n_idx] = new_n_pos
                    break
    
    def analyze_structure(self, structure):
        """分析结构性质
        
        参数:
            structure: ASE Atoms对象
        """
        # 转换为TorchSim状态
        state = ts.io.atoms_to_state(structure, device=self.device, dtype=self.dtype)
        
        # 运行MACE模型
        results = self.model(state)
        
        return {
            'energy': float(results['energy']),
            'forces': results['forces'].cpu().numpy(),
            'stress': results['stress'].cpu().numpy(),
            'volume': structure.get_volume(),
            'density': structure.get_masses().sum() / structure.get_volume()
        }
    
    def optimize_structure(self, structure, max_steps=100):
        """优化结构
        
        参数:
            structure: ASE Atoms对象
            max_steps: 最大优化步数
            
        返回:
            tuple: (trajectory, optimized_state)
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
        
        return trajectory, state
    
    def plot_optimization(self, trajectory, save_path=None):
        """绘制优化过程
        
        参数:
            trajectory: 优化轨迹数据
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
    # 创建分析器
    analyzer = HEPBAMACE()
    
    # 分析不同金属的HEPBA结构
    metals = ['Cu', 'Fe', 'Mn', 'Ni', 'Co']
    results = {}
    
    for metal in metals:
        print(f"\n分析 {metal}-HEPBA 结构:")
        
        # 创建结构
        structure = analyzer.create_hepba_structure(metal)
        
        # 分析初始结构
        initial_properties = analyzer.analyze_structure(structure)
        print(f"初始性质:")
        print(f"  能量: {initial_properties['energy']:.4f} eV")
        print(f"  体积: {initial_properties['volume']:.4f} Å³")
        print(f"  密度: {initial_properties['density']:.4f} g/cm³")
        
        # 优化结构
        print("\n开始结构优化...")
        trajectory, optimized_state = analyzer.optimize_structure(structure)
        
        # 分析优化后结构
        optimized_structure = structure.copy()
        optimized_structure.set_positions(optimized_state.positions.detach().cpu().numpy())
        optimized_properties = analyzer.analyze_structure(optimized_structure)
        
        print(f"\n优化后性质:")
        print(f"  能量: {optimized_properties['energy']:.4f} eV")
        print(f"  体积: {optimized_properties['volume']:.4f} Å³")
        print(f"  密度: {optimized_properties['density']:.4f} g/cm³")
        
        # 保存结果
        results[metal] = {
            'initial': initial_properties,
            'optimized': optimized_properties,
            'trajectory': trajectory
        }
        
        # 绘制优化过程
        save_path = Path(f"results/{metal}_optimization.png")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        analyzer.plot_optimization(trajectory, save_path)
    
    # 保存所有结果
    import json
    with open('results/hepba_analysis.json', 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main() 