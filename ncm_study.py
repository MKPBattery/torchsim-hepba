import numpy as np
import torch
from ase.build import bulk
from pymatgen.core import Structure, Lattice
from mace.calculators.foundations_models import mace_mp
import torch_sim as ts
from torch_sim.models.mace import MaceModel
from torch_sim.optimizers import unit_cell_fire
from torch_sim.runners import optimize
from torch_sim.trajectory import TrajectoryReporter
import matplotlib.pyplot as plt
import seaborn as sns

def create_ncm_structure(ni_position, mn_position, a=2.8, c=14.2):
    """
    创建NCM结构
    ni_position: Ni在c轴方向的位置 (0-1)
    mn_position: Mn在c轴方向的位置 (0-1)
    """
    # 创建六方晶胞
    lattice = Lattice.from_parameters(a, a, c, 90, 90, 120)
    
    # 定义原子位置和种类
    species = ["Li"] * 3 + ["Ni", "Co", "Mn"] + ["O"] * 6
    coords = [
        [1/3, 2/3, 0.25],  # Li1
        [1/3, 2/3, 0.75],  # Li2
        [0, 0, 0.5],       # Li3
        [0, 0, ni_position],  # Ni
        [1/3, 2/3, 0.5],   # Co
        [2/3, 1/3, mn_position],  # Mn
        [0, 0, 0],         # O1
        [0, 0, 1],         # O2
        [1/3, 2/3, 0],     # O3
        [1/3, 2/3, 1],     # O4
        [2/3, 1/3, 0],     # O5
        [2/3, 1/3, 1],     # O6
    ]
    
    return Structure(lattice, species, coords)

def calculate_formation_energy(structure, mace_model):
    """计算形成能"""
    state = optimize(
        system=structure,
        model=mace_model,
        optimizer=unit_cell_fire,
        max_steps=1000,
    )
    return float(state.energy)  # 转换为Python浮点数

def plot_energy_heatmap(energies, positions, save_path="ncm_stability.png"):
    """绘制能量热图"""
    # 设置绘图风格
    plt.style.use('seaborn')
    sns.set_palette("viridis")
    
    # 创建主图和颜色条
    fig = plt.figure(figsize=(12, 10))
    gs = plt.GridSpec(2, 2, width_ratios=[4, 0.3], height_ratios=[4, 1])
    
    # 主热图
    ax_heat = plt.subplot(gs[0, 0])
    im = ax_heat.imshow(energies, cmap='viridis', aspect='auto', 
                       extent=[positions[0], positions[-1], positions[0], positions[-1]])
    
    # 添加等高线
    levels = np.linspace(np.min(energies), np.max(energies), 10)
    contours = ax_heat.contour(positions, positions, energies, levels=levels, colors='white', alpha=0.5)
    ax_heat.clabel(contours, inline=True, fontsize=8, fmt='%.2f')
    
    # 设置轴标签和标题
    ax_heat.set_xlabel('Mn位置 (相对坐标)', fontsize=12)
    ax_heat.set_ylabel('Ni位置 (相对坐标)', fontsize=12)
    ax_heat.set_title('NCM材料中Ni和Mn位置对稳定性的影响\n(形成能分布)', fontsize=14, pad=20)
    
    # 添加颜色条
    cax = plt.subplot(gs[0, 1])
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('形成能 (eV)', fontsize=12)
    
    # 添加能量剖面图
    ax_profile_x = plt.subplot(gs[1, 0])
    for i, ni_pos in enumerate(positions):
        ax_profile_x.plot(positions, energies[i, :], 
                         label=f'Ni={ni_pos:.1f}', alpha=0.7, linewidth=1)
    ax_profile_x.set_xlabel('Mn位置 (相对坐标)', fontsize=12)
    ax_profile_x.set_ylabel('形成能 (eV)', fontsize=12)
    ax_profile_x.legend(bbox_to_anchor=(1.15, 1), loc='upper left', 
                       title='不同Ni位置的能量剖面')
    
    # 找到最稳定点
    min_idx = np.unravel_index(np.argmin(energies), energies.shape)
    min_ni, min_mn = positions[min_idx[0]], positions[min_idx[1]]
    ax_heat.plot(min_mn, min_ni, 'r*', markersize=15, 
                label=f'最稳定点\nNi={min_ni:.2f}\nMn={min_mn:.2f}\n能量={energies[min_idx]:.2f}eV')
    ax_heat.legend(loc='upper right', bbox_to_anchor=(0.95, 0.95))
    
    # 添加网格
    ax_heat.grid(True, linestyle='--', alpha=0.3)
    ax_profile_x.grid(True, linestyle='--', alpha=0.3)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 初始化MACE模型
    mace = mace_mp(model="small", return_raw_model=True)
    mace_model = MaceModel(
        model=mace,
        device=device,
        dtype=torch.float64,
        compute_forces=True,
    )
    
    # 研究不同Ni和Mn位置
    positions = np.linspace(0.1, 0.9, 15)  # 增加采样点数量，从9个增加到15个
    energies = np.zeros((len(positions), len(positions)))
    
    print("开始计算不同位置的稳定性...")
    for i, ni_pos in enumerate(positions):
        for j, mn_pos in enumerate(positions):
            print(f"计算 Ni位置={ni_pos:.2f}, Mn位置={mn_pos:.2f}")
            structure = create_ncm_structure(ni_pos, mn_pos)
            energy = calculate_formation_energy(structure, mace_model)
            energies[i, j] = energy
            print(f"形成能: {energy:.4f} eV")
            
            # 实时保存数据
            np.save('energies.npy', energies)
            np.save('positions.npy', positions)
    
    # 绘制热图
    plot_energy_heatmap(energies, positions)
    
    # 找到最稳定的结构
    min_energy_idx = np.unravel_index(np.argmin(energies), energies.shape)
    optimal_ni_pos = positions[min_energy_idx[0]]
    optimal_mn_pos = positions[min_energy_idx[1]]
    print(f"\n最稳定的结构:")
    print(f"Ni位置: {optimal_ni_pos:.2f}")
    print(f"Mn位置: {optimal_mn_pos:.2f}")
    print(f"形成能: {energies[min_energy_idx]:.4f} eV")
    
    # 保存最稳定结构
    optimal_structure = create_ncm_structure(optimal_ni_pos, optimal_mn_pos)
    optimal_structure.to(filename="optimal_ncm.cif")

if __name__ == "__main__":
    main() 