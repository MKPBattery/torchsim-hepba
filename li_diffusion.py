import numpy as np
import torch
from pymatgen.core import Structure
from mace.calculators.foundations_models import mace_mp
import torch_sim as ts
from torch_sim.models.mace import MaceModel
from torch_sim.integrators import nvt_langevin
from torch_sim.runners import integrate
from torch_sim.trajectory import TrajectoryReporter
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import seaborn as sns

def load_optimal_structure(filename="optimal_ncm.cif"):
    """加载最优NCM结构"""
    return Structure.from_file(filename)

def calculate_msd(positions, reference_positions):
    """计算均方位移 (Mean Square Displacement)"""
    return np.mean(np.sum((positions - reference_positions) ** 2, axis=1))

def analyze_li_diffusion(structure, temperature=600, n_steps=10000, timestep=0.001):
    """分析Li离子扩散
    
    Args:
        structure: 初始结构
        temperature: 温度（K）
        n_steps: 模拟步数
        timestep: 时间步长（ps）
    """
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
    
    # 设置轨迹文件
    trajectory_file = f"li_diffusion_T{temperature}.h5md"
    reporter = TrajectoryReporter(
        trajectory_file,
        state_frequency=10,  # 每10步保存一次状态
    )
    
    print(f"开始分子动力学模拟 (T={temperature}K)...")
    final_state = integrate(
        system=structure,
        model=mace_model,
        integrator=nvt_langevin,
        n_steps=n_steps,
        temperature=temperature,
        timestep=timestep,
        trajectory_reporter=reporter,
    )
    
    # 分析轨迹
    print("分析Li扩散轨迹...")
    with ts.trajectory.TorchSimTrajectory(trajectory_file) as traj:
        n_frames = len(traj)
        li_indices = [i for i, s in enumerate(structure.species) if s.symbol == 'Li']
        n_li = len(li_indices)
        
        # 提取Li原子轨迹
        li_positions = np.zeros((n_frames, n_li, 3))
        for i in range(n_frames):
            atoms = traj.get_atoms(i)
            li_positions[i] = atoms.positions[li_indices]
        
        # 计算MSD
        msd = np.zeros(n_frames)
        for i in range(n_frames):
            msd[i] = calculate_msd(li_positions[i], li_positions[0])
        
        # 计算扩散系数
        time = np.arange(n_frames) * timestep * 10  # 考虑state_frequency=10
        D = msd[1:] / (6 * time[1:])  # Einstein关系
        D_avg = np.mean(D[-100:])  # 取最后100帧的平均值
        
        # 绘制MSD随时间的变化
        plt.figure(figsize=(10, 6))
        plt.plot(time, msd, 'b-', label='MSD')
        plt.xlabel('时间 (ps)')
        plt.ylabel('MSD (Å²)')
        plt.title(f'Li离子均方位移 (T={temperature}K)')
        plt.grid(True)
        plt.legend()
        plt.savefig(f'li_msd_T{temperature}.png')
        plt.close()
        
        # 绘制Li离子密度分布
        plt.figure(figsize=(12, 4))
        
        # XY平面投影
        plt.subplot(131)
        x = li_positions[:, :, 0].flatten()
        y = li_positions[:, :, 1].flatten()
        xy = np.vstack([x, y])
        z = gaussian_kde(xy)(xy)
        plt.scatter(x, y, c=z, s=1, cmap='viridis')
        plt.xlabel('X (Å)')
        plt.ylabel('Y (Å)')
        plt.title('Li密度分布 (XY平面)')
        
        # XZ平面投影
        plt.subplot(132)
        x = li_positions[:, :, 0].flatten()
        z = li_positions[:, :, 2].flatten()
        xz = np.vstack([x, z])
        density = gaussian_kde(xz)(xz)
        plt.scatter(x, z, c=density, s=1, cmap='viridis')
        plt.xlabel('X (Å)')
        plt.ylabel('Z (Å)')
        plt.title('Li密度分布 (XZ平面)')
        
        # YZ平面投影
        plt.subplot(133)
        y = li_positions[:, :, 1].flatten()
        z = li_positions[:, :, 2].flatten()
        yz = np.vstack([y, z])
        density = gaussian_kde(yz)(yz)
        plt.scatter(y, z, c=density, s=1, cmap='viridis')
        plt.xlabel('Y (Å)')
        plt.ylabel('Z (Å)')
        plt.title('Li密度分布 (YZ平面)')
        
        plt.tight_layout()
        plt.savefig(f'li_density_T{temperature}.png', dpi=300)
        plt.close()
        
        return {
            'diffusion_coefficient': D_avg,
            'msd': msd,
            'time': time,
            'li_positions': li_positions
        }

def main():
    # 加载最优结构
    structure = load_optimal_structure()
    
    # 在不同温度下分析Li扩散
    temperatures = [400, 600, 800]
    results = {}
    
    for T in temperatures:
        print(f"\n分析温度 {T}K 下的Li扩散...")
        results[T] = analyze_li_diffusion(structure, temperature=T)
        
    # 绘制不同温度下的扩散系数
    D_values = [results[T]['diffusion_coefficient'] for T in temperatures]
    plt.figure(figsize=(8, 6))
    plt.plot(temperatures, D_values, 'ro-')
    plt.xlabel('温度 (K)')
    plt.ylabel('扩散系数 (Å²/ps)')
    plt.title('Li离子扩散系数随温度的变化')
    plt.grid(True)
    plt.savefig('diffusion_vs_temperature.png')
    plt.close()
    
    # 打印结果
    print("\n扩散分析结果:")
    for T in temperatures:
        print(f"温度 {T}K:")
        print(f"扩散系数: {results[T]['diffusion_coefficient']:.4e} Å²/ps")

if __name__ == "__main__":
    main() 