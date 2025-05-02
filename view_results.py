import numpy as np
import matplotlib.pyplot as plt
from torch_sim.trajectory import TorchSimTrajectory
import h5py

def print_trajectory_info(filename):
    print(f"\n分析文件: {filename}")
    with TorchSimTrajectory(filename) as traj:
        # 获取基本信息
        print(f"轨迹步数: {len(traj)}")
        print(f"原子数量: {len(traj.get_atoms(0))}")
        
        # 获取可用的数据
        print("\n可用的数据:")
        with h5py.File(filename, 'r') as f:
            for group in f['/particles/atoms/observables'].keys():
                print(f"- {group}")
        
        # 获取能量数据
        if 'potential_energy' in traj.get_array_names():
            energies = traj.get_array("potential_energy")
            print(f"\n势能范围: {min(energies):.4f} 到 {max(energies):.4f} eV")
            print(f"最终势能: {energies[-1]:.4f} eV")
        
        if 'kinetic_energy' in traj.get_array_names():
            energies = traj.get_array("kinetic_energy")
            print(f"动能范围: {min(energies):.4f} 到 {max(energies):.4f} eV")
            print(f"最终动能: {energies[-1]:.4f} eV")

# 查看Lennard-Jones模拟结果
print("=== Lennard-Jones模拟结果 ===")
print_trajectory_info("lj_trajectory.h5md")

# 查看批处理模拟结果
print("\n=== 批处理模拟结果 ===")
for i in range(4):
    filename = f"batch_traj_{i}.h5md"
    print(f"\n系统 {i}:")
    print_trajectory_info(filename)

# 绘制能量变化图
print("\n正在生成能量变化图...")
with TorchSimTrajectory("lj_trajectory.h5md") as traj:
    kinetic_energies = traj.get_array("kinetic_energy")
    potential_energies = traj.get_array("potential_energy")
    
    plt.figure(figsize=(10, 6))
    plt.plot(kinetic_energies, label='动能')
    plt.plot(potential_energies, label='势能')
    plt.plot(np.array(kinetic_energies) + np.array(potential_energies), label='总能量')
    plt.xlabel('时间步')
    plt.ylabel('能量 (eV)')
    plt.title('Lennard-Jones模拟能量变化')
    plt.legend()
    plt.savefig('lj_energies.png')
    plt.close()

for i in range(4):
    filename = f"batch_traj_{i}.h5md"
    with TorchSimTrajectory(filename) as traj:
        potential_energies = traj.get_array("potential_energy")
        plt.figure(figsize=(10, 6))
        plt.plot(potential_energies)
        plt.xlabel('时间步')
        plt.ylabel('势能 (eV)')
        plt.title(f'系统 {i} 的势能变化')
        plt.savefig(f'batch_energies_{i}.png')
        plt.close()

print("\n结果已保存为图片文件：")
print("- lj_energies.png: Lennard-Jones模拟的能量变化")
print("- batch_energies_0.png 到 batch_energies_3.png: 批处理模拟的能量变化") 