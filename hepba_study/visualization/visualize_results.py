import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ase.io import read
from ase.visualize import view
import seaborn as sns

class HEPBAVisualizer:
    def __init__(self, structure_file):
        """
        初始化HEPBA可视化类
        
        参数:
            structure_file: CIF文件路径
        """
        self.structure = read(structure_file)
        
    def plot_structure_3d(self, filename='structure_3d.png'):
        """绘制3D结构图"""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 获取原子位置和类型
        positions = self.structure.get_positions()
        symbols = self.structure.get_chemical_symbols()
        
        # 为不同元素设置不同颜色
        colors = {'Cu': 'red', 'Fe': 'blue', 'Mn': 'green', 
                 'Ni': 'purple', 'Co': 'orange', 'C': 'black', 'N': 'cyan'}
        
        # 绘制原子
        for i, (pos, symbol) in enumerate(zip(positions, symbols)):
            ax.scatter(pos[0], pos[1], pos[2], 
                      c=colors.get(symbol, 'gray'),
                      s=100, label=symbol if i == 0 else "")
        
        # 绘制键
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < 2.0:  # 假设键长小于2.0 Å
                    ax.plot([positions[i][0], positions[j][0]],
                           [positions[i][1], positions[j][1]],
                           [positions[i][2], positions[j][2]], 'k-')
        
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_zlabel('Z (Å)')
        plt.title('HEPBA结构3D可视化')
        plt.legend()
        plt.savefig(filename)
        plt.close()
    
    def plot_energy_landscape(self, energies, filename='energy_landscape.png'):
        """绘制能量景观图"""
        plt.figure(figsize=(10, 6))
        sns.heatmap(energies, cmap='viridis')
        plt.xlabel('X坐标')
        plt.ylabel('Y坐标')
        plt.title('HEPBA能量景观')
        plt.colorbar(label='能量 (eV)')
        plt.savefig(filename)
        plt.close()
    
    def plot_bond_length_distribution(self, bond_lengths, filename='bond_lengths.png'):
        """绘制键长分布图"""
        plt.figure(figsize=(10, 6))
        plt.hist(bond_lengths, bins=20, alpha=0.7)
        plt.xlabel('键长 (Å)')
        plt.ylabel('频数')
        plt.title('HEPBA键长分布')
        plt.savefig(filename)
        plt.close()
    
    def plot_active_sites(self, active_sites, filename='active_sites.png'):
        """绘制活性位点图"""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        positions = self.structure.get_positions()
        
        # 绘制所有原子
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                  c='gray', s=50, alpha=0.3)
        
        # 高亮显示活性位点
        active_positions = positions[active_sites]
        ax.scatter(active_positions[:, 0], active_positions[:, 1], active_positions[:, 2],
                  c='red', s=100, label='活性位点')
        
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_zlabel('Z (Å)')
        plt.title('HEPBA活性位点分布')
        plt.legend()
        plt.savefig(filename)
        plt.close()
    
    def create_animation(self, trajectory_file, output_file='animation.gif'):
        """创建结构演化动画"""
        # 这里需要实现动画创建的逻辑
        # 可以使用matplotlib.animation或ase.visualize
        pass

def main():
    # 测试可视化功能
    metals = ['Cu', 'Fe', 'Mn', 'Ni', 'Co']
    
    for metal in metals:
        print(f"\n可视化 {metal}-HEPBA 结构:")
        visualizer = HEPBAVisualizer(f'data/{metal}_hepba.cif')
        
        # 绘制3D结构
        visualizer.plot_structure_3d(f'visualization/{metal}_structure_3d.png')
        
        # 创建示例数据用于其他可视化
        energies = np.random.rand(10, 10)  # 示例能量数据
        bond_lengths = np.random.normal(2.0, 0.1, 100)  # 示例键长数据
        active_sites = np.random.choice(10, 3)  # 示例活性位点
        
        # 绘制其他图表
        visualizer.plot_energy_landscape(energies, f'visualization/{metal}_energy_landscape.png')
        visualizer.plot_bond_length_distribution(bond_lengths, f'visualization/{metal}_bond_lengths.png')
        visualizer.plot_active_sites(active_sites, f'visualization/{metal}_active_sites.png')

if __name__ == "__main__":
    main() 