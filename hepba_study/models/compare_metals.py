import json
import matplotlib.pyplot as plt
import numpy as np

def main():
    """主函数：生成所有金属的综合对比表"""
    # 所有金属
    metals = ['Fe', 'Co', 'Ni', 'Mn', 'Cu']
    
    # 合并所有结果
    all_results = {}
    for metal in metals:
        try:
            with open(f'hepba_analysis_results_{metal}.json', 'r') as f:
                all_results[metal] = json.load(f)[metal]
        except FileNotFoundError:
            print(f"警告：未找到 {metal} 的结果文件")
            continue
    
    if not all_results:
        print("错误：未找到任何金属的结果文件")
        return
    
    # 生成对比表
    print("\n" + "="*100)
    print("HEPBA结构金属对比表")
    print("="*100)
    
    # 表头
    print(f"{'金属':<8} {'初始能量(eV)':<15} {'最终能量(eV)':<15} {'密度(g/cm³)':<15} {'M-N键长(Å)':<15} {'C-N键长(Å)':<15}")
    print("-"*100)
    
    # 按最终能量排序
    sorted_metals = sorted(all_results.keys(), key=lambda x: all_results[x]['optimized']['energy'])
    
    for metal in sorted_metals:
        result = all_results[metal]
        print(f"{metal:<8} "
              f"{result['initial']['energy']:<15.2f} "
              f"{result['optimized']['energy']:<15.2f} "
              f"{result['optimized']['density']:<15.4f} "
              f"{result['optimized']['avg_bond_lengths']['M-N']:<15.4f} "
              f"{result['optimized']['avg_bond_lengths']['C-N']:<15.4f}")
    
    print("="*100)
    
    # 生成对比图表
    print("\n生成对比图表...")
    
    # 准备数据
    metals = sorted_metals
    initial_energies = [all_results[m]['initial']['energy'] for m in metals]
    optimized_energies = [all_results[m]['optimized']['energy'] for m in metals]
    densities = [all_results[m]['optimized']['density'] for m in metals]
    mn_lengths = [all_results[m]['optimized']['avg_bond_lengths']['M-N'] for m in metals]
    cn_lengths = [all_results[m]['optimized']['avg_bond_lengths']['C-N'] for m in metals]
    
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