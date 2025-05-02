# DFT计算参数配置
DFT_CONFIG = {
    'encut': 500,          # 截断能 (eV)
    'ismear': 0,           # 展宽方法
    'sigma': 0.05,         # 展宽参数 (eV)
    'ispin': 2,            # 自旋极化
    'lreal': 'Auto',       # 实空间投影
    'algo': 'Normal',      # 算法
    'ncore': 4,            # 并行核心数
    'kpts': [2, 2, 2],     # k点网格
    'xc': 'PBE',           # 交换关联泛函
    
    # 收敛标准
    'ediff': 1e-6,         # 能量收敛标准
    'ediffg': -0.02,       # 力收敛标准 (eV/Å)
    
    # 输出控制
    'nwrite': 2,           # 输出级别
    'lwave': False,        # 不保存波函数
    'lcharg': False,       # 不保存电荷密度
}

# 结构生成参数
STRUCTURE_CONFIG = {
    'lattice_constant': 10.2,  # 初始晶格常数 (Å)
    'metals': ['Cu', 'Fe', 'Mn', 'Ni', 'Co'],  # 研究的金属
    'vacuum': 10.0,        # 真空层厚度 (Å)
}

# 数据存储配置
STORAGE_CONFIG = {
    'base_dir': 'data/dft',  # 基础目录
    'structures_dir': 'structures',  # 结构文件目录
    'results_dir': 'results',  # 结果文件目录
    'analysis_dir': 'analysis',  # 分析结果目录
}

# 分析参数
ANALYSIS_CONFIG = {
    'force_threshold': 0.05,  # 力收敛阈值 (eV/Å)
    'stress_threshold': 0.1,  # 应力收敛阈值 (eV/Å³)
    'energy_threshold': 0.001,  # 能量收敛阈值 (eV)
} 