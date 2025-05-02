import numpy as np
from pymatgen.ext.matproj import MPRester
from pymatgen.core import Structure
import json
from pathlib import Path
import h5py

class DatabaseLoader:
    def __init__(self, api_key=None, base_dir='data/database'):
        """
        初始化数据库加载器
        
        参数:
            api_key: Materials Project API密钥
            base_dir: 数据存储目录
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化MP客户端
        self.mpr = MPRester(api_key) if api_key else None
        
    def load_hepba_structures(self, metals=['Cu', 'Fe', 'Mn', 'Ni', 'Co']):
        """
        从数据库加载HEPBA结构数据
        
        参数:
            metals: 要研究的金属列表
        """
        structures = {}
        properties = {}
        
        for metal in metals:
            print(f"\n加载 {metal}-HEPBA 数据:")
            
            # 从MP数据库搜索HEPBA结构
            query = {
                'elements': {'$all': [metal, 'C', 'N']},
                'nelements': 3,
                'spacegroup.number': 225  # 普鲁士蓝的对称性
            }
            
            try:
                # 获取结构数据
                entries = self.mpr.get_entries(query)
                
                if entries:
                    # 保存结构
                    struct_dir = self.base_dir / 'structures' / metal
                    struct_dir.mkdir(parents=True, exist_ok=True)
                    
                    for i, entry in enumerate(entries):
                        structure = entry.structure
                        structure.to(fmt='cif', filename=str(struct_dir / f'structure_{i}.cif'))
                        
                        # 保存性质
                        properties[f"{metal}_{i}"] = {
                            'energy': entry.energy,
                            'energy_per_atom': entry.energy_per_atom,
                            'formation_energy': entry.formation_energy_per_atom,
                            'band_gap': entry.band_gap,
                            'volume': structure.volume,
                            'density': structure.density
                        }
                    
                    structures[metal] = [entry.structure for entry in entries]
                    print(f"找到 {len(entries)} 个 {metal}-HEPBA 结构")
                else:
                    print(f"未找到 {metal}-HEPBA 结构")
                    
            except Exception as e:
                print(f"加载 {metal}-HEPBA 数据失败: {str(e)}")
        
        return structures, properties
    
    def save_data(self, structures, properties):
        """
        保存加载的数据
        
        参数:
            structures: 结构字典
            properties: 性质字典
        """
        # 保存为JSON
        with open(self.base_dir / 'database_results.json', 'w') as f:
            json.dump(properties, f, indent=4)
        
        # 保存为HDF5
        with h5py.File(self.base_dir / 'database_results.h5', 'w') as f:
            for metal, struct_list in structures.items():
                group = f.create_group(metal)
                for i, structure in enumerate(struct_list):
                    subgroup = group.create_group(f'structure_{i}')
                    # 保存晶胞参数
                    subgroup.create_dataset('cell', data=structure.lattice.matrix)
                    # 保存原子位置
                    subgroup.create_dataset('positions', data=structure.cart_coords)
                    # 保存原子类型
                    subgroup.create_dataset('species', data=[str(s) for s in structure.species])
    
    def analyze_data(self):
        """
        分析加载的数据
        """
        # 读取性质数据
        with open(self.base_dir / 'database_results.json', 'r') as f:
            properties = json.load(f)
        
        # 分析每个金属的结果
        analysis = {}
        for key, data in properties.items():
            metal = key.split('_')[0]
            if metal not in analysis:
                analysis[metal] = {
                    'count': 0,
                    'energies': [],
                    'formation_energies': [],
                    'band_gaps': [],
                    'volumes': [],
                    'densities': []
                }
            
            analysis[metal]['count'] += 1
            analysis[metal]['energies'].append(data['energy'])
            analysis[metal]['formation_energies'].append(data['formation_energy'])
            analysis[metal]['band_gaps'].append(data['band_gap'])
            analysis[metal]['volumes'].append(data['volume'])
            analysis[metal]['densities'].append(data['density'])
        
        # 计算统计量
        for metal in analysis:
            data = analysis[metal]
            data['mean_energy'] = np.mean(data['energies'])
            data['mean_formation_energy'] = np.mean(data['formation_energies'])
            data['mean_band_gap'] = np.mean(data['band_gaps'])
            data['mean_volume'] = np.mean(data['volumes'])
            data['mean_density'] = np.mean(data['densities'])
            
            data['std_energy'] = np.std(data['energies'])
            data['std_formation_energy'] = np.std(data['formation_energies'])
            data['std_band_gap'] = np.std(data['band_gaps'])
            data['std_volume'] = np.std(data['volumes'])
            data['std_density'] = np.std(data['densities'])
        
        # 保存分析结果
        with open(self.base_dir / 'analysis.json', 'w') as f:
            json.dump(analysis, f, indent=4)
        
        return analysis

def main():
    # 创建数据库加载器
    loader = DatabaseLoader()
    
    # 加载数据
    print("从数据库加载HEPBA数据...")
    structures, properties = loader.load_hepba_structures()
    
    # 保存数据
    print("\n保存加载的数据...")
    loader.save_data(structures, properties)
    
    # 分析数据
    print("\n分析数据...")
    analysis = loader.analyze_data()
    
    print("\n分析完成！")
    for metal, data in analysis.items():
        print(f"\n{metal}-HEPBA 分析结果:")
        print(f"结构数量: {data['count']}")
        print(f"平均形成能: {data['mean_formation_energy']:.6f} ± {data['std_formation_energy']:.6f} eV/atom")
        print(f"平均带隙: {data['mean_band_gap']:.6f} ± {data['std_band_gap']:.6f} eV")
        print(f"平均体积: {data['mean_volume']:.6f} ± {data['std_volume']:.6f} Å³")
        print(f"平均密度: {data['mean_density']:.6f} ± {data['std_density']:.6f} g/cm³")

if __name__ == "__main__":
    main() 