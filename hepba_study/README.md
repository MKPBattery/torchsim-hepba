# HEPBA电极材料研究项目

## 项目概述
本项目旨在研究高熵普鲁士蓝类似物(HEPBA)电极材料，通过引入多种过渡金属元素（Cu、Fe、Mn、Ni、Co）来优化其电化学性能。研究重点包括：

1. 单一金属掺杂对晶胞结构的影响
2. 自由能变化分析
3. 活性位点研究
4. 键力分析
5. 离子脱嵌过程中的结构演变

## 项目结构
```
hepba_study/
├── data/              # 存储原始数据和计算结果
├── analysis/          # 分析脚本
├── visualization/     # 可视化工具
└── models/           # 分子模拟模型
```

## 研究内容
1. 单一金属掺杂研究
   - 晶胞结构优化
   - 自由能计算
   - 活性位点分析
   - 键力分析

2. 多金属掺杂研究
   - 高熵效应分析
   - 协同作用研究
   - 电化学性能预测

## 依赖项
- Python 3.8+
- ASE (Atomic Simulation Environment)
- VASP
- pymatgen
- matplotlib
- numpy
- pandas

## 安装说明
```bash
# 克隆仓库
git clone [repository_url]

# 安装依赖
pip install -r requirements.txt
```

## 使用说明
1. 运行结构优化研究：
```bash
python -m hepba_study.structure_optimization.main
```

2. 自定义参数：
- 修改`main.py`中的参数设置
- 调整优化步数
- 更改元素组成

## 输出文件
- `optimized_hepba.cif`: 优化后的结构文件
- `optimization_energy.png`: 能量演化图
- `structure_properties.png`: 结构性质图

## 注意事项
1. 确保GPU可用性
2. 检查依赖版本兼容性
3. 定期保存计算结果
4. 注意内存使用情况

## 贡献指南
1. Fork项目
2. 创建特性分支
3. 提交更改
4. 发起Pull Request

## 许可证
[许可证类型]

## 联系方式
[联系信息]

## 更新日志
### v0.1.0 (2024-04-25)
- 初始版本发布
- 完成基本结构优化功能
- 实现基础可视化功能 