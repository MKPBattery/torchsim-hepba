# TorchSim HEPBA Project

This project provides an automated, extensible platform for the atomistic modeling and analysis of high-entropy Prussian blue analogues (HEPBA). By integrating state-of-the-art machine learning interatomic potentials (such as MACE) with efficient structure generation, optimization, and property analysis workflows, the project enables:

- Flexible construction of multi-metal HEPBA supercells, including structural water
- High-throughput structure optimization and property prediction using ML potentials
- Batch comparison and visualization of key structural and energetic properties across different metal compositions
- Output of standard structure files (.cif), analysis results (.json), and publication-ready plots

The codebase is designed for scientific rigor, reproducibility, and extensibility, making it a powerful tool for modern materials discovery and high-throughput screening.

[![CI](https://github.com/radical-ai/torch-sim/actions/workflows/test.yml/badge.svg)](https://github.com/radical-ai/torch-sim/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/radical-ai/torch-sim/branch/main/graph/badge.svg)](https://codecov.io/gh/radical-ai/torch-sim)
[![This project supports Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg?logo=python&logoColor=white)](https://python.org/downloads)
[![PyPI](https://img.shields.io/pypi/v/torch_sim_atomistic?logo=pypi&logoColor=white)](https://pypi.org/project/torch_sim_atomistic)
[![Zenodo](https://img.shields.io/badge/Zenodo-15127004-blue?logo=Zenodo&logoColor=white)][zenodo]

[zenodo]: https://zenodo.org/records/15127004

<!-- help docs find start of prose in readme, DO NOT REMOVE -->
TorchSim is a next-generation open-source atomistic simulation engine for the MLIP
era. By rewriting the core primitives of atomistic simulation in Pytorch, it allows
orders of magnitude acceleration of popular machine learning potentials.

* Automatic batching and GPU memory management allowing significant simulation speedup
* Support for MACE, Fairchem, SevenNet, ORB, MatterSim, graph-pes, and metatensor MLIP models
* Support for classical lennard jones, morse, and soft-sphere potentials
* Molecular dynamics integration schemes like NVE, NVT Langevin, and NPT Langevin
* Relaxation of atomic positions and cell with gradient descent and FIRE
* Swap monte carlo and hybrid swap monte carlo algorithm
* An extensible binary trajectory writing format with support for arbitrary properties
* A simple and intuitive high-level API for new users
* Integration with ASE, Pymatgen, and Phonopy
* and more: differentiable simulation, elastic properties, custom workflows...

## Quick Start

Here is a quick demonstration of many of the core features of TorchSim:
native support for GPUs, MLIP models, ASE integration, simple API,
autobatching, and trajectory reporting, all in under 40 lines of code.

### Running batched MD
<!-- tested in tests/test_runners::test_readme_example, update as needed -->

```py
import torch
import torch_sim as ts

# run natively on gpus
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# easily load the model from mace-mp
from mace.calculators.foundations_models import mace_mp
from torch_sim.models import MaceModel
mace = mace_mp(model="small", return_raw_model=True)
mace_model = MaceModel(model=mace, device=device)

from ase.build import bulk
cu_atoms = bulk("Cu", "fcc", a=3.58, cubic=True).repeat((2, 2, 2))
many_cu_atoms = [cu_atoms] * 50
trajectory_files = [f"Cu_traj_{i}.h5md" for i in range(len(many_cu_atoms))]

# run them all simultaneously with batching
final_state = ts.integrate(
    system=many_cu_atoms,
    model=mace_model,
    n_steps=50,
    timestep=0.002,
    temperature=1000,
    integrator=ts.nvt_langevin,
    trajectory_reporter=dict(filenames=trajectory_files, state_frequency=10),
)
final_atoms_list = final_state.to_atoms()

# extract the final energy from the trajectory file
final_energies = []
for filename in trajectory_files:
    with ts.TorchSimTrajectory(filename) as traj:
        final_energies.append(traj.get_array("potential_energy")[-1])

print(final_energies)
```

### Running batched relaxation

To then relax those structures with FIRE is just a few more lines.

```py
# relax all of the high temperature states
relaxed_state = ts.optimize(
    system=final_state,
    model=mace_model,
    optimizer=ts.frechet_cell_fire,
    autobatcher=True,
)

print(relaxed_state.energy)
```

## Speedup

TorchSim achieves up to 100x speedup compared to ASE with popular MLIPs.

![Speedup comparison](/docs/_static/speedup_plot.svg)

This figure compares the time per atom of ASE and `torch_sim`. Time per atom is defined
as the number of atoms / total time. While ASE can only run a single system of `n_atoms`
(on the $x$ axis), `torch_sim` can run as many systems as will fit in memory. On an H100 80 GB card,
the max atoms that could fit in memory was ~8,000 for [GemNet](https://github.com/FAIR-Chem/fairchem), ~10,000 for [MACE](https://github.com/ACEsuit/mace), and ~2,500
for [SevenNet](https://github.com/MDIL-SNU/SevenNet). This metric describes model performance by capturing speed and memory
usage simultaneously.

## Installation

### PyPI Installation

```sh
pip install torch-sim-atomistic
```

### Installing from source

```sh
git clone https://github.com/radical-ai/torch-sim
cd torch-sim
pip install .
```

## Examples

To understand how TorchSim works, start with the [comprehensive tutorials](https://radical-ai.github.io/torch-sim/user/overview.html) in the documentation.

## Core Modules

TorchSim's package structure is summarized in the [API reference](https://radical-ai.github.io/torch-sim/reference/index.html) documentation and drawn as a treemap below.

![TorchSim package treemap](https://github.com/user-attachments/assets/1e67879b-cdca-4ebc-bbbd-061fed90dfed)

## License

TorchSim is released under an [MIT license](LICENSE).

## Citation

A manuscript is in preparation. Meanwhile, if you use TorchSim in your research, please [cite the Zenodo archive][zenodo].

## Physical-Chemical Principles and Physical Reliability in MACE Machine Learning Potentials

This project utilizes MACE (Machine-Learning Atomic Cluster Expansion) and related machine learning interatomic potentials for large-scale atomistic simulations. Although MACE is fundamentally data-driven, its model design and training process incorporate a wide range of physical and chemical principles to ensure that the model is not a "black-box fit," but a physically reliable and efficient approximation. Key aspects include:

1. **Symmetry Constraints**
   - Translational, rotational, and atomic permutation symmetries are automatically satisfied, ensuring the invariance of energy and forces under these operations.
   - Local environment descriptors are constructed to be symmetry-invariant.

2. **Additivity of Energy**
   - The total energy is decomposed into a sum of local atomic energies, reflecting the physical principle of locality in atomic interactions.
   - Only atoms within a certain cutoff radius contribute to the local energy, neglecting long-range interactions as appropriate.

3. **Physical Consistency of Forces**
   - Forces are obtained as the negative gradient of energy with respect to atomic positions; both energy and forces are fitted during training to ensure physical consistency.
   - Stress tensors can also be predicted, maintaining the correct energy-force-stress relationships.

4. **Element Distinction and Chemical Information**
   - Atomic species are encoded in the input, allowing the model to distinguish different chemical elements.
   - Multi-body interactions (two-body, three-body, four-body, etc.) are included to capture complex chemical bonding.

5. **Physically Representative Training Sets**
   - Training data covers a wide range of structures, temperatures, defects, and compositions to ensure model generalization.
   - Unphysical high-energy structures are excluded to prevent the model from learning non-physical trends.

6. **Conservation Laws and Physical Constraints**
   - Energy conservation and Newton's third law (equal and opposite forces) are automatically satisfied.
   - Higher-order physical conservation (e.g., magnetism, polarity) can be supported in advanced models.

7. **Physical Interpretability**
   - Local energy distributions can be analyzed to understand structural stability.
   - Backpropagation enables tracing the physical origin of energy and force contributions.

8. **Generalization and Physical Reasonableness**
   - The model is reliable only within the physical-chemical space covered by the training set; extrapolation is flagged or unreliable.
   - Physical priors (e.g., hard-sphere repulsion, bond length ranges) can be incorporated to prevent non-physical predictions.

**In summary, MACE and similar ML potentials incorporate fundamental physical and chemical principles at every stage—from model structure and input/output to training objectives and data selection—ensuring that simulation results are physically meaningful and scientifically valuable.**

## Challenges and Limitations of Atomistic Simulations

While atomistic simulations (including DFT and ML-based potentials) are powerful tools for understanding and predicting material properties, it is important to recognize their inherent computational challenges and scientific limitations:

### 1. Enormous Computational Cost
- **Number of Atoms**: Real materials contain on the order of $10^{23}$ atoms, but simulations are typically limited to hundreds (DFT) or up to millions (ML potentials) of atoms due to computational constraints.
- **Degrees of Freedom**: Each atom has three spatial coordinates; a system with 1,000 atoms has 3,000 degrees of freedom, leading to a highly complex energy landscape.
- **Complex Interactions**: Accurate modeling requires considering not only pairwise but also many-body interactions, and for DFT, all electronic degrees of freedom, causing computational cost to scale steeply with system size.
- **Short Time Steps**: Atomic vibrations occur on the femtosecond (10⁻¹⁵ s) scale, so molecular dynamics simulations require millions to billions of time steps to reach nanosecond or microsecond timescales.
- **Sampling Limitations**: High-entropy or disordered materials have an astronomical number of possible atomic configurations; a single simulation samples only a tiny fraction of this space.

### 2. Model Approximations and Uncertainties
- **Potential Energy Surface Approximations**: DFT relies on exchange-correlation functionals, and ML potentials are trained on finite datasets—neither is a perfect representation of physical reality.
- **Transferability**: ML models may not reliably predict properties for structures or chemistries not represented in the training set.

### 3. System Size and Boundary Effects
- **Finite Size Effects**: Simulated systems are much smaller than real materials, and artificial boundary conditions (e.g., periodic boundaries) can introduce artifacts.

### 4. Dynamics and Statistical Averaging
- **Limited Sampling**: Many material properties depend on long-time, ensemble-averaged behavior, while simulations typically provide only short trajectories or snapshots.
- **Real-World Complexity**: Experimental materials contain impurities, defects, and stresses that are difficult to fully capture in simulations.

### 5. Scale Gap Between Simulation and Experiment
- **Macroscopic vs. Microscopic**: Experiments measure macroscopic averages or distributions, while simulations provide microscopic snapshots or limited statistical samples.
- **Multiscale Phenomena**: Many important processes (e.g., phase transitions, aging, interfacial reactions) span multiple length and time scales, beyond the reach of single-scale atomistic simulations.

### 6. Practical Examples
- **DFT**: Simulating a 100-atom system may require hours to days on a supercomputer.
- **ML Potentials**: Can handle tens of thousands of atoms, but still face memory and sampling bottlenecks for larger or longer simulations.
- **High-Entropy Materials**: The number of possible atomic arrangements is astronomical; a single simulation represents only a minuscule subset.

### 7. Scientific Attitude
- **Value**: Atomistic simulations are invaluable for revealing microscopic mechanisms, generating structural hypotheses, predicting trends, and guiding experiments.
- **Limitations**: Results should not be interpreted as absolute truth, but as one piece of evidence to be validated and complemented by experimental data.

**In summary, atomistic simulations provide deep insights into the microscopic world, but their results are subject to computational, statistical, and model-based uncertainties. They are best used as a tool for hypothesis generation and trend prediction, in close conjunction with experimental validation.**

## Project Achievements and Implemented Features

This project has accomplished the following core functionalities and results:

### 1. Automated Generation and Optimization of High-Entropy Prussian Blue Analogues (HEPBA)
- Flexible specification of multiple metal elements at the M site, enabling high-entropy design.
- Generation of realistic 3D supercell structures, including structural water.
- Automatic output of structure files (.cif) for downstream visualization and analysis.

### 2. Integration of MACE and Other Machine Learning Potentials
- Successful loading and application of MACE and related ML interatomic potentials for efficient prediction of energy, forces, and stress in large systems.
- GPU acceleration supported, significantly improving simulation efficiency.

### 3. Structure Optimization and Property Analysis
- Support for optimization algorithms such as FIRE, enabling automatic relaxation to stable structures.
- Automatic output of energy and force trajectory plots during optimization.
- Automated analysis and reporting of key properties: energy, density, volume, M-N/C-N bond lengths, etc.

### 4. Batch Comparison and Analysis of Multi-Metal Systems
- Batch generation, optimization, and comparison of single-metal HEPBA structures (Fe, Co, Ni, Mn, Cu, etc.).
- Automatic generation of comparison tables and plots (e.g., energy, density, bond lengths), facilitating structure-property relationship analysis.

### 5. Structure Visualization and Data Output
- Automatic saving of both initial and optimized structures (.cif), directly viewable in tools like VESTA.
- Automatic saving of analysis results (.json), comparison plots (.png), and other data for further processing and publication.

### 6. Scientific Rigor and Extensibility
- README provides detailed explanations of the physical-chemical constraints in MACE and the scientific limitations of atomistic simulations, demonstrating the project's scientific rigor.
- Codebase is well-structured and extensible, supporting further development for more metal combinations, different potentials, and new simulation tasks.

**In summary, this project enables automated structure generation, ML potential-driven optimization and analysis, batch comparison, and visualization for high-entropy Prussian blue analogues, with strong scientific grounding and extensibility—making it a powerful tool for modern materials simulation and high-throughput screening.**

## Output Files and Their Descriptions

This project generates a variety of output files during structure generation, optimization, and analysis. Below is a detailed description of each output type:

### 1. Structure Files (`.cif`)
- **`initial_{metal}_hepba.cif`**: The initial (unrelaxed) structure of the HEPBA supercell for each metal (e.g., `initial_Fe_hepba.cif`).
- **`optimized_{metal}_hepba.cif`**: The optimized (relaxed) structure after energy minimization for each metal (e.g., `optimized_Cu_hepba.cif`).
- **Usage**: These files can be visualized and analyzed using crystallographic software such as VESTA, OVITO, or ASE.

### 2. Analysis Results (`.json`)
- **`hepba_analysis_results_{metal}.json`**: Contains detailed property data for each metal's HEPBA structure, including:
  - Initial and optimized energy
  - Volume and density
  - Average bond lengths (M-N, C-N, etc.)
  - Other structural and energetic properties
- **Usage**: Useful for quantitative comparison, further data analysis, or integration into reports and publications.

### 3. Optimization Trajectory Plots (`.png`)
- **`optimization_{metal}_trajectory.png`**: Plots showing the evolution of energy and maximum force during the structure optimization process for each metal.
- **Usage**: Helps assess the convergence and stability of the optimization process.

### 4. Comparison Plots (`metal_comparison.png`)
- **`metal_comparison.png`**: A summary figure comparing key properties (energy, density, bond lengths, etc.) across all studied metals in a single view.
- **Usage**: Facilitates visual comparison and highlights trends or differences between different HEPBA compositions.

### 5. Other Data Files
- **Trajectory files (`.h5`, `.h5md`, `.hdf5`, `.traj`)**: If molecular dynamics or batch simulations are performed, these files store atomic trajectories and can be analyzed with ASE or TorchSim tools.

**In summary, the output files provide comprehensive structural, energetic, and comparative information for high-entropy Prussian blue analogues, supporting both in-depth analysis and publication-quality visualization.**
