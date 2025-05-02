from pymatgen.analysis.local_env import VoronoiNN
import numpy as np
import torch

def analyze_optimized_structure(results):
    """
    Analyze optimized structure
    
    Args:
        results: Optimization results dictionary containing:
            - final_structure: Optimized structure
            - final_energy: Final energy
            - forces: Atomic forces
            - stress: Stress tensor
    
    Returns:
        dict: Analysis results containing:
            - average_bond_length: Average bond length
            - bond_length_std: Bond length standard deviation
            - lattice_angles: Lattice angles
            - lattice_lengths: Lattice lengths
            - pressure: Pressure
            - final_energy: Final energy
            - forces: Atomic forces
            - stress: Stress tensor
    
    Raises:
        ValueError: If input parameters are invalid
        RuntimeError: If analysis fails
    """
    if not isinstance(results, dict):
        raise ValueError("results must be a dictionary")
    required_keys = ['final_structure', 'final_energy', 'forces', 'stress']
    if not all(key in results for key in required_keys):
        raise ValueError(f"results must contain all required keys: {required_keys}")
    
    try:
        structure = results['final_structure']
        
        # Calculate bond lengths
        nn = VoronoiNN()
        bond_lengths = []
        for i, site in enumerate(structure):
            if site.species_string in ['C', 'N']:
                continue
            neighbors = nn.get_nn_info(structure, i)
            for neighbor in neighbors:
                bond_lengths.append(neighbor['weight'])
        
        # Calculate lattice distortion
        lattice = structure.lattice
        angles = lattice.angles
        lengths = lattice.lengths
        
        # Calculate stress and pressure
        stress = results['stress']
        if isinstance(stress, torch.Tensor):
            stress = stress.cpu().numpy()
        pressure = -np.trace(stress) / 3.0 if isinstance(stress, np.ndarray) else float(stress)
        
        # Convert forces to numpy if they are tensors
        forces = results['forces']
        if isinstance(forces, torch.Tensor):
            forces = forces.cpu().numpy()
        
        return {
            'average_bond_length': float(np.mean(bond_lengths)),
            'bond_length_std': float(np.std(bond_lengths)),
            'lattice_angles': angles,
            'lattice_lengths': lengths,
            'pressure': pressure,
            'final_energy': float(results['final_energy']),
            'forces': forces,
            'stress': stress
        }
    
    except Exception as e:
        raise RuntimeError(f"Analysis failed: {str(e)}") 