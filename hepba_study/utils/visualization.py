import matplotlib.pyplot as plt
import numpy as np

def plot_optimization_energy(trajectory, save_path='optimization_energy.png'):
    """
    Plot energy evolution during optimization
    
    Args:
        trajectory: Optimization trajectory
        save_path: Path to save the plot
    
    Raises:
        ValueError: If input parameters are invalid
    """
    if not trajectory:
        raise ValueError("trajectory cannot be empty")
    
    try:
        energies = [step.energy for step in trajectory]
        plt.figure(figsize=(10, 6))
        plt.plot(energies, 'b-')
        plt.xlabel('Optimization Steps')
        plt.ylabel('Energy (eV)')
        plt.title('Energy Evolution During Structure Optimization')
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()
    
    except Exception as e:
        raise RuntimeError(f"Failed to plot optimization energy: {str(e)}")

def plot_structure_properties(analysis_results, save_path='structure_properties.png'):
    """
    Plot key structure properties
    
    Args:
        analysis_results: Analysis results dictionary
        save_path: Path to save the plot
    
    Raises:
        ValueError: If input parameters are invalid
    """
    if not isinstance(analysis_results, dict):
        raise ValueError("analysis_results must be a dictionary")
    
    required_keys = ['lattice_lengths', 'lattice_angles', 'stress', 'forces']
    if not all(key in analysis_results for key in required_keys):
        raise ValueError(f"analysis_results must contain all required keys: {required_keys}")
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot lattice parameters
        axes[0, 0].bar(['a', 'b', 'c'], analysis_results['lattice_lengths'])
        axes[0, 0].set_ylabel('Length (Å)')
        axes[0, 0].set_title('Lattice Parameters')
        
        # Plot lattice angles
        axes[0, 1].bar(['α', 'β', 'γ'], analysis_results['lattice_angles'])
        axes[0, 1].set_ylabel('Angle (degrees)')
        axes[0, 1].set_title('Lattice Angles')
        
        # Plot stress components
        stress = analysis_results['stress']
        if isinstance(stress, np.ndarray):
            stress = stress.flatten()
            if len(stress) == 9:  # Full stress tensor
                stress_labels = ['xx', 'yy', 'zz', 'xy', 'yz', 'zx', 'yx', 'zy', 'xz']
            elif len(stress) == 6:  # Voigt notation
                stress_labels = ['xx', 'yy', 'zz', 'xy', 'yz', 'zx']
            else:  # Diagonal components only
                stress_labels = ['xx', 'yy', 'zz']
                stress = stress[:3]
        else:
            stress_labels = ['xx', 'yy', 'zz']
            stress = [stress, stress, stress]  # Assume isotropic stress
        
        axes[1, 0].bar(stress_labels, stress)
        axes[1, 0].set_ylabel('Stress (GPa)')
        axes[1, 0].set_title('Stress Components')
        
        # Plot force distribution
        forces = analysis_results['forces']
        if isinstance(forces, np.ndarray) and len(forces) > 0:
            force_magnitudes = np.linalg.norm(forces, axis=1)
            axes[1, 1].hist(force_magnitudes, bins=20)
            axes[1, 1].set_xlabel('Force Magnitude (eV/Å)')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].set_title('Force Distribution')
        else:
            axes[1, 1].text(0.5, 0.5, 'No force data available', 
                           ha='center', va='center')
            axes[1, 1].set_title('Force Distribution')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    
    except Exception as e:
        raise RuntimeError(f"Failed to plot structure properties: {str(e)}") 