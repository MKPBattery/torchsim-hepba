import torch
from mace.calculators.foundations_models import mace_mp
from torch_sim.models.mace import MaceModel
from torch_sim.optimizers import unit_cell_fire
from torch_sim.runners import optimize
from pymatgen.core.structure import Structure
from ase import Atoms

def optimize_hepba_structure(structure, max_steps=1000):
    """
    Optimize HEPBA structure
    
    Args:
        structure: Initial structure
        max_steps: Maximum optimization steps
    
    Returns:
        dict: Optimization results containing:
            - final_structure: Optimized structure
            - final_energy: Final energy
            - forces: Atomic forces
            - stress: Stress tensor
            - positions: Atomic positions
            - cell: Unit cell
            - trajectory: Optimization trajectory
    
    Raises:
        ValueError: If input parameters are invalid
        RuntimeError: If optimization fails
    """
    if not isinstance(structure, (Structure, Atoms)):
        raise ValueError("structure must be a pymatgen Structure or ASE Atoms object")
    if not isinstance(max_steps, int) or max_steps <= 0:
        raise ValueError("max_steps must be a positive integer")
    
    try:
        # Set device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        # Initialize MACE model
        mace = mace_mp(model="small", return_raw_model=True)
        mace_model = MaceModel(
            model=mace,
            device=device,
            dtype=torch.float64,
            compute_forces=True,
        )
        
        # Run optimization
        print("Starting structure optimization...")
        state = optimize(
            system=structure,
            model=mace_model,
            optimizer=unit_cell_fire,
            max_steps=max_steps,
        )
        
        # Convert state to pymatgen Structure
        final_structure = state.to_structures()[0]
        
        return {
            'final_structure': final_structure,
            'final_energy': float(state.energy),
            'forces': state.forces,
            'stress': state.stress,
            'positions': state.positions,
            'cell': state.cell,
            'trajectory': state.trajectory if hasattr(state, 'trajectory') else []
        }
    
    except Exception as e:
        raise RuntimeError(f"Optimization failed: {str(e)}") 