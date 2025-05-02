from pymatgen.core import Structure, Lattice
import numpy as np

def create_hepba_structure(composition, lattice_params, metal_ratios):
    """
    Create HEPBA structure
    
    Args:
        composition: List of elements, e.g., ['Fe', 'Mn', 'Co', 'Ni', 'Cu']
        lattice_params: Lattice parameters [a, b, c, alpha, beta, gamma]
        metal_ratios: Ratios of each metal, e.g., [0.2, 0.2, 0.2, 0.2, 0.2]
    
    Returns:
        Structure: pymatgen Structure object
    
    Raises:
        ValueError: If input parameters are invalid
    """
    # Validate input parameters
    if not isinstance(composition, list) or not all(isinstance(elem, str) for elem in composition):
        raise ValueError("composition must be a list of strings")
    if not isinstance(lattice_params, (list, tuple)) or len(lattice_params) != 6:
        raise ValueError("lattice_params must be a list or tuple of 6 numbers")
    if not isinstance(metal_ratios, (list, tuple)) or len(metal_ratios) != len(composition):
        raise ValueError("metal_ratios must have the same length as composition")
    if not np.isclose(sum(metal_ratios), 1.0, atol=1e-6):
        raise ValueError("metal_ratios must sum to 1.0")
    
    try:
        # Create lattice
        lattice = Lattice.from_parameters(*lattice_params)
        
        # Define atomic positions (based on typical Prussian Blue Analog structure)
        # Metal sites
        metal_sites = [
            [0, 0, 0],          # M1
            [0.5, 0.5, 0],      # M2
            [0.5, 0, 0.5],      # M3
            [0, 0.5, 0.5],      # M4
            [0.5, 0.5, 0.5]     # M5
        ]
        
        # Cyanide sites
        cn_sites = [
            [0.25, 0.25, 0.25],  # C
            [0.75, 0.75, 0.75]   # N
        ]
        
        # Randomly assign metals according to ratios
        species = []
        coords = []
        
        # Add metal atoms
        for i, site in enumerate(metal_sites):
            # Randomly select metal based on ratios
            metal = np.random.choice(composition, p=metal_ratios)
            species.append(metal)
            coords.append(site)
        
        # Add cyanide
        species.extend(['C', 'N'])
        coords.extend(cn_sites)
        
        return Structure(lattice, species, coords)
    
    except Exception as e:
        raise ValueError(f"Failed to create structure: {str(e)}") 