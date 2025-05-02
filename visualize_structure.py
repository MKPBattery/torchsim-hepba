import numpy as np
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io import write
import matplotlib.pyplot as plt
from ase.visualize.plot import plot_atoms

# Load the structure
structure = Structure.from_file('optimized_hepba.cif')
atoms = AseAtomsAdaptor.get_atoms(structure)

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

# Top view (along z-axis)
plot_atoms(atoms, ax1, rotation='0z,0x,0y', show_unit_cell=2)
ax1.set_title('Top View (along z-axis)')

# Perspective view
plot_atoms(atoms, ax2, rotation='30z,45x,0y', show_unit_cell=2)
ax2.set_title('Perspective View')

# Adjust layout and save
plt.tight_layout()
plt.savefig('hepba_structure.png', dpi=300, bbox_inches='tight')
plt.close()

# Also save structure in different formats for 3D visualization
write('hepba_structure.xyz', atoms)
write('hepba_structure_view.cif', atoms) 