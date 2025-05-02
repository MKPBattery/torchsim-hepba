import numpy as np
from hepba_study.structure_optimization.create_structure import create_hepba_structure
from hepba_study.structure_optimization.optimize_structure import optimize_hepba_structure
from hepba_study.structure_optimization.analyze_structure import analyze_optimized_structure
from hepba_study.utils.visualization import plot_optimization_energy, plot_structure_properties

def main():
    """
    Main function to run the HEPBA structure optimization study
    
    Raises:
        RuntimeError: If any step of the process fails
    """
    try:
        # Set parameters
        composition = ['Fe', 'Co', 'Ni', 'Mn', 'Cu']
        metal_ratios = [0.2, 0.2, 0.2, 0.2, 0.2]
        lattice_params = [10.0, 10.0, 10.0, 90.0, 90.0, 90.0]  # [a, b, c, alpha, beta, gamma]
        
        # Create initial structure
        print("Creating initial structure...")
        structure = create_hepba_structure(composition, lattice_params, metal_ratios)
        
        # Optimize structure
        print("Optimizing structure...")
        optimization_results = optimize_hepba_structure(structure, max_steps=100)
        
        # Analyze results
        print("Analyzing results...")
        analysis_results = analyze_optimized_structure(optimization_results)
        
        # Generate visualizations
        print("Generating visualizations...")
        if optimization_results.get('trajectory'):
            plot_optimization_energy(optimization_results['trajectory'])
        else:
            print("Warning: No trajectory data available for energy plot")
        plot_structure_properties(analysis_results)
        
        # Print analysis results
        print("\nAnalysis Results:")
        print(f"Final Energy: {analysis_results['final_energy']:.6f} eV")
        print(f"Average Bond Length: {analysis_results['average_bond_length']:.6f} Å")
        print(f"Bond Length Standard Deviation: {analysis_results['bond_length_std']:.6f} Å")
        print(f"Lattice Parameters: {analysis_results['lattice_lengths']}")
        print(f"Lattice Angles: {analysis_results['lattice_angles']}")
        
        # Handle pressure value
        pressure = analysis_results['pressure']
        if isinstance(pressure, (float, int)):
            print(f"Pressure: {pressure:.6f} GPa")
        elif isinstance(pressure, np.ndarray):
            if pressure.ndim == 2:  # Full stress tensor
                pressure = -np.trace(pressure) / 3.0
            elif pressure.ndim == 1:  # Diagonal components
                pressure = -np.mean(pressure)
            else:
                pressure = float(pressure)  # Single value
            print(f"Pressure: {pressure:.6f} GPa")
        else:
            print("Warning: Pressure data not available")
        
        # Save optimized structure
        print("\nSaving optimized structure...")
        optimization_results['final_structure'].to(filename='optimized_hepba.cif')
        
        print("\nOptimization study completed successfully!")
    
    except Exception as e:
        raise RuntimeError(f"Optimization study failed: {str(e)}")

if __name__ == "__main__":
    main() 