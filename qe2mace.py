"""
qe2mace.py
----------

Script to parse Quantum ESPRESSO input/output files and convert them into a MACE-compatible extended XYZ file for machine learning interatomic potential training.

- Collects all .in/.out pairs from a flat directory (collected_inputs_outputs/)
- Extracts cell, atomic positions, species, total energy, and forces
- Converts units to eV and eV/Å as needed
- Outputs a single mace_training_data.xyz file

Author: Adam Goga
Date: 2025-05-29
"""

import os
import glob
import re
import numpy as np
from ase import Atoms
from ase.io import write

# Directory containing the collected files
dirname = "collected_inputs_outputs"

# Output xyz file
output_xyz = "mace_training_data.xyz"

# Helper functions
def parse_qe_in(infile):
    """
    Parse Quantum ESPRESSO .in file to extract cell, atom symbols, and positions.

    Args:
        infile (str): Path to the Quantum ESPRESSO .in file to be parsed.

    Returns:
        tuple: A tuple containing:
            - cell (list of list of float): 3x3 cell vectors.
            - atoms (list of str): Atomic symbols.
            - positions (list of list of float): Atomic positions in Angstrom.
    """
    with open(infile) as f:
        lines = f.readlines()
    cell = []
    atoms = []
    species = []
    positions = []
    read_cell = False
    read_atoms = False
    for i, line in enumerate(lines):
        if line.strip().startswith('CELL_PARAMETERS'):
            read_cell = True
            cell = []
            continue
        if read_cell:
            if len(cell) < 3:
                cell.append([float(x) for x in line.split()[:3]])
                if len(cell) == 3:
                    read_cell = False
            continue
        if line.strip().startswith('ATOMIC_SPECIES'):
            j = i + 1
            while lines[j].strip() and not lines[j].startswith('ATOMIC_POSITIONS'):
                species.append(lines[j].split()[0])
                j += 1
        if line.strip().startswith('ATOMIC_POSITIONS'):
            read_atoms = True
            continue
        if read_atoms:
            if not line.strip() or line.strip().startswith('K_POINTS'):
                break
            parts = line.split()
            if len(parts) >= 4:
                atoms.append(parts[0])
                positions.append([float(x) for x in parts[1:4]])
    return cell, atoms, positions

def parse_qe_out(outfile):
    """
    Parse Quantum ESPRESSO .out file to extract total energy and forces.

    Args:
        outfile (str): Path to the Quantum ESPRESSO .out file to be parsed.

    Returns:
        tuple: A tuple containing:
            - energy (float): Total energy in eV.
            - forces (np.ndarray): Forces in eV/Å.
    """
    with open(outfile) as f:
        lines = f.readlines()
    # Energy
    energy = None
    for line in reversed(lines):
        if '!    total energy' in line:
            energy = float(line.split('=')[1].split()[0]) * 13.605698  # Ry to eV
            break
    # Forces
    forces = []
    for i, line in enumerate(lines):
        if 'Forces acting on atoms' in line:
            j = i + 1
            while j < len(lines):
                l = lines[j].strip()
                if l == '':
                    j += 1
                    continue
                if 'force =' in l:
                    force_vals = l.split('force =')[1].split()
                    if len(force_vals) >= 3:
                        try:
                            fx = float(force_vals[0])
                            fy = float(force_vals[1])
                            fz = float(force_vals[2])
                            forces.append([fx, fy, fz])
                        except ValueError:
                            pass
                    j += 1
                else:
                    break
            break
    # Convert forces from Ry/au to eV/Å
    # 1 Ry/bohr = 25.71104309541616 eV/Å
    forces = [[f * 25.711043 for f in atom] for atom in forces]
    return energy, forces

def main():
    """
    Main routine: loops over all .in/.out pairs, parses data, and writes to extended XYZ.
    """
    xyz_structures = []
    in_files = sorted(glob.glob(os.path.join(dirname, '*.in')))
    print(f"Found {len(in_files)} .in files.")
    for in_file in in_files:
        base = os.path.splitext(os.path.basename(in_file))[0]
        out_file = os.path.join(dirname, base + '.out')
        if not os.path.exists(out_file):
            continue
        cell, symbols, positions = parse_qe_in(in_file)
        energy, forces = parse_qe_out(out_file)
        if not (cell and symbols and positions and energy is not None and forces):
            continue
        atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)
        atoms.info['energy'] = energy
        atoms.arrays['forces'] = np.array(forces)
        xyz_structures.append(atoms)
    if xyz_structures:
        write(output_xyz, xyz_structures)
        print(f"Wrote {len(xyz_structures)} structures to {output_xyz}")
    else:
        print("No structures parsed.")

if __name__ == "__main__":
    main()
