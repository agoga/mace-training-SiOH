# mace-training-SIOH

This repository by [agoga](https://github.com/agoga) demonstrates a real-world workflow for preparing and (soon) training machine learning interatomic potentials (MACE) for SiOH systems, using Quantum ESPRESSO DFT calculations as the data source.

## Overview
- **Collects** all Quantum ESPRESSO `.in` and `.out` files from a project.
- **Parses** atomic structures, energies, and forces.
- **Converts** to an extended XYZ file (`mace_training_data.xyz`) compatible with [MACE](https://github.com/ACEsuit/mace).

## Project Status & Roadmap
This repository is a **work in progress**. Currently, it provides:
- Scripts to collect and convert DFT data to a MACE-ready format.

**Planned additions:**
- Scripts and workflows to train a MACE potential on the generated data.
- Example training runs and evaluation scripts.
- Documentation for the full end-to-end process.

## Typical Workflow
1. **Collect DFT Data:** Place all Quantum ESPRESSO `.in` and `.out` files in the `collected_inputs_outputs/` directory. Each pair must have matching base names.
2. **Convert to XYZ:**
   ```bash
   conda env create -f environment.yml
   conda activate mace
   python qe2mace.py
   ```
   This produces `mace_training_data.xyz`.
3. **(Planned) Train MACE Potential:**
   - Scripts and instructions for training with MACE will be added soon.
4. **(Planned) Validate and Use Potential:**
   - Example scripts for validation and deployment will be provided.

## File Structure
- `qe2mace.py` — Main script for parsing and conversion (Author: Adam Goga).
- `collected_inputs_outputs/` — Flat directory of all `.in`/`.out` files.
- `mace_training_data.xyz` — Output for MACE.
- `environment.yml` — Conda environment for reproducibility.
- (Planned) `train_mace.py`, `validate_mace.py`, etc. — Scripts for training and evaluation.

## License
MIT License. See [LICENSE](LICENSE).

## Citation
If you use this workflow, please cite Quantum ESPRESSO, MACE, and Adam Goga as appropriate.
