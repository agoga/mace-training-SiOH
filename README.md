# mace-training-SIOH

This repository demonstrates a real-world workflow for preparing and (soon) training machine learning interatomic potentials (MACE) for SiOH systems, using Quantum ESPRESSO DFT calculations as the data source.

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
3. **Train MACE Potential:**
   Example command with recommended settings for Si/O/H systems:
   ```bash
   python train_mace.py
   # or, if running directly:
   mace_run_train \
     --model MACE \
     --train_file mace_training_data.xyz \
     --valid_file mace_training_data.xyz \
     --energy_weight 1.0 \
     --forces_weight 100.0 \
     --max_num_epochs 50 \
     --batch_size 1 \
     --device cuda \
     --work_dir mace_model_output \
     --name SiOH-test \
     --E0s '{1: -13.6, 8: -204.0, 14: -290.0}' \
     --num_workers 0 \
     --pin_memory False \
     --valid_batch_size 1 \
     --num_channels 128 \
     --num_interactions 4 \
     --max_L 1 \
     --r_max 6.0 \
     --lr 0.001 \
     --ema_decay 0.99 \
     --scheduler ReduceLROnPlateau \
     --seed 42
   ```
   These settings are based on best practices for Si/O/H systems (crystalline/amorphous Si, SiO₂, SiOx, H-containing) and are memory-friendly for most GPUs.
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
