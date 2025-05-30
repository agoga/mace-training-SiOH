#!/bin/bash
#SBATCH --account=soldeg
#SBATCH --partition=gpu-h100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=170G
#SBATCH --time=2-00:00:00
#SBATCH -J mace-train
#SBATCH --mail-user="adgoga@ucdavis.edu"
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --output=mace_train_%j.log

# Environment setup
module purge
module spider cuda 2>/dev/null | grep -q 11.8 && module load cuda/11.8 || echo "⚠️ cuda/11.8 not found, skipping module load"

source ~/.bashrc
conda activate mace

# Optional: log GPU memory before start
nvidia-smi

# Run training script
python train_mace.py