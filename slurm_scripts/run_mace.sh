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


# Load the correct CUDA version
module load cuda/12.3


# Activate environment
source ~/.bashrc
conda activate mace

# Enable PyTorch memory expansion
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Log GPU info
nvidia-smi

# Run training
python train_mace.py