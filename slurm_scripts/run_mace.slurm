#!/bin/bash
#SBATCH --account=soldeg
#SBATCH --partition=gpu-h100s
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH -J mace-train
#SBATCH --mail-user="adgoga@ucdavis.edu"
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --output=mace_train_%j.log

module purge
module load cuda/11.8

# Activate your conda environment
source ~/.bashrc
conda activate mace

# Run your training script
python train_mace.py
