"""
train_mace.py (Python API version)
-------------------------------

Script to train a MACE interatomic potential using the data in mace_training_data_small.xyz.
This version uses the MACE Python API directly, avoiding the need for the 'mace-train' CLI.

Author: Adam Goga
Date: 2025-05-29
"""

import os
from ase.io import read
import subprocess
import torch
import gc

# Path to training data
train_file = "mace_training_data.xyz"
output_dir = "mace_model_output"
os.makedirs(output_dir, exist_ok=True)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Read training data
structures = read(train_file, ":")

# Training parameters (adjust as needed)
train_params = {
    "model_type": "MACE",
    "energy_weight": 1.0,
    "force_weight": 100.0,
    "num_epochs": 50,
    "batch_size": 4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "output_dir": output_dir,
    "seed": 42,
    "num_workers": 0,         # Disable multi-process data loading
    "pin_memory": False,      # Disable pinning memory
    "valid_batch_size": 1,    # Validation batch size = 1
}

# Clear GPU memory before training
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()

print(f"Training MACE model on {len(structures)} structures using {train_params['device']}...")

# Build the CLI command with memory-friendly settings
command = [
    "mace_run_train",
    "--model", "MACE",
    "--train_file", train_file,
    "--valid_file", train_file,
    "--energy_weight", "1.0",
    "--forces_weight", "100.0",
    "--max_num_epochs", "50",
    "--batch_size", "1",  # Lowered for memory safety
    "--device", "cuda",
    "--work_dir", output_dir,
    "--name", "SiOH-test",
    "--E0s", "{1: -13.6, 8: -204.0, 14: -290.0}",
    "--num_workers", "0",
    "--pin_memory", "False",
    "--valid_batch_size", "1",
    # Recommended Si/O/H hyperparameters:
    "--num_channels", "64",
    "--num_interactions", "2",
    "--max_L", "1",
    "--r_max", "5.0",
    "--lr", "0.001",
    "--ema_decay", "0.99",
    "--scheduler", "ReduceLROnPlateau",
    "--seed", "42"
]

subprocess.run(command, check=True)

print(f"Training complete. Model and logs are in {output_dir}/")
