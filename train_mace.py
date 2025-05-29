"""
train_mace.py
-------------

Script to train a MACE interatomic potential using the data in mace_training_data.xyz.
This script is configured for GPU training (CUDA) and is suitable for a single NVIDIA GPU (e.g., RTX 3060 Ti).

Author: Adam Goga
Date: 2025-05-29
"""

import os
import subprocess

# Path to training data
data_file = "mace_training_data_small.xyz"

# Output directory for model and logs
output_dir = "mace_model_output"
os.makedirs(output_dir, exist_ok=True)

# MACE training command (basic example)
# You can adjust parameters as needed for your system and dataset size
command = [
    "mace-train",
    "--config-type", "default",
    "--train-file", data_file,
    "--valid-file", data_file,  # For demonstration, use the same file for validation
    "--model-type", "MACE",
    "--energy-weight", "1.0",
    "--force-weight", "100.0",
    "--num-epochs", "50",
    "--batch-size", "4",
    "--device", "cuda",
    "--output-dir", output_dir,
    "--seed", "42"
]

print("Running MACE training on GPU...")
print("Command:", " ".join(command))

subprocess.run(command, check=True)

print(f"Training complete. Model and logs are in {output_dir}/")
