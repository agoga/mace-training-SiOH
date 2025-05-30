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

# Training parameters
train_params = {
    "model_type": "MACE",
    "energy_weight": 1.0,
    "force_weight": 100.0,
    "num_epochs": 50,
    "batch_size": 1,  # lowered from 4
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "output_dir": output_dir,
    "seed": 42,
    "num_workers": 0,
    "pin_memory": False,
    "valid_batch_size": 1,
}

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()

print(f"Training MACE model on {len(structures)} structures using {train_params['device']}...")

command = [
    "mace_run_train",
    "--model", train_params["model_type"],
    "--train_file", train_file,
    "--valid_file", train_file,
    "--energy_weight", str(train_params["energy_weight"]),
    "--forces_weight", str(train_params["force_weight"]),
    "--max_num_epochs", str(train_params["num_epochs"]),
    "--batch_size", str(train_params["batch_size"]),
    "--device", train_params["device"],
    "--work_dir", output_dir,
    "--name", "SiOH-test",
    "--E0s", "{1: -13.6, 8: -204.0, 14: -290.0}",
    "--num_workers", str(train_params["num_workers"]),
    "--pin_memory", str(train_params["pin_memory"]),
    "--valid_batch_size", str(train_params["valid_batch_size"]),
    "--num_channels", "128",            # reduced from 256
    "--num_interactions", "4",          # reduced from 6
    "--max_L", "2",
    "--r_max", "6.0",
    "--lr", "0.001",
    "--ema_decay", "0.99",
    "--scheduler", "ReduceLROnPlateau",
    "--seed", str(train_params["seed"])
]

subprocess.run(command, check=True)

print(f"Training complete. Model and logs are in {output_dir}/")
