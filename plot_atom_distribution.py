import os
from ase.io import read
import matplotlib.pyplot as plt

train_file = "mace_training_data.xyz"
output_dir = "mace_model_output"
os.makedirs(output_dir, exist_ok=True)

# Read training data with progress output
print("Reading structures from", train_file)
structures = []
with open(train_file) as f:
    lines = f.readlines()

n_structures = 0
idx = 0
N = len(lines)
while idx < N:
    try:
        natoms = int(lines[idx])
    except Exception:
        break
    n_structures += 1
    if n_structures % 100 == 0:
        print(f"Read {n_structures} structures so far...")
    idx += natoms + 2
print(f"Total structures read: {n_structures}")

# Now actually load them with ase.io.read
structures = read(train_file, ":")

# Plot and save histogram of atom counts
num_atoms = [len(a) for a in structures]
plt.figure(figsize=(8, 5))
plt.hist(num_atoms, bins=20, edgecolor='black')
plt.xlabel('Number of atoms per structure')
plt.ylabel('Count')
plt.title('Distribution of number of atoms in training data')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "atom_count_hist.png"))
plt.close()
print(f"Saved atom count histogram to {os.path.join(output_dir, 'atom_count_hist.png')}")
