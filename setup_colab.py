# file: setup_colab.py
"""
Setup script to fix imports and directory structure for running in Google Colab.
Run this script after cloning the repository but before running train.py.
"""

import os
import sys
import shutil
import subprocess

print("Starting Colab setup...")

# Install required packages
print("Installing torch-geometric and rdkit...")
subprocess.run(["pip", "install", "torch-geometric", "rdkit", "-q"], check=True)
print("Done.")

# Get PyTorch version and determine correct CUDA version for PyG
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

cuda_suffix = ""
if torch.cuda.is_available():
    try:
        cuda_version = torch.version.cuda
        # Ensure cuda_version is not None before replacing '.'
        if cuda_version:
            cuda_suffix = f"cu{cuda_version.replace('.', '')}"
            print(f"CUDA version: {cuda_version}")
            print(f"GPU device: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA version detection failed, defaulting to CPU.")
            cuda_suffix = "cpu"
    except Exception as e:
        print(f"Error detecting CUDA version: {e}. Defaulting to CPU.")
        cuda_suffix = "cpu"
else:
    print("CUDA not available, using CPU.")
    cuda_suffix = "cpu"

# Install PyG scatter and sparse dependencies
# print(f"Installing PyG dependencies for torch {torch.__version__} + {cuda_suffix}...")
# pyg_whl_url = f"https://data.pyg.org/whl/torch-{torch.__version__}+{cuda_suffix}.html"
# try:
#     subprocess.run(["pip", "install", "torch-scatter", "torch-sparse", "-f", pyg_whl_url, "-q"], check=True)
#     print("PyG dependencies installed successfully.")
# except subprocess.CalledProcessError as e:
#     print(f"Error installing PyG dependencies: {e}")
#     print("Please check the URL and compatibility: ", pyg_whl_url)
#     print("Attempting install without specific version link (might be slower or fail)...")
#     try:
#         subprocess.run(["pip", "install", "torch-scatter", "torch-sparse", "-q"], check=True)
#         print("Installed torch-scatter and torch-sparse without specific wheel URL.")
#     except subprocess.CalledProcessError as e2:
#         print(f"Failed to install torch-scatter/torch-sparse: {e2}. Training might fail.")

# Check project root (simple check)
project_root = "."
if not os.path.exists(os.path.join(project_root, "src")):
     print(f"Warning: 'src' directory not found in the current directory ({os.getcwd()}). Make sure you run this from the repository root.")

# Make sure required directories exist
print("Ensuring required directories exist...")
os.makedirs(os.path.join(project_root, "data/raw"), exist_ok=True)
os.makedirs(os.path.join(project_root, "data/processed"), exist_ok=True)
os.makedirs(os.path.join(project_root, "models"), exist_ok=True)
print("Done.")

# Create/update __init__.py files to enable imports (Good Practice)
print("Creating __init__.py files...")
init_files = [
    os.path.join(project_root, "src/__init__.py"),
    os.path.join(project_root, "src/data/__init__.py"),
    # Note: You don't have a src/models directory in your structure, the models dir is at the root
    # os.path.join(project_root, "src/models/__init__.py") # Remove or adapt if structure changes
]
for init_file in init_files:
    if not os.path.exists(init_file):
        with open(init_file, "w") as f:
            f.write(f"# Automatically created by setup_colab.py to enable package imports")
print("Done.")

# Create a fix_imports.py file in the root directory to modify Python's import system
# This is the key part for making imports work without modifying source files
print("Creating fix_imports.py...")
fix_imports_path = os.path.join(project_root, "fix_imports.py")
with open(fix_imports_path, "w") as f:
    f.write("""
# fix_imports.py
import sys
import os

# Add the project root directory to the Python path
project_root = os.path.abspath('.')
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Added project root to sys.path: {project_root}")

# Optionally, add the 'src' directory itself if needed, though adding the root is usually sufficient
# src_path = os.path.abspath('./src')
# if src_path not in sys.path:
#    sys.path.insert(0, src_path)
#    print(f"Added src directory to sys.path: {src_path}")

# You can verify the path includes your project root now
# print("\\nCurrent sys.path:")
# for p in sys.path:
#     print(p)
# print("-" * 20)
    """)
print(f"Created {fix_imports_path}")

# Create a sample solubility dataset if needed
sample_data_path = os.path.join(project_root, "data/raw/solubility_dataset.csv")
if not os.path.exists(sample_data_path):
    print(f"Creating sample solubility dataset at {sample_data_path}...")
    with open(sample_data_path, "w") as f:
        f.write("""id,name,smiles,solubility
1,Glucose,C(C1C(C(C(C(O1)O)O)O)O)O,0.8
2,Caffeine,CN1C=NC2=C1C(=O)N(C(=O)N2C)C,-2.2
3,Aspirin,CC(=O)OC1=CC=CC=C1C(=O)O,-4.5
4,Ibuprofen,CC(C)CC1=CC=C(C=C1)C(C)C(=O)O,-7.5
5,Cholesterol,CC(C)CCCC(C)C1CCC2C1(CCC3C2CC=C4C3(CCC(C4)O)C)C,-12.5
6,Paracetamol,CC(=O)NC1=CC=C(C=C1)O,-1.5
7,Ampicillin,CC1(C(N2C(S1)C(C2=O)NC(=O)C(C3=CC=CC=C3)N)C(=O)O)C,-1.0
8,Benzene,C1=CC=CC=C1,-2.0
9,Toluene,CC1=CC=CC=C1,-2.7
10,Naphthalene,C1=CC2=CC=CC=C2C=C1,-3.6""")
    print("Done.")
else:
    print(f"Dataset already exists at {sample_data_path}")

# --- REMOVED THE SECTIONS THAT MODIFY train.py and dataset.py ---
# The original imports in those files should now work when run
# via the run_train.py wrapper or after importing fix_imports.

# Create a wrapper script for train.py that includes the import fixes
print("Creating run_train.py wrapper...")
run_train_path = os.path.join(project_root, "run_train.py")
with open(run_train_path, "w") as f:
    f.write(f"""
# run_train.py
# This script ensures the Python path is set correctly before running the main training code.

print("Importing fix_imports to set up sys.path...")
import fix_imports # This executes the code in fix_imports.py

print("Importing training function...")
# Now that sys.path is fixed, this import should work
from src.train import train_model

if __name__ == "__main__":
    print("Starting training process...")
    # Call the train_model function with appropriate parameters
    # Ensure data paths are relative to the project root
    data_root_path = "{project_root}/data"
    model = train_model(data_root_path, epochs=50) # Adjust epochs etc. as needed
    print("Training finished.")
    """)
print(f"Created {run_train_path}")

print("\nSetup complete!")
print(f"\nTo run the training script, use: !python {run_train_path}")
print(f"To test molecule visualization, use: !python -c \"import fix_imports; from src.data.test_graph import visualize_molecule_graph; visualize_molecule_graph('CC(=O)OC1=CC=CC=C1C(=O)O')\"")
print(f"To run data exploration, use: !python -c \"import fix_imports; from src.data.explore import explore_dataset; explore_dataset('{project_root}/data/raw/solubility_dataset.csv')\"")