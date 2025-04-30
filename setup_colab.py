"""
Setup script to fix imports and directory structure for running in Google Colab.
Run this script after cloning the repository but before running train.py.
"""

import os
import sys
import shutil

# Install required packages
import subprocess
subprocess.run(["pip", "install", "torch-geometric", "-q"])
subprocess.run(["pip", "install", "rdkit", "-q"])

# Get PyTorch version and determine correct CUDA version for PyG
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

cuda_suffix = ""
if torch.cuda.is_available():
    cuda_version = torch.version.cuda
    cuda_suffix = f"cu{cuda_version.replace('.', '')}"
    print(f"CUDA version: {cuda_version}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
else:
    cuda_suffix = "cpu"

# Install PyG scatter and sparse dependencies
pyg_whl_url = f"https://data.pyg.org/whl/torch-{torch.__version__}+{cuda_suffix}.html"
subprocess.run(["pip", "install", "torch-scatter", "torch-sparse", "-f", pyg_whl_url])

# Check if we're in the project root or need to go up one level
if os.path.exists("src"):
    project_root = "."
else:
    project_root = ".."

# Make sure required directories exist
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Create/update __init__.py files to enable imports
if not os.path.exists("src/__init__.py"):
    with open("src/__init__.py", "w") as f:
        f.write("# Enable imports from src")
        
if not os.path.exists("src/data/__init__.py"):
    with open("src/data/__init__.py", "w") as f:
        f.write("# Enable imports from src.data")
        
if not os.path.exists("src/models/__init__.py"):
    with open("src/models/__init__.py", "w") as f:
        f.write("# Enable imports from src.models")

# Create a fix_imports.py file in the root directory to modify Python's import system
with open("fix_imports.py", "w") as f:
    f.write("""
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.abspath('.'))

# Add src directories to Python path
sys.path.insert(0, os.path.abspath('./src'))
sys.path.insert(0, os.path.abspath('./src/data'))
sys.path.insert(0, os.path.abspath('./src/models'))
    """)

# Create a sample solubility dataset if needed
if not os.path.exists("data/raw/solubility_dataset.csv"):
    print("Creating sample solubility dataset...")
    with open("data/raw/solubility_dataset.csv", "w") as f:
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

# Update train.py to fix import issues
with open("src/train.py", "r") as f:
    train_content = f.read()

train_content_fixed = train_content.replace("from src.data.dataset import SolubilityDataset", 
                                           "from dataset import SolubilityDataset")
train_content_fixed = train_content_fixed.replace("from src.models.gnn_model import SolubilityGNN", 
                                                "from gnn_model import SolubilityGNN")
train_content_fixed = train_content_fixed.replace("import matplotlib.pyplot as pl", 
                                                "import matplotlib.pyplot as plt")

with open("src/train.py", "w") as f:
    f.write(train_content_fixed)

# Update dataset.py to fix import issues
with open("src/data/dataset.py", "r") as f:
    dataset_content = f.read()

dataset_content_fixed = dataset_content.replace("from .molecule_graph import MoleculeGraph", 
                                              "from molecule_graph import MoleculeGraph")

with open("src/data/dataset.py", "w") as f:
    f.write(dataset_content_fixed)

# Create a wrapper script for train.py that includes the import fixes
with open("run_train.py", "w") as f:
    f.write("""
# Import the fix_imports module to set up Python path
import fix_imports

# Now run the training script
from src.train import train_model

if __name__ == "__main__":
    # Call the train_model function with appropriate parameters
    data_path = "data"
    train_model(data_path, epochs=50)
    """)

print("\nSetup complete!")
print("\nTo run the training script, use: !python run_train.py")
print("To test molecule visualization, use: !python src/data/test_graph.py")