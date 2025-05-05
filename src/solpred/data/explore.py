import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw
import matplotlib.pyplot as plt

def explore_dataset(file_path):
    """Explore and visualize the solubility dataset"""

    data = pd.read_csv(file_path)

    print(f"Dataset shape: {data.shape}")

    print("\nColumn types:")
    print(data.dtypes)
    print("\n Summary statistics for solubility:")
    print(data["solubility"].describe())

    valid_smiles = 0
    for smiles in data["smiles"]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            valid_smiles += 1
    
    print(f"\nValid SMILES strings: {valid_smiles}/{len(data)}")

    # Show example molecule
    if len(data) > 0:
        example_smiles = data["smiles"].iloc[0]
        example_mol = Chem.MolFromSmiles(example_smiles)
        if example_mol:
            print(f"\nExample molecule (ID: {data['id'].iloc[0]}, name: {data['name'].iloc[0]}):")
            print(f"SMILES: {example_smiles}")
            print(f"Solubility: {data['solubility'].iloc[0]}")

            # Calculate basic properties of example molecule
            print("\nBasic properties:")
            print(f"Molecular Weight: {Descriptors.MolWt(example_mol):.2f}")
            print(f"LogP: {Descriptors.MolLogP(example_mol):.2f}")
            print(f"Number of H-Bond Donors: {Descriptors.NumHDonors(example_mol)}")
            print(f"Number of H-Bond Acceptors: {Descriptors.NumHAcceptors(example_mol)}")

            # # Visualize molecule
            Draw.MolToFile(example_mol, f"example_molecule.png")
            print(f"\nMolecule saved as 'example_molecule.png'")
            
    
    plt.figure(figsize=(10, 6))
    plt.hist(data["solubility"], bins=20, edgecolor='black')
    plt.title("Distribution of Solubility Values")
    plt.xlabel("Solubility")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.savefig("solubility_distribution.png")
    plt.show()

    return data
    
    




if __name__ == "__main__":
    data = explore_dataset("data/raw/solubility_dataset.csv")