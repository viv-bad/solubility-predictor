import torch
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
import matplotlib.pyplot as plt  # Import pyplot as plt
import numpy as np
from molecule_graph import MoleculeGraph

def visualize_molecule_graph(smiles):
    """
    Visualize a molecule and its graph representation.
    
    Args:
        smiles: SMILES string of the molecule
    """
    # Parse SMILES string with RDKit
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Failed to parse SMILES: {smiles}")
        return
    
    # Convert to graph
    graph = MoleculeGraph.smiles_to_graph(smiles)
    
    # Print graph info
    print(f"SMILES: {smiles}")
    print(f"Number of atoms (nodes): {graph.x.shape[0]}")
    print(f"Number of bonds (edges): {graph.edge_index.shape[1] // 2}")  # Divide by 2 because edges are bidirectional
    print(f"Node feature dimension: {graph.x.shape[1]}")
    print(f"Edge feature dimension: {graph.edge_attr.shape[1]}")
    
    # Generate 2D coordinates for the molecule
    mol = Chem.Mol(mol)  # Make a copy of the molecule
    mol.RemoveAllConformers()  # Remove any existing conformers
    AllChem.Compute2DCoords(mol)  # Compute 2D coordinates
    
    # Draw molecule
    img = Draw.MolToImage(mol, size=(400, 300))
    
    # Create a simple visualization of the graph
    plt.figure(figsize=(12, 6))
    
    # Subplot for the molecule
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("RDKit Molecule Representation")
    plt.axis('off')
    
    # Subplot for the graph
    plt.subplot(1, 2, 2)
    
    # Get 2D coordinates from the generated conformer
    pos = {}
    conformer = mol.GetConformer()
    for i, atom in enumerate(mol.GetAtoms()):
        position = conformer.GetAtomPosition(i)
        pos[i] = (position.x, position.y)
    
    # Plot nodes
    for i in range(graph.x.shape[0]):
        plt.plot(pos[i][0], pos[i][1], 'o', markersize=10, color='skyblue')
        plt.text(pos[i][0], pos[i][1], f"{mol.GetAtomWithIdx(i).GetSymbol()}", 
                 ha='center', va='center')
    
    # Plot edges
    for i in range(0, graph.edge_index.shape[1], 2):  # Step by 2 to avoid duplicate edges
        start_idx = graph.edge_index[0, i].item()
        end_idx = graph.edge_index[1, i].item()
        plt.plot([pos[start_idx][0], pos[end_idx][0]], 
                 [pos[start_idx][1], pos[end_idx][1]], '-', color='gray')
    
    plt.title("Graph Representation")
    plt.axis('equal')
    plt.axis('off')
    
    plt.tight_layout()
    molecule_name = mol.GetProp("_Name") if mol.HasProp("_Name") else "mol"
    # plt.savefig(f'molecule_graph_{molecule_name}.png')
    plt.show()
    print(f"Visualization saved as molecule_graph_{molecule_name}.png")
    # plt.close()

if __name__ == "__main__":
    # Test with examples from different solubility ranges
    test_smiles = [
        # High Solubility
        "C(C1C(C(C(C(O1)O)O)O)O)O",  # Glucose
        
        # Moderate-High Solubility
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        
        # Moderate Solubility
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
        
        # Low Solubility
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
        
        # Very Low Solubility
        "CC(C)CCCC(C)C1CCC2C1(CCC3C2CC=C4C3(CCC(C4)O)C)C"  # Cholesterol
    ]
    
    for smile in test_smiles:
        visualize_molecule_graph(smile)
        print("\n" + "-"*50 + "\n")