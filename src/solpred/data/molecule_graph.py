import torch
from torch_geometric.data import Data
from rdkit import Chem
import numpy as np

class MoleculeGraph:
    """
    Class for converting molecules to graph representations.
    """
    # Atom features for node representation
    ATOM_FEATURES = {
        'atomic_num': list(range(1, 119)), 
        'degree': [0,1,2,3,4,5,6,7,8,9,10], # number of bonds
        'formal_charge': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
        'chiral_tag': [0,1,2,3],
        'hybridization': [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D, Chem.rdchem.HybridizationType.SP3D2],
        'num_h': [0,1,2,3,4,5,6,7,8], # number of hydrogens
        'is_aromatic': [0,1]   
    }

    # Bond features for edge representation
    BOND_FEATURES = {
        'bond_type': [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC],
        'is_conjugated': [0,1],
        'is_in_ring': [0,1],
        'stereo': [
            Chem.rdchem.BondStereo.STEREONONE,
            Chem.rdchem.BondStereo.STEREOANY,
            Chem.rdchem.BondStereo.STEREOZ,
            Chem.rdchem.BondStereo.STEREOE,
        ]
    }

    @staticmethod
    def _one_hot_encoding(val, feature_dict):
        """
        One-hot encode a feature value based on the feature dictionary.

        Args:
            val: The feature value to encode
            feature_dict: Dictionary mapping feature names to possible values

        Returns:
            One-hot encoded feature vector
        """

        if val not in feature_dict:
            return [0] * len(feature_dict)
        return [1 if val == v else 0 for v in feature_dict]
    
    @classmethod
    def _atom_features(cls, atom):
        """
        Extract atom features and encode them as a vector.

        Args:
            atom: RDKit atom object
        
        Returns:
            Feature vector for the atom
        """

        features = []

        features += cls._one_hot_encoding(atom.GetAtomicNum(), cls.ATOM_FEATURES['atomic_num'])
        features += cls._one_hot_encoding(atom.GetDegree(), cls.ATOM_FEATURES['degree'])
        features += cls._one_hot_encoding(atom.GetFormalCharge(), cls.ATOM_FEATURES['formal_charge'])
        features += cls._one_hot_encoding(atom.GetChiralTag(), cls.ATOM_FEATURES['chiral_tag'])
        features += cls._one_hot_encoding(atom.GetHybridization(), cls.ATOM_FEATURES['hybridization'])
        features += cls._one_hot_encoding(atom.GetNumExplicitHs(), cls.ATOM_FEATURES['num_h'])
        features += cls._one_hot_encoding(atom.GetIsAromatic(), cls.ATOM_FEATURES['is_aromatic'])

        return features
    
    @classmethod
    def _bond_features(cls, bond):
        """
        Extracts bond features and encodes them as a vector.

        Args: 
            bond: RDKit bond object
        
        Returns:
            Feature vector for the bond
        """

        features = []

        features += cls._one_hot_encoding(bond.GetBondType(), cls.BOND_FEATURES['bond_type'])
        features += cls._one_hot_encoding(bond.GetIsConjugated(), cls.BOND_FEATURES['is_conjugated'])
        features += cls._one_hot_encoding(bond.IsInRing(), cls.BOND_FEATURES['is_in_ring'])
        features += cls._one_hot_encoding(bond.GetStereo(), cls.BOND_FEATURES['stereo'])

        return features
    
    @classmethod
    def smiles_to_graph(cls, smiles):
        """
        Convert a SMILES string to a PyTorch Geometric graph.

        Args:
            smiles: SMILES string of the molecule

        Returns:
            PyTorch Geometric Data object or None if parsing fails
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"Warning: RDKit failed to parse SMILES: {smiles}. Skipping molecule.")
                return None

            # Sanitize molecule (optional, but can help fix some valence issues)
            Chem.SanitizeMol(mol)

        except Exception as e:
            print(f"Error processing SMILES '{smiles}': {e}. Skipping molecule.")
            return None

        # get atom features
        atom_features = []
        for atom in mol.GetAtoms():
            atom_features.append(cls._atom_features(atom))
        x = torch.tensor(atom_features, dtype=torch.float)

        #get edge indices and features
        edge_indices = []
        edge_attrs = []

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_indices.append([i, j])
            edge_indices.append([j, i]) # Add bidirectional edges

            bond_features = cls._bond_features(bond) # This is now only called with valid bonds
            edge_attrs.append(bond_features)
            edge_attrs.append(bond_features)


        # if no bonds, make dummy/empty edge tensors
        if not edge_attrs: # Check if edge_attrs list is empty
            # Calculate bond feature dimension directly from the definition
            bond_feature_dims = sum(len(v) for v in cls.BOND_FEATURES.values())
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, bond_feature_dims), dtype=torch.float)
        else:
            # Convert lists to tensors
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

        # make PyTorch Geometric Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            smiles=smiles # Keep SMILES for reference if needed
        )

        return data


        

