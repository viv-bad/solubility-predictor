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

        Reuturns:
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

        feature += cls._one_hot_encoding(atom.GetAtomicNum(), cls.ATOM_FEATURES['atomic_num'])
        feature += cls._one_hot_encoding(atom.GetDegree(), cls.ATOM_FEATURES['degree'])
        feature += cls._one_hot_encoding(atom.GetFormalCharge(), cls.ATOM_FEATURES['formal_charge'])
        feature += cls._one_hot_encoding(atom.GetChiralTag(), cls.ATOM_FEATURES['chiral_tag'])
        feature += cls._one_hot_encoding(atom.GetHybridization(), cls.ATOM_FEATURES['hybridization'])
        feature += cls._one_hot_encoding(atom.GetNumExplicitHs(), cls.ATOM_FEATURES['num_h'])
        feature += cls._one_hot_encoding(atom.GetIsAromatic(), cls.ATOM_FEATURES['is_aromatic'])

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

        feature += cls._one_hot_encoding(bond.GetBondType(), cls.BOND_FEATURES['bond_type'])
        feature += cls._one_hot_encoding(bond.GetIsConjugated(), cls.BOND_FEATURES['is_conjugated'])
        feature += cls._one_hot_encoding(bond.IsInRing(), cls.BOND_FEATURES['is_in_ring'])
        feature += cls._one_hot_encoding(bond.GetStereo(), cls.BOND_FEATURES['stereo'])

        return features
    
    @classmethod
    def smiles_to_graph(cls, smiles):
        """
        Convert a SMILES string to a PyTorch Geometric graph.

        Args: 
            smiles: SMILES string of the molecule

        Returns:
            PyTorch Geometric Data object
        """

        # Parse SMILES string with RDKit
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            return None
        
        # get atom features
        atom_features = []
        for atom in mol.GetAtoms():
            atom_features.append(cls._atom_features(atom))

        # convert to tensor
        x = torch.tensor(atom_features, dtype=torch.float)

        #get edge indices and features
        edge_indices = []
        edge_attrs = []

        for bond in mol.GetBonds():

            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            # add bidirectional edges
            edge_indices.append([i, j])
            edge_indices.append([j, i])

            # add bond features
            bond_features = cls._bond_features(bond)
            edge_attrs.append(bond_features)
            edge_attrs.append(bond_features)

            
        # if no bonds, make dummy/empty edge tensors
        if len(edge_indices) == 0: 
            edge_index = torch.zeros((2,0), dtype=torch.long)
            edge_attr = torch.zeros((0, len(cls._bond_features(None))), dtype=torch.float)
        else:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        
        # make PyTorch Geometric Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            smiles=smiles
        )

        return data


        

