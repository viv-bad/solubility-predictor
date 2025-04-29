import os
import pandas as pd
import torch
from torch_geometric.data import Dataset, InMemoryDataset
from .molecule_graph import MoleculeGraph

class SolubilityDataset(InMemoryDataset):
    """
    PyTorch Geometric dataset for molecular solubility prediction.
    """

    def __init__(self, root, csv_file, transform=None, pre_transform=None, pre_filter=None):
        """
        Args:
            root (str): Root directory where the dataset should be saved.
            csv_file (str): Path to the CSV file containing the SMILES and solubility data.
            transform (callable, optional): Optional transform to be applied to each graph.
            pre_transform (callable, optional): Optional transform to be applied to each graph before saving.
            pre_filter (callable, optional): Optional filter to be applied to each graph before saving.
        """

        self.csv_file = csv_file
        super(SolubilityDataset, self).__init__(root, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])

    
    @property
    def raw_file_names(self):
        """
        Returns a list of raw file names in the dataset.
        """
        return [os.path.basename(self.csv_file)]
    
    @property
    def processed_file_names(self):
        """
        Returns a list of processed file names in the dataset.
        """
        return ['solubility_data.pt']
    
    def download(self):
        """
        Download the dataset if it doesn't exist in the raw directory.
        """
        # TODO: Implement download from repo
        pass
    
    def process(self):
        """
        Process the raw data and save it to the processed directory.
        """
        # read the CSV file
        df = pd.read_csv(self.raw_paths[0])
        
        # Convert SMILES to graphs
        data_list = []
        for idx, row in df.iterrows():
            # get smiles and solubility
            smiles = row['smiles']
            solubility = float(row['solubility'])

            # convert to graph
            graph = MoleculeGraph.smiles_to_graph(smiles)

            if graph is None:
                continue

            # add solubility as target
            graph.y = torch.tensor([solubility], dtype=torch.float)

            graph.id = row['id']

            # apply pre-filter
            if self.pre_filter is not None and not self.pre_filter(graph):
                continue
            if self.pre_transform is not None:
                graph = self.pre_transform(graph)

            data_list.append(graph)

        data, slices = self.collate(data_list) # slices for separate graph indexing after collation
        torch.save((data, slices), self.processed_paths[0])
        
    
    