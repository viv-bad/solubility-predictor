# file: src/data/dataset.py
import os
import pandas as pd
import torch
from torch_geometric.data import Dataset, InMemoryDataset
# Ensure the relative import works because fix_imports adds the project root to sys.path
# and Python can find src.data.molecule_graph
from solpred.data.molecule_graph import MoleculeGraph 

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
        # Call super().__init__ BEFORE trying to load processed data
        super().__init__(root, transform, pre_transform, pre_filter) # Pass log=True if needed

        # Load the processed data, specifying weights_only=False
        try:
            self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False) # <--- ADD weights_only=False HERE
        except FileNotFoundError:
             # This case should ideally be handled by InMemoryDataset automatically triggering _process
             # but good to be aware of.
             print(f"Processed file not found at {self.processed_paths[0]}. Processing should trigger.")
        except Exception as e:
             print(f"Error loading processed file: {e}")
             # Optionally raise the error or handle it appropriately
             raise e


    @property
    def raw_file_names(self):
        """
        Returns a list of raw file names in the dataset. Assumes csv_file is in the raw_dir.
        """
        # Should return only the basename relative to raw_dir
        return [os.path.basename(self.csv_file)]

    @property
    def processed_file_names(self):
        """
        Returns a list of processed file names in the dataset.
        """
        # This name should match what you save in process()
        return ['solubility_data.pt']

    def download(self):
        """
        Download the dataset if it doesn't exist in the raw directory.
        Placeholder - implement if your raw data needs downloading.
        Checks if the specific csv_file exists in raw_paths.
        """
        raw_path = os.path.join(self.raw_dir, self.raw_file_names[0])
        if not os.path.exists(raw_path):
            print(f"Raw file {raw_path} not found. Please place it manually or implement download().")
            # Example: raise FileNotFoundError(f"Raw file not found: {raw_path}")
            # Or attempt download here:
            # download_url(url_to_your_data, self.raw_dir)
            pass # Currently does nothing if file isn't there

    def process(self):
        """
        Process the raw data and save it to the processed directory.
        """
        # Construct the full path to the raw CSV file
        raw_path = os.path.join(self.raw_dir, self.raw_file_names[0])
        print(f"Processing data from {raw_path}...")

        try:
            df = pd.read_csv(raw_path)
        except FileNotFoundError:
            print(f"Error: Raw data file not found at {raw_path}. Please ensure it exists.")
            # Trigger download check or raise error
            self.download() # Attempt download if implemented
            # Re-try reading after potential download
            try:
                df = pd.read_csv(raw_path)
            except FileNotFoundError:
                 raise FileNotFoundError(f"Raw data file still not found after download attempt: {raw_path}")

        data_list = []
        num_processed = 0
        num_skipped = 0
        for idx, row in df.iterrows():
            smiles = row['smiles']
            try:
                solubility = float(row['solubility'])
            except (ValueError, TypeError):
                 print(f"Warning: Invalid solubility value '{row['solubility']}' for SMILES {smiles} at index {idx}. Skipping.")
                 num_skipped += 1
                 continue

            # Convert SMILES to graph
            graph = MoleculeGraph.smiles_to_graph(smiles)

            if graph is None:
                # smiles_to_graph should print a warning internally if it returns None
                num_skipped += 1
                continue

            # Add solubility as target
            graph.y = torch.tensor([[solubility]], dtype=torch.float)
            # Add optional ID if needed later
            # graph.id = row['id'] # Uncomment if ID is needed

            # Apply pre-filter and pre-transform if provided
            if self.pre_filter is not None and not self.pre_filter(graph):
                num_skipped += 1
                continue
            if self.pre_transform is not None:
                graph = self.pre_transform(graph)

            data_list.append(graph)
            num_processed += 1

        if not data_list:
             raise ValueError("No valid molecules were processed from the input file. Check data and SMILES parsing.")

        print(f"Successfully processed {num_processed} molecules, skipped {num_skipped}.")

        # Collate and save
        data, slices = self.collate(data_list)
        save_path = self.processed_paths[0]
        print(f"Saving processed data to {save_path}...")
        torch.save((data, slices), save_path)
        print("Processing complete.")