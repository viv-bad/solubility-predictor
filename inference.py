import os
import torch
import argparse
import pandas as pandas
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, Descriptors
import matplotlib.pyplot as plt
import io
from PIL import Image
import pubchempy

from solpred.data.molecule_graph import MoleculeGraph
from solpred.models.gnn_model import SolubilityGNN

class SolubilityPredictor:
    """Class for making solubility predictions using a pre-trained GNN model."""

    def __init__(self, model_path, node_feature_dim=160, edge_feature_dim=12, hidden_dim=64, num_layers=3):
        """
        Initialize the predictor with a trained model.

        Args:
            model_path: Path to the trained model file
            node_feature_dim: Number of features for each node
            edge_feature_dim: Number of features for each edge
            hidden_dim: Number of hidden dimensions in the GNN
            num_layers: Number of GNN layers
        """

        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Initialize model
        self.model = SolubilityGNN(node_feature_dim=node_feature_dim, edge_feature_dim=edge_feature_dim, hidden_dim=hidden_dim, num_layers=num_layers).to(self.device)

        # Load saved weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        print(f"Model loaded from model_path")

    def predict_from_smiles(self, smiles):
        """
        Predict solubility from a SMILES string.

        Args:
            smiles: SMILES string of the molecule

        Returns:
            Dictionary containing the prediction and molecule information
        """

        # Check if smiles is valid
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"error": "Invalid SMILES string"}

        # Convert SMILES to molecule graph
        graph = MoleculeGraph.smiles_to_graph(smiles)
        if graph is None:
            return {"error": f"Failed to create molecule graph: {smiles}"}

        # Move graph to device
        x = graph.x.to(self.device)
        edge_index = graph.edge_index.to(self.device)
        edge_attr = graph.edge_attr.to(self.device)

        # Create batch information (single molecule, so all nodes belong to batch 0)
        batch = torch.zeros(x.size(0), dtype=torch.long).to(self.device)

        # Create data object with the same structure expected by the model
        data = type('', (), {})()
        data.x = x
        data.edge_index = edge_index
        data.edge_attr = edge_attr
        data.batch = batch

        # Perform prediction
        with torch.inference_mode():
            prediction = self.model(data).item()
        
        # get molecule properties
        mol_weight = Chem.Descriptors.MolWt(mol)
        logp = Chem.Descriptors.MolLogP(mol)
        num_atoms = mol.GetNumAtoms()

        # Interpret solubility
        solubility_level = self._interpret_solubility(prediction)

        compound_name = "N/A" # Default value
        try:
            # Attempt PubChem lookup
            compounds = pubchempy.get_compounds(smiles, namespace="smiles")
            if compounds:
                # Prioritize IUPAC name, fallback to first synonym if needed
                compound_name = compounds[0].iupac_name if compounds[0].iupac_name else compounds[0].synonyms[0] if compounds[0].synonyms else "Name Lookup Failed"
            else:
                compound_name = "Not Found in PubChem"
        except pubchempy.BadRequestError:
            # Handle cases where PubChem rejects the SMILES
            print(f"Warning: PubChem BadRequest for SMILES: {smiles}")
            compound_name = "PubChem Lookup Failed (Bad Request)"
        except Exception as e:
            # Handle other potential lookup errors (network issues, etc.)
            print(f"Warning: PubChem lookup failed for SMILES {smiles}: {e}")
            compound_name = "PubChem Lookup Failed (Error)"


        return {
            "smiles": smiles,
            # "compound_name": match.iupac_name, # <-- Replace this line
            "compound_name": compound_name,     # <-- With this line
            "predicted_solubility": prediction,
            "solubility_level": solubility_level,
            "mol_weight": mol_weight,
            "logp": logp,
            "num_atoms": num_atoms
        }

    def predict_batch(self, smiles_list):
        """
        Predict solubility for a batch of SMILES strings.

        Args:
            smiles_list: List of SMILES strings

        Returns:
            List of dictionaries containing the prediction and molecule information for each SMILES string
        """

        results = []
        for smiles in smiles_list:
            results.append(self.predict_from_smiles(smiles))
        return results

    def predict_from_csv(self, csv_path, smiles_col="smiles", output_path=None):
        """
        Predict solubility for all SMILES in a CSV file.

        Args:
            csv_path: Path to the CSV file
            smiles_col: Name of the column containing the SMILES strings
            output_path: Path to save the output CSV file (optional)
        
        Returns:
            DataFrame with original data and predictions
        """

        # Read CSV file
        df = pandas.read_csv(csv_path)

        if smiles_col not in df.columns:
            raise ValueError(f"Column '{smiles_col}' not found in CSV file")
        
        # Get smiles from the specified column
        smiles_list = df[smiles_col].tolist()
        
        results = []

        for smiles in smiles_list:
            results.append(self.predict_from_smiles(smiles))
        
        # Create output DataFrame
        results_df = pd.DataFrame(results)

        merged_df = pd.concat([df, results_df.drop(smiles_col, axis=1, errors='ignore')], axis=1)

        if output_path:
            merged_df.to_csv(output_path, index=False)
            print(f"Results saved to {output_path}")
        
        return merged_df
    
    def _interpret_solubility(self, solubility_value):
        """
        Interpret numerical solubility value into a category. 

        Args:
            solubility_value: Predicted numerical solubility value (log scale)
        
        Returns:
            String describing the solubility level
        """

        if solubility_value > 0:
            return "Very High Solubility"
        elif solubility_value > -2:
            return "High Solubility"
        elif solubility_value > -4:
            return "Moderate Solubility"
        elif solubility_value > -6:
            return "Low Solubility"
        else:
            return "Very Low Solubility"
    
    def visualize_molecule(self, smiles, show_prediction=True, save_path=None):
        """
        Generate a visualisation of the molecule with its predicted solubility.

        Args:
            smiles: SMILES string of the molecule
            show_prediction: Whether to display the predicted solubility
            save_path: Path to save the image file (optional)
        
        Returns:
            PIL Image object of the molecule
        """

        # Convert SMILES to molecule
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")

        # make prediction if requested
        title = ""
        if show_prediction:
            result = self.predict_from_smiles(smiles)
            title = f"Predicted Solubility: {result['predicted_solubility']:.2f} ({result['solubility_level']})"
        
        # Generate 2D coords for viz
        mol = Chem.Mol(mol)
        mol.RemoveAllConformers()
        AllChem.Compute2DCoords(mol)

        # draw molecule
        fig, ax = plt.subplots(figsize=(10, 6))
        img = Draw.MolToImage(mol, size=(400, 300))
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')

        # save to file if requested
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Molecule visualization saved to {save_path}")
        
        # convert to PIL image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        plt.close(fig)

        return img
        
def main():
    """Main function for command-line interface."""

    parser = argparse.ArgumentParser(description="Predict molecular solubility using a trained GNN model.")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model file")
    parser.add_argument("--smiles", type=str, help="SMILES string of the molecule to predict")
    parser.add_argument("--csv", type=str, help="Path to the CSV file containing SMILES strings")
    parser.add_argument("--smiles_col", type=str, default="smiles", help="Name of the column containing SMILES strings in a csv (default: smiles)")
    parser.add_argument("--output", type=str, help="Path to save the output CSV file")
    parser.add_argument("--visualize", action="store_true", help="Generate and visualize molecule")
    parser.add_argument("--viz_output", type=str, help="Path to save the visualization image")
    args = parser.parse_args()

    # Initialize predictor
    predictor = SolubilityPredictor(args.model)

    if args.smiles:
        result = predictor.predict_from_smiles(args.smiles)
        print("\nPrediction Result:")
        for key, value in result.items():
            print(f" {key}: {value}")
        
        if args.visualize:
            img = predictor.visualize_molecule(args.smiles, save_path=args.viz_output)
            if not args.viz_output:
                img.show()
        
    elif args.csv:
        results_df = predictor.predict_from_csv(args.csv, args.smiles_col, args.output)
        print(f"\nFirst few predictions:")
        print(results.df.head())
        print(f"Total predictions: {len(results_df)}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()