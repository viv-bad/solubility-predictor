import os 
import torch
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, AllChem
import seaborn as sns
from torch_geometric.loader import DataLoader

from solpred.data.dataset import SolubilityDataset
from solpred.models.gnn_model import SolubilityGNN
from solpred.data.molecule_graph import MoleculeGraph
from inference import SolubilityPredictor



def analyze_embedding_space(model_path, data_path, output_dir="analysis_results"):
    """
    Analyze the learned embedding space of the GNN model using t-SNE visualization. 

    Args:
        model_path: Path to the saved trained model file
        data_path: Path to the dataset file
        output_dir: Directory to save the analysis results
    """

    # make output dirs if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # load dataset
    dataset = SolubilityDataset(root=data_path, csv_file=os.path.join(data_path, 'raw/solubility_data.csv'))

    # init predictor
    predictor = SolubilityPredictor(model_path)
    model = predictor.model
    device = predictor.device

    # create data loader
    data_loader = DataLoader(dataset, batch_size=64, shuffle=False)

    # extract embeddings and solubility values
    embeddings = []
    solubility_values = []
    molecule_ids = []

    model.eval()
    with torch.inference_mode():
        for batch in data_loader:
            batch = batch.to(device)

            #get node embeddings
            x = model.node_embedding(batch.x)

            # apply GNN layers to get node representations
            for gnn_layer in model.gnn_layers:
                x = gnn_layer(x, batch.edge_index, batch.edge_attr)
                x = torch.relu(x)

            # pool to get graph level embeddings
            from torch_geometric.nn import global_mean_pool
            graph_embedding = global_mean_pool(x, batch.batch)

            #store embeddings and solubility values
            embeddings.append(graph_embedding.cpu().numpy())
            solubility_values.append(batch.y.cpu().numpy())
            
            # Check if id exists as an attribute, otherwise use index in dataset
            if hasattr(batch, 'id'):
                molecule_ids.extend([id.item() if hasattr(id, 'item') else id for id in batch.id])
            else:
                # Use smiles as fallback identifier
                molecule_ids.extend(batch.smiles)

    # concatenate embeddings and solubility values
    embeddings = np.vstack(embeddings)
    solubility_values = np.vstack(solubility_values).flatten()

    # apply t-SNE to reduce dimensionality
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings)

    df = pd.DataFrame({
        'tsne_1': embeddings_2d[:, 0],
        'tsne_2': embeddings_2d[:, 1],
        'solubility': solubility_values,
        'molecule_id': molecule_ids
    })

   # Visualize t-SNE plot colored by solubility
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(
        df['tsne_1'], df['tsne_2'], 
        c=df['solubility'], 
        cmap='viridis', 
        alpha=0.7, 
        s=50
    )
    plt.colorbar(scatter, label='Solubility')
    plt.title('t-SNE Visualization of Molecular Embeddings Colored by Solubility', fontsize=14)
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tsne_visualization.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"t-SNE visualization saved to {os.path.join(output_dir, 'tsne_visualization.png')}")
    
    # Save embeddings for further analysis
    df.to_csv(os.path.join(output_dir, 'molecule_embeddings.csv'), index=False)
    print(f"Embeddings saved to {os.path.join(output_dir, 'molecule_embeddings.csv')}")
    
    return df

def analyze_error_patterns(model_path, data_path, output_dir="analysis_results"):
    """
    Analyze error patterns in the model's predictions.

    Args:
        model_path: Path to the saved trained model file
        data_path: Path to the dataset file
        output_dir: Directory to save the analysis results
    
    """

    # create output dirs if doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # load dataset
    dataset = SolubilityDataset(root=data_path, csv_file=os.path.join(data_path, 'raw/solubility_data.csv'))

    # init predictor
    predictor = SolubilityPredictor(model_path)

    # create data loader
    data_loader = DataLoader(dataset, batch_size=64, shuffle=False)

    # make predictions and calculate errors
    all_smiles = []
    actual_values = []
    predicted_values = []
    molecule_ids = []

    for batch in data_loader:
        # get smiles strings
        all_smiles.extend(batch.smiles)

        # get actual solubility values
        actual_values.extend(batch.y.cpu().numpy().flatten())

        # Get molecule IDs if available, otherwise use smiles
        if hasattr(batch, 'id'):
            molecule_ids.extend([id.item() if hasattr(id, 'item') else id for id in batch.id])
        else:
            molecule_ids.extend(batch.smiles)

        # make preds
        batch_predictions = []
        for smiles in batch.smiles:
            # Set lookup_name to False to avoid PubChem API calls during analysis
            result = predictor.predict_from_smiles(smiles, lookup_name=False)
            batch_predictions.append(result['predicted_solubility'])
        
        predicted_values.extend(batch_predictions)

    # calculate errors
    errors = np.array(predicted_values) - np.array(actual_values)

    # make dataframe for analysis
    df = pd.DataFrame({
        'id': molecule_ids,
        'smiles': all_smiles,
        'actual_solubility': actual_values,
        'predicted_solubility': predicted_values,
        'error': errors,
        'abs_error': np.abs(errors)
    })

    # sort by absolute error to find worst predictions
    df_sorted = df.sort_values('abs_error', ascending=False)

    # save error analysis to csv
    df_sorted.to_csv(os.path.join(output_dir, 'error_analysis.csv'), index=False)
    print(f"Error analysis saved to {os.path.join(output_dir, 'error_analysis.csv')}")

    # calculate molecular descriptors to correlate with errors
    molecular_descriptors = []

    for smiles in all_smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            descriptors = {
                'MolWt': Descriptors.MolWt(mol),
                'LogP': Descriptors.MolLogP(mol),
                'NumHDonors': Descriptors.NumHDonors(mol),
                'NumHAcceptors': Descriptors.NumHAcceptors(mol),
                'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
                'NumAromaticRings': Descriptors.NumAromaticRings(mol),
                'NumHeavyAtoms': mol.GetNumHeavyAtoms(),
                'FractionCSP3': Descriptors.FractionCSP3(mol),
                'TPSA': Descriptors.TPSA(mol)
            }
            molecular_descriptors.append(descriptors)
        else:
            # If molecule is invalid, add NaN values
            descriptors = {
                'MolWt': float('nan'),
                'LogP': float('nan'),
                'NumHDonors': float('nan'),
                'NumHAcceptors': float('nan'),
                'NumRotatableBonds': float('nan'),
                'NumAromaticRings': float('nan'),
                'NumHeavyAtoms': float('nan'),
                'FractionCSP3': float('nan'),
                'TPSA': float('nan')
            }
            molecular_descriptors.append(descriptors)
    
    # add descriptors to dataframe
    descriptors_df = pd.DataFrame(molecular_descriptors)
    df_with_descriptors = pd.concat([df, descriptors_df], axis = 1)

    # Calculate correlation between errors and molecular descriptors
    correlation_df = df_with_descriptors[['abs_error', 'MolWt', 'LogP', 'NumHDonors', 
                                          'NumHAcceptors', 'NumRotatableBonds', 
                                          'NumAromaticRings', 'NumHeavyAtoms', 
                                          'FractionCSP3', 'TPSA']].corr()
    
    # Visualize correlation matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_df, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Between Prediction Errors and Molecular Descriptors', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_correlation.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Error correlation matrix saved to {os.path.join(output_dir, 'error_correlation.png')}")
    
    # Visualize top 5 highest and lowest error molecules
    visualize_extreme_molecules(df_sorted, output_dir)
    
    return df_with_descriptors

def visualize_extreme_molecules(df_sorted, output_dir):
    """
    Visualize molecules with highest and lowest prediction errors.
    
    Args:
        df_sorted: DataFrame sorted by absolute error
        output_dir: Directory to save visualizations
    """
    # Create directory for molecule images
    molecules_dir = os.path.join(output_dir, 'extreme_molecules')
    os.makedirs(molecules_dir, exist_ok=True)
    
    # Get top 5 highest error molecules
    highest_error = df_sorted.head(5)
    
    # Get top 5 lowest error molecules
    lowest_error = df_sorted.tail(5)
    
    # Visualize highest error molecules
    plt.figure(figsize=(15, 10))
    for i, (_, row) in enumerate(highest_error.iterrows()):
        mol = Chem.MolFromSmiles(row['smiles'])
        if mol is not None:
            mol = Chem.Mol(mol)
            mol.RemoveAllConformers()
            AllChem.Compute2DCoords(mol)
            img = Draw.MolToImage(mol, size=(300, 300))
            
            plt.subplot(2, 5, i+1)
            plt.imshow(img)
            plt.title(f"Error: {row['error']:.2f}\nActual: {row['actual_solubility']:.2f}\nPred: {row['predicted_solubility']:.2f}")
            plt.axis('off')
    
    # Visualize lowest error molecules
    for i, (_, row) in enumerate(lowest_error.iterrows()):
        mol = Chem.MolFromSmiles(row['smiles'])
        if mol is not None:
            mol = Chem.Mol(mol)
            mol.RemoveAllConformers()
            AllChem.Compute2DCoords(mol)
            img = Draw.MolToImage(mol, size=(300, 300))
            
            plt.subplot(2, 5, i+6)
            plt.imshow(img)
            plt.title(f"Error: {row['error']:.2f}\nActual: {row['actual_solubility']:.2f}\nPred: {row['predicted_solubility']:.2f}")
            plt.axis('off')
    
    plt.suptitle('Top 5 Highest vs Lowest Error Molecules', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(os.path.join(output_dir, 'extreme_error_molecules.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Extreme error molecules visualization saved to {os.path.join(output_dir, 'extreme_error_molecules.png')}")

def analyze_node_importance(model_path, data_path, output_dir="analysis_results"):
    """
    Analyze the importance of different atom types and functional groups.
    
    Args:
        model_path: Path to the saved model checkpoint
        data_path: Path to the dataset
        output_dir: Directory to save analysis results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    dataset = SolubilityDataset(root=data_path, csv_file=os.path.join(data_path, 'raw/solubility_data.csv'))
    
    # Initialize predictor and model
    predictor = SolubilityPredictor(model_path)
    model = predictor.model
    device = predictor.device
    
    # Create a list of atom types to analyze
    atom_types = [
        ('Carbon', 'C', 6),
        ('Nitrogen', 'N', 7),
        ('Oxygen', 'O', 8),
        ('Fluorine', 'F', 9),
        ('Chlorine', 'Cl', 17),
        ('Bromine', 'Br', 35),
        ('Sulfur', 'S', 16),
        ('Phosphorus', 'P', 15)
    ]
    
    # Analyze a sample of molecules
    sample_size = min(100, len(dataset))
    sample_indices = np.random.choice(len(dataset), sample_size, replace=False)
    sample_dataset = dataset[sample_indices]
    
    atom_importance_data = []
    
    for data in sample_dataset:
        # Get SMILES string
        smiles = data.smiles
        
        # Convert to molecule
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        
        # Get original prediction (with lookup_name=False to avoid PubChem API calls)
        original_result = predictor.predict_from_smiles(smiles, lookup_name=False)
        original_prediction = original_result['predicted_solubility']
        
        # Analyze each atom type
        for atom_name, atom_symbol, atomic_num in atom_types:
            # Find atoms of this type
            atom_indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == atomic_num]
            
            if not atom_indices:
                continue
            
            # Create modified molecules by replacing atoms one by one
            for atom_idx in atom_indices:
                # We'll try to replace with silicon as a dummy substitution (element 14)
                # This is just for demonstration - in practice, more sophisticated perturbations would be used
                modified_mol = Chem.RWMol(mol)
                modified_atom = modified_mol.GetAtomWithIdx(atom_idx)
                original_type = modified_atom.GetAtomicNum()
                
                # Replace with silicon
                modified_atom.SetAtomicNum(14)  # Silicon
                modified_smiles = Chem.MolToSmiles(modified_mol)
                
                # Check if the modification resulted in a valid molecule
                check_mol = Chem.MolFromSmiles(modified_smiles)
                if check_mol is None:
                    continue
                
                # Get prediction for modified molecule (with lookup_name=False)
                try:
                    modified_result = predictor.predict_from_smiles(modified_smiles, lookup_name=False)
                    modified_prediction = modified_result['predicted_solubility']
                    
                    # Calculate importance as absolute change in prediction
                    importance = abs(original_prediction - modified_prediction)
                    
                    # Store results
                    atom_importance_data.append({
                        'smiles': smiles,
                        'atom_type': atom_name,
                        'atom_idx': atom_idx,
                        'original_prediction': original_prediction,
                        'modified_prediction': modified_prediction,
                        'importance': importance
                    })
                except Exception as e:
                    print(f"Error processing modified molecule: {e}")
    
    # Convert to DataFrame
    if atom_importance_data:
        importance_df = pd.DataFrame(atom_importance_data)
        
        # Calculate average importance by atom type
        avg_importance = importance_df.groupby('atom_type')['importance'].mean().reset_index()
        avg_importance = avg_importance.sort_values('importance', ascending=False)
        
        # Visualize average importance by atom type
        plt.figure(figsize=(10, 6))
        sns.barplot(x='atom_type', y='importance', data=avg_importance)
        plt.title('Average Importance of Atom Types for Solubility Prediction', fontsize=14)
        plt.xlabel('Atom Type', fontsize=12)
        plt.ylabel('Importance (Change in Prediction)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'atom_importance.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Atom importance analysis saved to {os.path.join(output_dir, 'atom_importance.png')}")
        
        # Save importance data
        importance_df.to_csv(os.path.join(output_dir, 'atom_importance.csv'), index=False)
        print(f"Atom importance data saved to {os.path.join(output_dir, 'atom_importance.csv')}")
        
        return importance_df
    else:
        print("No valid atom importance data collected.")
        return None

    
def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description="Analyze a trained GNN model for solubility prediction")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model file")
    parser.add_argument("--data", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--output", type=str, default="analysis_results", help="Path to save analysis results")
    parser.add_argument("--embeddings", action="store_true", help="Analyze embedding space")
    parser.add_argument("--errors", action="store_true", help="Analyze error patterns")
    parser.add_argument("--importance", action="store_true", help="Analyze node importance")
    parser.add_argument("--all", action="store_true", help="Run all analyses")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    if args.all or args.embeddings:
        print("\n=== Analyzing Embedding Space ===")
        analyze_embedding_space(args.model, args.data, args.output)
    
    if args.all or args.errors:
        print("\n=== Analyzing Error Patterns ===")
        analyze_error_patterns(args.model, args.data, args.output)
    
    if args.all or args.importance:
        print("\n=== Analyzing Node Importance ===")
        analyze_node_importance(args.model, args.data, args.output)
    
    print("\nAnalysis complete. Results saved to", args.output)

if __name__ == "__main__":
    main()