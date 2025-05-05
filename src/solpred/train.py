import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import argparse # <-- Add argparse

# Assuming these imports are correct after restructuring
from solpred.data.dataset import SolubilityDataset
from solpred.models.gnn_model import SolubilityGNN

# Define a clear output directory at the project root level
DEFAULT_OUTPUT_DIR = "models" # Use the top-level models dir for artifacts

def train_model(data_path, output_dir=DEFAULT_OUTPUT_DIR, batch_size=32, hidden_dim=64, num_layers=3, lr=0.001, epochs = 100, seed=42):
    """
    Train a GNN model for solubility prediction.

    Args:
        data_path: Path to the directory containing raw/processed data folders.
        output_dir: Directory to save the trained model and plots.
        # ... other args
    """

    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Set device (GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Ensure the output directory exists ---
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving outputs to: {os.path.abspath(output_dir)}") # Log absolute path for clarity

    # Load dataset (use data_path for dataset loading)
    dataset = SolubilityDataset(root = data_path, csv_file = os.path.join(data_path, 'raw/solubility_data.csv')) # Or your specific CSV file name

    # split dataset into train, validation and test sets
    # ... (rest of split logic remains the same) ...
    train_idx, test_idx = train_test_split(
        np.arange(len(dataset)), test_size=0.2, random_state=seed
    )
    train_idx, val_idx = train_test_split(
        train_idx, test_size=0.25, random_state=seed
    )

    train_dataset = dataset[train_idx]
    val_dataset = dataset[val_idx]
    test_dataset = dataset[test_idx]

    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Test set size: {len(test_dataset)}")


    # Create data loaders
    # ... (loader logic remains the same) ...
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model and move to device
    # ... (model init remains the same) ...
    model = SolubilityGNN(
        node_feature_dim = train_dataset[0].x.shape[1],
        edge_feature_dim=train_dataset[0].edge_attr.shape[1],
        hidden_dim=hidden_dim,
        num_layers=num_layers
    ).to(device)

    # define optimizer and loss fns
    # ... (optimizer/loss remains the same) ...
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # Training loop
    train_losses, val_losses = [], []
    best_val_loss = float('inf')

    # --- Define best model path using output_dir ---
    best_model_path = os.path.join(output_dir, 'best_model.pth')

    for epoch in range(epochs):
        # ... (training steps remain the same) ...
        model.train()
        train_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch)
            loss = loss_fn(pred, batch.y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch.num_graphs
        train_loss /= len(train_dataset)
        train_losses.append(train_loss)

        # Validation
        # ... (validation steps remain the same) ...
        model.eval()
        val_loss = 0
        with torch.inference_mode():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch)
                loss = loss_fn(pred, batch.y)
                val_loss += loss.item() * batch.num_graphs
        val_loss /= len(val_dataset)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save best model (using the updated best_model_path)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved to {best_model_path}")

    # Plot training curve (saving to output_dir)
    plt.figure(figsize=(10,6))
    plt.plot(range(1, epochs+1), train_losses, label="Training Loss")
    plt.plot(range(1, epochs+1), val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (MSE)")
    plt.title("Training and Validation Loss")
    plt.legend()
    # --- Save plot using output_dir ---
    training_curve_path = os.path.join(output_dir, 'training_curve.png')
    plt.savefig(training_curve_path)
    print(f"Training curve saved to {training_curve_path}")
    plt.close() # Close the plot to free memory

    # Load best model for eval
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    # evaluate on test set
    # ... (evaluation logic remains the same) ...
    model.eval()
    test_preds, test_targets = [], []
    with torch.inference_mode():
        for batch in test_loader:
            batch = batch.to(device)
            pred = model(batch)
            test_preds.append(pred.cpu())
            test_targets.append(batch.y.cpu())
    test_preds = torch.cat(test_preds, dim = 0).numpy()
    test_targets = torch.cat(test_targets, dim=0).numpy()

    # Calculate metrics
    # ... (metric calculation remains the same) ...
    rmse = np.sqrt(mean_squared_error(test_targets, test_preds))
    r2 = r2_score(test_targets, test_preds)
    print(f"\nTest Results:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R2 Score: {r2:.4f}")

    # plot predictions vs actual (saving to output_dir)
    plt.figure(figsize=(8, 8)) # Make it square for better visualization
    plt.scatter(test_targets, test_preds, alpha=0.5, label="Predictions", s=20) # Smaller points can be clearer
    # Determine plot limits based on data range for a tighter fit
    min_val = min(test_targets.min(), test_preds.min()) * 0.95
    max_val = max(test_targets.max(), test_preds.max()) * 1.05
    plt.plot([min_val, max_val], [min_val, max_val], '--', color='red', label="Ideal Fit (y=x)")
    plt.xlabel("Actual Solubility")
    plt.ylabel("Predicted Solubility")
    plt.title("Test Set: Predicted vs Actual Solubility")
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box') # Ensure square aspect ratio
    # --- Save plot using output_dir ---
    scatter_plot_path = os.path.join(output_dir, 'prediction_scatter.png')
    plt.savefig(scatter_plot_path)
    print(f"Prediction scatter plot saved to {scatter_plot_path}")
    plt.close() # Close the plot

    return model

# --- Add main execution block with argparse ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a GNN model for solubility prediction.")
    parser.add_argument("--data", type=str, default="./data", help="Path to the data directory (containing raw/processed subfolders)")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory to save trained model and plots")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension size in GNN layers")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of GNN layers")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    print("Starting training process with arguments:")
    print(f"  Data Path: {args.data}")
    print(f"  Output Dir: {args.output_dir}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.lr}")
    # Add other args if needed

    train_model(
        data_path=args.data,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        lr=args.lr,
        epochs=args.epochs,
        seed=args.seed
    )
    print("Training finished.")