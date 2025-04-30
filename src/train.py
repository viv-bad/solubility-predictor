import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as pl
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from src.data.dataset import SolubilityDataset
from src.models.gnn_model import SolubilityGNN

def train_model(data_path, batch_size=32, hidden_dim=64, num_layers=3, lr=0.001, epochs = 100, seed=42):
    """
    Train a GNN model for solubility prediction.

    Args: 
        data_path: Path to the processed data
        batch_size: Batch size for training
        hidden_dim: Number of hidden dimensions in the GNN
        num_layers: Number of GNN layers
        lr: Learning rate
        epochs: Number of training epochs
        seed: Random seed for reproducibility (testing only) 
    """

    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Set device (GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset
    dataset = SolubilityDataset(root = data_path, csv_file = os.path.join(data_path, 'raw/solubility_data.csv'))

    # split dataset into train, validation and test sets
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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model and move to device
    model = SolubilityGNN(
        node_feature_dim = train_dataset[0].x.shape[1],
        edge_feature_dim=train_dataset[0].edge_attr.shape[1],
        hidden_dim=hidden_dim,
        num_layers=num_layers
    ).to(device)

    # define optimizer and loss fns
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # Training loop
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_model_path = os.path.join(data_path, 'models/best_model.pth')

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            # Move batch to device
            batch = batch.to(device)
            
            optimizer.zero_grad()

            # forward pass
            pred = model(batch)

            # calculate loss
            loss = loss_fn(pred, batch.y)

            # backward pass
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch.num_graphs
        
        train_loss /= len(train_dataset)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0
        with torch.inference_mode():
            for batch in val_loader:
                # Move batch to device
                batch = batch.to(device)
                
                pred = model(batch)
                loss = loss_fn(pred, batch.y)
                val_loss += loss.item() * batch.num_graphs
        
        val_loss /= len(val_dataset)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved at {best_model_path}")
        
    # Plot training curve
    plt.figure(figsize=(10,6))
    plt.plot(range(1, epochs+1), train_losses, label="Training Loss")
    plt.plot(range(1, epochs+1), val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (MSE)")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig(os.path.join(data_path, '../models/training_curve.png'))

    # Load best model for eval
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    # evaluate on test set
    model.eval()
    test_preds, test_targets = [], []
    with torch.inference_mode():
        for batch in test_loader:
            # Move batch to device
            batch = batch.to(device)
            
            pred = model(batch)
            test_preds.append(pred.cpu())  # Move predictions back to CPU
            test_targets.append(batch.y.cpu())  # Move targets back to CPU
    
    test_preds = torch.cat(test_preds, dim = 0).numpy()
    test_targets = torch.cat(test_targets, dim=0).numpy()

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(test_targets, test_preds))
    r2 = r2_score(test_targets, test_preds)

    print(f"Test RMSE: {rmse:.4f}, R2 Score: {r2:.4f}")

    # plot predictions vs actual
    plt.figure(figsize=(10,6))
    plt.scatter(test_targets, test_preds, alpha=0.5)
    plt.plot([min(test_targets), max(test_targets)], [min(test_targets), max(test_targets)], '--', color='red')
    plt.xlabel("Actual Solubility")
    plt.ylabel("Predicted Solubility")
    plt.title("Predictions vs Actual")
    plt.legend()
    plt.savefig(os.path.join(data_path, '../models/prediction_scatter.png'))

    return model

if __name__ == "__main__":
    # update this path to data dir
    data_path = "data"
    train_model(data_path, epochs=50)
    
    
    
