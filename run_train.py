# run_train.py
# This script ensures the Python path is set correctly before running the main training code.

import argparse

print("Importing fix_imports to set up sys.path...")
import fix_imports # This executes the code in fix_imports.py

print("Importing training function...")
# Now that sys.path is fixed, this import should work
from src.train import train_model

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Train a model with specified parameters.')
    parser.add_argument('--data_path', type=str, default='./data',
                        help='Path to the data directory (default: ./data)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs (default: 50)')
    
    # Parse arguments
    args = parser.parse_args()
    
    print("Starting training process...")
    # Call the train_model function with command line arguments
    model = train_model(args.data_path, epochs=args.epochs)
    print("Training finished.")
    