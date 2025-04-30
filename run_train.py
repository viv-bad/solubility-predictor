
# run_train.py
# This script ensures the Python path is set correctly before running the main training code.

print("Importing fix_imports to set up sys.path...")
import fix_imports # This executes the code in fix_imports.py

print("Importing training function...")
# Now that sys.path is fixed, this import should work
from src.train import train_model

if __name__ == "__main__":
    print("Starting training process...")
    # Call the train_model function with appropriate parameters
    # Ensure data paths are relative to the project root
    data_root_path = "./data"
    model = train_model(data_root_path, epochs=50) # Adjust epochs etc. as needed
    print("Training finished.")
    