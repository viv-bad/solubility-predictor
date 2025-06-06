# Molecular Solubility Predictor (SolPred)

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch)](https://pytorch.org/)
[![PyTorch Geometric](https://img.shields.io/badge/PyTorch_Geometric-grey?style=flat)](https://pyg.org/)
[![RDKit](https://img.shields.io/badge/RDKit-orange?style=flat)](https://www.rdkit.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker)](https://www.docker.com/)

## Overview

SolPred is a comprehensive project for predicting the aqueous solubility of chemical compounds using a Graph Neural Network (GNN). It takes a molecule's SMILES string as input, converts it into a graph representation, and feeds it into a trained PyTorch Geometric model to predict its solubility (logS value).

The project includes:

1.  **Core ML Pipeline (`src/solpred`):** Data loading, preprocessing (SMILES to graph), GNN model definition, training script, and inference logic.
2.  **Model Analysis (`analyze_model.py`):** Scripts to evaluate model performance, visualize embedding spaces (t-SNE), analyze prediction errors, and investigate feature importance.
3.  **FastAPI Service (`api/`):** A robust API built with FastAPI to serve solubility predictions, validate SMILES, generate molecule visualizations, and provide model information.
4.  **Dockerization (`Dockerfile`, `docker-compose.yml`):** Configuration to easily build and run the API service (and potentially a frontend) using Docker containers.

This repository contains the backend ML model and API service. A separate repository likely holds the corresponding frontend web application (Nuxt.js/Vue.js).

## Live Demo

**Try it now:** [https://solpred.netlify.app/](https://solpred.netlify.app/)

## Features

### Core ML & Analysis

- **Graph-Based Learning:** Utilizes Graph Neural Networks (GNNs) via PyTorch Geometric to capture molecular structure effectively.
- **Data Handling:** Converts SMILES strings to molecular graphs with relevant atom and bond features using RDKit. Uses `torch_geometric.data.Dataset` for efficient loading.
- **Training:** Includes a script (`src/solpred/train.py`) for training the GNN model, including data splitting, validation, model saving, and results plotting.
- **Inference:** Provides a `SolubilityPredictor` class (`inference.py`) for easy prediction from SMILES, batches, or CSV files.
- **Model Evaluation & Analysis:** Offers `analyze_model.py` script for:
  - t-SNE visualization of learned molecular embeddings.
  - Detailed error analysis and correlation with molecular descriptors.
  - Basic feature/atom importance analysis.

### API Service (FastAPI)

- **Single Prediction (`/predict`):** Predict solubility for one molecule.
- **Batch Prediction (`/batch-predict`):** Predict solubility for multiple molecules efficiently.
- **SMILES Validation (`/validate-smiles`):** Check if a SMILES string is chemically valid using RDKit.
- **Molecule Visualization (`/visualize-molecule`):** Generate a 2D image of a molecule from SMILES.
- **Combined Prediction & Visualization (`/predict-with-visualization`):** Get prediction results and the molecule image in one API call.
- **Sample Molecules (`/sample-molecules`):** Retrieve a predefined list of molecules for easy testing.
- **Model Information (`/model-info`):** Get details about the loaded GNN model.
- **Health Check (`/health`):** Endpoint to monitor API and model status.
- **Interactive Docs (`/docs`):** Automatically generated OpenAPI documentation.

## Technology Stack

- **Programming Language:** Python 3.11+
- **Machine Learning:** PyTorch, PyTorch Geometric
- **Cheminformatics:** RDKit
- **API Framework:** FastAPI
- **Data Handling:** Pandas, NumPy
- **Validation:** Pydantic
- **Visualization:** Matplotlib, Seaborn
- **Containerization:** Docker, Docker Compose
- **Packaging:** Setuptools (via `pyproject.toml`)

## Project Structure

```
.
├── api/ # FastAPI application code
│ ├── **init**.py
│ ├── config.py
│ ├── main.py # FastAPI app definition and endpoints
│ ├── README.md # API specific documentation
│ ├── service.py # Business logic for predictions
│ ├── utils.py # API utility functions (validation, image gen)
│ └── validation.py # Pydantic models for request/response validation
├── data/ # Data files (Requires user input or generation)
│ ├── raw/ # Raw dataset files (e.g., solubility_data.csv)
│ └── processed/ # Processed PyTorch Geometric data (generated by dataset.py)
├── models/ # Saved model artifacts and training outputs
│ ├── best_model.pth # Example: Trained model weights
│ ├── training_curve.png # Example: Output plot
│ └── ...
├── notebooks/ # Jupyter notebooks for experimentation and setup
│ └── solubility_prediction_colab.ipynb
├── src/
│ └── solpred/ # Source code package for the ML pipeline
│ ├── **init**.py
│ ├── data/ # Data loading and processing modules
│ │ ├── **init**.py
│ │ ├── dataset.py
│ │ ├── explore.py
│ │ ├── molecule_graph.py
│ │ └── test_graph.py
│ ├── models/ # Model definition modules
│ │ ├── **init**.py
│ │ └── gnn_model.py
│ └── train.py # Training script
├── analysis_results/ # Default output directory for analyze_model.py
├── solubility-predictor-webapp/ # (Likely) Separate directory/submodule for frontend
├── .gitignore
├── analyze_model.py # Script for model analysis
├── docker-compose.yml # Docker Compose configuration for services
├── Dockerfile # Dockerfile for building the backend API image
├── inference.py # Script/class for running inference
├── pyproject.toml # Python package configuration
├── README.md # This file
├── requirements.txt # Python dependencies
├── test_api.py # Script for testing API endpoints
└── ... # Other config files (e.g., .env.example)
```

## Prerequisites

- Python (3.11 or later recommended)
- `pip` and `venv` (standard Python tools)
- Docker Engine
- Docker Compose
- Git (for cloning)

## Installation & Setup (Local Development)

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/viv-bad/solubility-predictor
    cd solubility-predictor
    ```

2.  **Create and Activate Virtual Environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    # venv\Scripts\activate    # On Windows
    ```

3.  **Install Dependencies:**
    _(Note TODO:: `requirements.txt` includes dev/analysis tools. Consider splitting into `requirements.txt` (core) and `requirements-dev.txt` later)_

    ```bash
    pip install -r requirements.txt
    ```

    _(Depending on your OS and existing installations, PyTorch/PyG/RDKit might require specific installation steps - refer to their official documentation if needed.)_

4.  **Install the `solpred` Package (Editable Mode):** This makes the `src/solpred` code importable.

    ```bash
    pip install -e .
    ```

5.  **Prepare Data:**

    - Place your raw dataset CSV file (e.g., `solubility_data.csv` containing 'smiles' and 'solubility' columns) inside the `data/raw/` directory. Create the directories if they don't exist (`mkdir -p data/raw`).
    - The processed data (`data/processed/solubility_data.pt`) will be generated automatically the first time the `SolubilityDataset` class is used (e.g., during training).

6.  **Obtain Trained Model:**
    - Either train a model using the instructions below (see **Usage -> Training**), which will save `best_model.pth` to the `models/` directory.
    - Or, if you have a pre-trained `best_model.pth` file, place it in the `models/` directory (create it if needed: `mkdir models`).

## Usage

_(Ensure your virtual environment is activated: `source venv/bin/activate`)_
_(Run commands from the project root directory)_

### 1. Training the Model

- Modify training parameters (epochs, learning rate, batch size, etc.) via command-line arguments.
- The script uses data from `--data` directory and saves the best model and plots to `--output_dir`.

```bash
# Example: Train for 50 epochs using data in ./data and saving results to ./models
python -m solpred.train --data ./data --output_dir ./models --epochs 50

# Example: Train with different hidden dimensions
python -m solpred.train --data ./data --output_dir ./models --epochs 100 --hidden_dim 128
```

- Outputs (`best_model.pth`, `training_curve.png`, `prediction_scatter.png`) will be saved in the specified `output_dir` (default: `./models`).

### 2. Running Inference (Command Line)

- Use `inference.py` to predict solubility for single SMILES or from a CSV file. Requires a trained model file.

```bash
# Predict for a single SMILES string
python inference.py --model models/best_model.pth --smiles "CCO"

# Predict and visualize a single SMILES
python inference.py --model models/best_model.pth --smiles "CC(=O)OC1=CC=CC=C1C(=O)O" --visualize

# Predict from a CSV file (assuming input.csv has a 'smiles' column)
# python inference.py --model models/best_model.pth --csv path/to/input.csv --output path/to/predictions.csv
```

### 3. Analyzing the Model

- Use `analyze_model.py` to perform various analyses on a trained model. Requires the dataset and a trained model file.

```bash
# Run all analyses (embeddings, errors, importance)
python analyze_model.py --model models/best_model.pth --data ./data --all --output analysis_results

# Run only error analysis
python analyze_model.py --model models/best_model.pth --data ./data --errors --output analysis_results
```

- Results (plots, CSV files) will be saved in the specified `--output` directory (default: `analysis_results/`).

### 4. Running the API Service (Docker Compose - Recommended)

- This is the easiest way to run the API, potentially alongside the frontend.
- Ensure you have a trained model (`best_model.pth`) in the `models/` directory. _Note: The default compose file mounts `./data/models` into the container, ensure your model path in `api/config.py` (`MODEL_PATH`) resolves correctly within the container (e.g., `/app/data/models/best_model.pth`). You may need to adjust `MODEL_DIR` in `api/config.py` to `data/models`._

```bash
# Build the Docker images (first time or after changes)
docker compose build

# Start the backend API service (and frontend if defined)
docker compose up backend # Or just 'docker compose up' to start all services

# Stop the services
docker compose down
```

- The API will be available at `http://localhost:8000`.
- The interactive API documentation (Swagger UI) will be at `http://localhost:8000/docs`.
- The health check endpoint is `http://localhost:8000/health`.

### 5. Testing the API

- A basic test script is provided. Run it _after_ starting the API service (e.g., via Docker Compose).

```bash
# Run API tests against the default http://localhost:8000
python test_api.py

# Test against a different URL
# python test_api.py --url http://deployed-api.com
```

## API Documentation

- **Interactive (Swagger UI):** Access `http://localhost:8000/docs` when the API service is running.
- **Detailed Information:** Refer to `api/README.md` for endpoint details and usage examples.

## Frontend Application

This repository focuses on the backend. The corresponding frontend web application (Nuxt.js/Vue.js) is available in a separate repository or subdirectory (https://github.com/viv-bad/solubility-predictor-webapp). Refer to its README for setup and usage instructions. The `docker-compose.yml` file is configured to build and run it alongside the backend.
