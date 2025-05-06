# SolPred API

This API provides endpoints for predicting the solubility of molecules from their SMILES representations. It uses a Graph Neural Network (GNN) trained on a solubility dataset.

## Live Demo

**Try it now:** [https://solpred.netlify.app/](https://solpred.netlify.app/)

## Features

- Predict solubility for a single molecule from SMILES string
- Batch predictions for multiple molecules
- Visualize the molecular structure
- Combined prediction and visualization in a single request
- Sample molecules with varying solubility levels
- Model information and health checks

## Installation

### Prerequisites

- Python 3.8+
- RDKit
- PyTorch
- PyTorch Geometric

### Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/viv-bad/solubility-predictor.git
   cd solubility-prediction
   ```

2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Make sure you have a trained model available at `data/models/best_model.pth`. If not, follow the instructions in the main README to train a model.

## Running the API

### Using the Python Script

```bash
docker compose build
docker compose up
```

Options:

- `--host`: Host to bind the server to (default: "0.0.0.0")
- `--port`: Port to bind the server to (default: 8000)
- `--reload`: Enable auto-reload for development
- `--workers`: Number of worker processes (default: 1)
- `--log-level`: Log level (choices: debug, info, warning, error, critical, default: info)

### Using the Shell Script (Unix/Linux/Mac)

```bash
chmod +x run_api.sh  # Make the script executable (first time only)
./run_api.sh
```

The shell script accepts the same options as the Python script.

## API Endpoints

Once the server is running, you can access the OpenAPI documentation at:

```
http://localhost:8000/docs
```

### Root Endpoint

- **GET /** - Check if the API is running

### Health Check

- **GET /health** - Check the health of the API and the model

### Prediction Endpoints

- **POST /predict** - Predict solubility for a single molecule

  - Request body: `{"smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"}`

- **POST /batch-predict** - Predict solubility for multiple molecules

  - Request body: `{"smiles_list": ["CC(=O)OC1=CC=CC=C1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"]}`

- **POST /visualize-molecule** - Generate a visualization of a molecule

  - Form data: `smiles=CC(=O)OC1=CC=CC=C1C(=O)O`

- **POST /predict-with-visualization** - Predict solubility and generate visualization
  - Form data: `smiles=CC(=O)OC1=CC=CC=C1C(=O)O`

### Utility Endpoints

- **GET /sample-molecules** - Get a list of sample molecules with varying solubility levels

- **GET /model-info** - Get information about the loaded model

- **POST /validate-smiles** - Validate a SMILES string
  - Form data: `smiles=CC(=O)OC1=CC=CC=C1C(=O)O`

## Sample Usage with curl

### Predict solubility for aspirin

```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"}'
```

### Batch predict for multiple molecules

```bash
curl -X 'POST' \
  'http://localhost:8000/batch-predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"smiles_list": ["CC(=O)OC1=CC=CC=C1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"]}'
```

### Predict with visualization

```bash
curl -X 'POST' \
  'http://localhost:8000/predict-with-visualization' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/x-www-form-urlencoded' \
  -d 'smiles=CC(=O)OC1=CC=CC=C1C(=O)O'
```

## Python Client Example

```python
import requests
import json
import base64
from PIL import Image
import io

# API endpoint
API_URL = "http://localhost:8000"

# Predict solubility for a single molecule
def predict_solubility(smiles):
    response = requests.post(
        f"{API_URL}/predict",
        json={"smiles": smiles}
    )
    return response.json()

# Example usage
smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
result = predict_solubility(smiles)
print(f"Predicted solubility: {result['predicted_solubility']}")
print(f"Solubility level: {result['solubility_level']}")

# Predict with visualization
def predict_with_visualization(smiles):
    response = requests.post(
        f"{API_URL}/predict-with-visualization",
        data={"smiles": smiles}
    )
    result = response.json()

    # Display the molecule image
    if "image" in result:
        image_data = base64.b64decode(result["image"])
        image = Image.open(io.BytesIO(image_data))
        image.show()

    return result

# Example usage
result = predict_with_visualization("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")  # Caffeine
print(f"Predicted solubility: {result['predicted_solubility']}")
print(f"Solubility level: {result['solubility_level']}")
```
