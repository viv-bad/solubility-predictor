from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import os
import sys

# project added to root path to fix imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from inference import SolubilityPredictor

app = FastAPI(
    title="Molecular Solubility Prediction API",
    description="API for predicting molecular solubility from SMILES strings",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], # TODO: in prod, we need to specify the actual frontend url
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Initialize the model
MODEL_PATH = os.path.join("data", "models", "best_model.pth")
predictor = SolubilityPredictor(MODEL_PATH)

# Request/Response model definitions TOOD: move to separate file
class SmilesInput(BaseModel):
    smiles: str

class BatchSmilesInput(BaseModel):
    smiles_list: List[str]

class PredictionOutput(BaseModel):
    smiles: str
    predicted_solubility: float = Field(..., description="Predicted solubility in log(mol/L)")
    solubility_level: str # TODO: set to interface/type with list of solubility levels
    mol_weight: float
    logp: float
    num_atoms: int

class BatchPredictionOutput(BaseModel):
    predictions: List[PredictionOutput]

@app.get("/")
def read_root():
    return {"Message": "Molecular Solubility Prediction API is running"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionOutput)
def predict_solubility(input_data: SmilesInput):
    try:
        result = predictor.predict_from_smiles(input_data.smiles)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/batch-predict", response_model=BatchPredictionOutput)
def batch_predict(input_data: BatchSmilesInput):
    try:
        results = predictor.predict_batch(input_data.smiles_list)
        return {"predictions": results}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
        

