from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import os
import sys
import tempfile
import io
import base64
from PIL import Image
import logging

# Add project root to path to fix imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import our modules
from api.config import get_settings
from api.validation import SmilesInput, BatchSmilesInput, SolubilityPrediction, BatchPredictionResponse
from api.service import prediction_service
from api.utils import validate_smiles, smiles_to_base64_image, get_sample_molecules

# Setup logging
settings = get_settings()
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION,
    debug=settings.DEBUG
)

# Add CORS middleware to allow cross-origin requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOW_ORIGINS,
    allow_credentials=settings.ALLOW_CREDENTIALS,
    allow_methods=settings.ALLOW_METHODS,
    allow_headers=settings.ALLOW_HEADERS,
)

@app.get("/")
def read_root():
    """Root endpoint to check if the API is running"""
    return {"Message": "Molecular Solubility Prediction API is running"}

@app.get("/health")
def health_check():
    """Health check endpoint."""
    try:
        # Check if model can be loaded
        prediction_service.initialize()
        # Get additional model info
        model_info = prediction_service.get_model_info()
        return {
            "status": "healthy", 
            "model_loaded": True,
            "model_info": model_info
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "unhealthy", "error": str(e)}
        )

@app.post("/predict", response_model=SolubilityPrediction)
def predict_solubility(input_data: SmilesInput):
    """
    Predict solubility of a single molecule from SMILES string.

    Returns prediction results including solubility value and molecular properties.
    """
    try:
        # make prediction using service
        result = prediction_service.predict_solubility(input_data.smiles)
        logger.info(f"Prediction successful for {input_data.smiles}")
        return result
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=422, detail=f"Invalid input: {str(e)}")
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/batch-predict", response_model=BatchPredictionResponse)
def batch_predict(input_data: BatchSmilesInput):
    """
    Predict solubility for multiple molecules from a list of SMILES strings.
    
    Returns prediction results for each molecule in the batch.
    """
    try:
        # Validate SMILES
        invalid_smiles = []
        valid_smiles = []
        
        for smiles in input_data.smiles_list:
            is_valid, _ = validate_smiles(smiles)
            if not is_valid:
                invalid_smiles.append(smiles)
            else:
                valid_smiles.append(smiles)
        
        if invalid_smiles:
            logger.warning(f"Found {len(invalid_smiles)} invalid SMILES strings")
            
        # Make batch prediction for valid SMILES
        if valid_smiles:
            results = prediction_service.predict_batch(valid_smiles)
            logger.info(f"Batch prediction successful for {len(valid_smiles)} molecules")
        else:
            results = []
            
        return {
            "predictions": results,
            "invalid_count": len(invalid_smiles),
            "valid_count": len(valid_smiles),
            "invalid_smiles": invalid_smiles if invalid_smiles else None
        }
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.post("/visualize-molecule")
async def visualize_molecule(smiles: str = Form(...)):
    """
    Generate a visualization of a molecule from SMILES string.

    Returns a base64 encoded image of the molecule.
    """
    try:
        # convert SMILES to base64 
        img_str = smiles_to_base64_image(smiles)
        if img_str is None:
            raise HTTPException(status_code=422, detail=f"Invalid SMILES string: {smiles}")
        return { "image": img_str}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Visualization error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Visualization failed: {str(e)}")

@app.post("/predict-with-visualization")
async def predict_with_visualization(smiles: str = Form(...)):
    """
    Predict solubility and generate molecule visualization in one call.
    
    Returns prediction results and a base64-encoded image of the molecule.
    """
    try:
        # Validate SMILES
        is_valid, _ = validate_smiles(smiles)
        if not is_valid:
            raise HTTPException(status_code=422, detail=f"Invalid SMILES string: {smiles}")
        
        # Make prediction
        result = prediction_service.predict_solubility(smiles)
        
        # Generate molecule image
        img_str = smiles_to_base64_image(smiles)
        if img_str is None:
            logger.warning(f"Failed to generate molecule image for valid SMILES: {smiles}")
        else:
            # Add image to result
            result["image"] = img_str
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction with visualization error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process request: {str(e)}")

@app.get("/sample-molecules")
def get_sample_molecules_route():
    """Get a list of sample molecules with varying solubility levels."""
    return {"samples": get_sample_molecules()}


@app.get("/model-info")
def get_model_info():
    """
    Get information about the loaded model.
    """
    try:
        # initialize service if not already done
        prediction_service.initialize()
        # get model info
        model_info = prediction_service.get_model_info()
        return {
         "model_info": model_info
        }
    except Exception as e:
        logger.error(f"Failed to get model info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

@app.post("/validate-smiles")
def validate_smiles_route(smiles: str = Form(...)):
    """
    Validate a SMILES string.
    
    Returns whether the SMILES string is valid.
    """
    is_valid, mol = validate_smiles(smiles)
    if is_valid and mol is not None:
        return {
            "valid": True,
            "atom_count": mol.GetNumAtoms(),
            "bond_count": mol.GetNumBonds(),
            "has_aromaticity": any(atom.GetIsAromatic() for atom in mol.GetAtoms())
            # TODO: add more properties as needed
        }
    return {"valid": False}


# TODO: consider pre-loading the model on app startup instead to avoid latency from first request to prediction endpoints - @app.on_event("startup")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
        

