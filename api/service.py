# api/service.py
import os
import logging
from typing import List, Dict, Any, Union
import torch

# Fix imports to properly access the model
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from inference import SolubilityPredictor
from api.config import get_settings

# Setup logging
logger = logging.getLogger(__name__)

class PredictionService:
    """Service for handling molecular solubility predictions."""
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern to ensure only one instance of the service exists."""
        if cls._instance is None:
            cls._instance = super(PredictionService, cls).__new__(cls)
            cls._instance.predictor = None
        return cls._instance
    
    def initialize(self, model_path=None):
        """
        Initialize the prediction service with a model.
        
        Args:
            model_path: Path to the model file. If None, use the path from settings.
        """
        if self.predictor is not None:
            logger.info("Predictor already initialized")
            return
        
        settings = get_settings()
        model_path = model_path or settings.MODEL_PATH
        
        logger.info(f"Initializing predictor with model: {model_path}")
        try:
            self.predictor = SolubilityPredictor(model_path)
            logger.info("Predictor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize predictor: {str(e)}")
            raise RuntimeError(f"Failed to initialize predictor: {str(e)}")
    
    def get_predictor(self):
        """Get the predictor, initializing it if necessary."""
        if self.predictor is None:
            self.initialize()
        return self.predictor
    
    def predict_solubility(self, smiles: str) -> Dict[str, Any]:
        """
        Predict solubility for a molecule from SMILES string.
        
        Args:
            smiles: SMILES string of the molecule
            
        Returns:
            Dictionary containing prediction results
        """
        predictor = self.get_predictor()
        return predictor.predict_from_smiles(smiles)
    
    def predict_batch(self, smiles_list: List[str]) -> List[Dict[str, Any]]:
        """
        Predict solubility for multiple molecules from SMILES strings.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            List of dictionaries containing prediction results
        """
        predictor = self.get_predictor()
        return predictor.predict_batch(smiles_list)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary containing model information
        """
        predictor = self.get_predictor()
        model = predictor.model
        
        return {
            "model_type": model.__class__.__name__,
            "device": str(predictor.device),
            "using_gpu": torch.cuda.is_available(),
            "parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)
        }

# Create a singleton instance
prediction_service = PredictionService()