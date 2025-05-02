# api/validation.py
from rdkit import Chem
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional

class SmilesInput(BaseModel):
    """Model for single SMILES input."""
    smiles: str = Field(..., description="SMILES string representation of the molecule")
    
    @validator('smiles')
    def validate_smiles(cls, v):
        """Validate that the SMILES string is valid."""
        v = v.strip()
        mol = Chem.MolFromSmiles(v)
        if mol is None:
            raise ValueError("Invalid SMILES string")
        return v

class BatchSmilesInput(BaseModel):
    """Model for batch SMILES input."""
    smiles_list: List[str] = Field(..., description="List of SMILES strings to predict")
    
    @validator('smiles_list')
    def validate_smiles_list(cls, smiles_list):
        """Check that the list contains at least one SMILES and validate each one."""
        if not smiles_list:
            raise ValueError("SMILES list cannot be empty")
            
        # Check for valid/invalid SMILES, but allow them through for more detailed reporting
        for i, smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles.strip())
            if mol is None:
                # We don't raise an error here - instead we'll return invalid_smiles in the response
                pass
        
        return [s.strip() for s in smiles_list]

class SolubilityPrediction(BaseModel):
    """Model for solubility prediction results."""
    smiles: str
    compound_name: str
    predicted_solubility: float
    solubility_level: str
    mol_weight: float
    logp: float
    num_atoms: int
    # Add any additional fields your model predicts

class BatchPredictionResponse(BaseModel):
    """Model for batch prediction response."""
    predictions: List[Dict[str, Any]]
    invalid_count: int
    valid_count: int
    invalid_smiles: Optional[List[str]]