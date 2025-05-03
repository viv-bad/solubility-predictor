# api/config.py
import os
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    """API settings that can be overridden with environment variables."""
    
    # API Settings
    API_TITLE: str = "Molecular Solubility Prediction API"
    API_DESCRIPTION: str = "API for predicting molecular solubility from SMILES strings using a Graph Neural Network"
    API_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # CORS Settings
    ALLOW_ORIGINS: list = ["*", "http://localhost:3000", "http://frontend:3000"]
    ALLOW_CREDENTIALS: bool = True
    ALLOW_METHODS: list = ["*"]
    ALLOW_HEADERS: list = ["*"]
    
    # Model Settings
    MODEL_DIR: str = os.path.join("data", "models")
    MODEL_FILENAME: str = "best_model.pth"
    
    @property
    def MODEL_PATH(self):
        """Get the full path to the model file."""
        return os.path.join(self.MODEL_DIR, self.MODEL_FILENAME)
    
    # Logging Settings
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

@lru_cache()
def get_settings():
    """Get API settings, cached for performance."""
    return Settings()