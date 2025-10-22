"""
Configuration settings for the AI Analytics Platform
"""
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

@dataclass
class Config:
    """Main configuration class"""
    
    # App Settings
    APP_NAME: str = "AI Analytics Platform"
    VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # Data Settings
    MAX_FILE_SIZE: int = 200 * 1024 * 1024  # 200MB
    MAX_FILE_SIZE_MB: int = 200  # 200MB for display
    MAX_ROWS: int = 100000  # Maximum rows to process
    SUPPORTED_FORMATS: List[str] = field(default_factory=lambda: ['.csv', '.json', '.xlsx', '.parquet'])
    
    # AI Model Settings
    OPENAI_API_KEY: Optional[str] = os.getenv('OPENAI_API_KEY')
    MODEL_CACHE_DIR: str = './models'
    
    # Blockchain Settings
    BLOCKCHAIN_NETWORK: str = 'ganache'  # Local test network
    CONTRACT_ADDRESS: Optional[str] = None
    
    # Privacy Settings
    DIFFERENTIAL_PRIVACY_EPSILON: float = 1.0
    FEDERATED_LEARNING_ROUNDS: int = 5
    
    # Visualization Settings
    PLOT_THEME: str = 'plotly_white'
    DEFAULT_COLORS: List[str] = field(default_factory=lambda: ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])

# Global config instance
config = Config()

# Sample datasets configuration
SAMPLE_DATASETS = {
    'healthcare': {
        'name': 'Healthcare Patient Data',
        'description': 'Synthetic patient records with demographics and health metrics',
        'features': ['age', 'gender', 'bmi', 'blood_pressure', 'cholesterol', 'diagnosis']
    },
    'finance': {
        'name': 'Financial Transaction Data',
        'description': 'Synthetic financial transactions and customer data',
        'features': ['customer_id', 'transaction_amount', 'transaction_type', 'account_balance', 'credit_score']
    },
    'iot': {
        'name': 'IoT Sensor Data',
        'description': 'Synthetic IoT device sensor readings',
        'features': ['device_id', 'timestamp', 'temperature', 'humidity', 'pressure', 'status']
    }
}