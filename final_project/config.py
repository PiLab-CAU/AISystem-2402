import os
from dataclasses import dataclass

@dataclass
class Config:
    # Paths
    train_path: str = "/workspace/train"
    test_path: str = "/workspace/test"
    results_path: str = "./results"
    
    # Model parameters
    n_samples: int = 5
    
    # Runtime settings
    save_predictions: bool = True
    save_results: bool = True
    
    def __post_init__(self):
        # Create results directory if it doesn't exist
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)