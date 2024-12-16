import os
from dataclasses import dataclass

@dataclass
class Config:
    # Paths
    train_path: str = "./dataset/train"
    test_path: str = "./dataset/test"
    results_path: str = "zeroshot-clip/results"
    
    # Model parameters
    anomaly_threshold: float = 0.2
    n_samples: int = 5
    
    # Runtime settings
    save_predictions: bool = True
    save_results: bool = True
    
    def __post_init__(self):
        # Create results directory if it doesn't exist
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)