import os
from dataclasses import dataclass
from typing import Dict, List
from dataclasses import dataclass, field


@dataclass
class Config:
    ensemble_thresholds: List[float] = field(default_factory=lambda: [0.15, 0.2, 0.25])
    
    # Paths
    train_path: str = "./origin/train"
    test_path: str = "./origin/test"
    results_path: str = "./results"
    
    # Model parameters
    anomaly_threshold: float = 0.2
    n_samples: int = 5
    
    # Runtime settings
    save_predictions: bool = False
    save_results: bool = True
    
    def __post_init__(self):
        # Create results directory if it doesn't exist
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)