import os
from dataclasses import dataclass

@dataclass
class Config:
    # Paths
    train_path: str = "./train"
    test_path: str = "./test"
    results_path: str = "./results"
    # train_path: str = "/Users/kimhaeseong/AISystem-2402/zeroshot-clip/train"
    # test_path: str = "/Users/kimhaeseong/AISystem-2402/zeroshot-clip/test"
    # results_path: str = "/Users/kimhaeseong/AISystem-2402/zeroshot-clip/results"
    
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