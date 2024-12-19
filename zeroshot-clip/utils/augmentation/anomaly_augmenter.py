import numpy as np
from PIL import Image
from typing import Optional
from .geometric import LocalDeformation, RandomDeletion, RandomCuts
from .noise import GaussianNoise, TextureCorruption
from .color import RedDotAnomaly, PatternInjection, ColorDistortion

class AnomalyAugmenter:
    def __init__(self, severity: float = 0.5):
        self.severity = severity
        self.rng = np.random.RandomState(42)
        
        self.augmentations = {
            'pattern_injection': (PatternInjection(severity * 1.4), 0.3), 
            'red_dot': (RedDotAnomaly(severity * 1.5), 0.15),          
            'texture_corruption': (TextureCorruption(severity * 1.4), 0.15), 
            'random_deletion': (RandomDeletion(severity * 1.3), 0.15),    
            'random_cuts': (RandomCuts(severity * 1.3), 0.15),            
            'local_deformation': (LocalDeformation(severity * 1.2), 0.05),
            'gaussian_noise': (GaussianNoise(severity * 1.3), 0.05)    
        }
    
    def generate_anomaly(self, image: Image.Image) -> Optional[Image.Image]:
        try:
            img = image.copy()
            augmentations, weights = zip(*self.augmentations.values())
            selected_aug = self.rng.choice(augmentations, p=weights)
            return selected_aug(img)
            
        except Exception as e:
            print(f"Error in anomaly generation: {str(e)}")
            return image