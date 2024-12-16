import torch
import clip

class CLIPModel:
    def __init__(self, device: str):
        """
        Initialize CLIP model with weighted feature extraction.
        
        Args:
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.device = device
        self.models = {}
        self.preprocess = None
        self.weights = {
            'ViT-B/32': 0.3,
            'ViT-B/16': 0.7
        }
        self._load_clip_model()
        
    def _load_clip_model(self):
        """
        Load multiple CLIP models and preprocessing function.
        """
        for model_name in self.weights.keys():
            model, preprocess = clip.load(model_name, self.device)
            self.models[model_name] = model
            
        self.preprocess = preprocess
    
    def extract_features(self, image: torch.Tensor) -> torch.Tensor:
        """
        Extract and combine features from multiple CLIP models with weights.
        
        Args:
            image: Input image tensor
            
        Returns:
            torch.Tensor: Weighted and combined normalized feature vector
        """
        features = []
        
        with torch.no_grad():
            for model_name, model in self.models.items():
                feat = model.encode_image(image)
                feat = feat / feat.norm(dim=-1, keepdim=True)
                feat = feat * self.weights[model_name]
                features.append(feat)
            
            combined_features = torch.cat(features, dim=-1)
            combined_features = combined_features / combined_features.norm(dim=-1, keepdim=True)
            
            return combined_features