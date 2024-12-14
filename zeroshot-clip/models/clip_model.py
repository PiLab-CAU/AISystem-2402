import torch
import clip

class CLIPModel:
    def __init__(self, device: str):
        """
        Initialize CLIP model.
        
        Args:
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.device = device
        self.models = {}
        self.preprocess = None
        self._load_clip_model()
        
    def _load_clip_model(self):
        """
        Load multiple CLIP models and preprocessing function.
        """
        model_names = ['ViT-B/32', 'ViT-B/16']
        
        for name in model_names:
            model, preprocess = clip.load(name, self.device)
            self.models[name] = model
        
        self.preprocess = preprocess
    
    def extract_features(self, image: torch.Tensor) -> torch.Tensor:
        """
        Extract and combine features from multiple CLIP models.
        
        Args:
            image: Input image tensor
            
        Returns:
            torch.Tensor: Combined normalized feature vector
        """
        features = []
        with torch.no_grad():
            for model in self.models.values():
                feat = model.encode_image(image)
                feat = feat / feat.norm(dim=-1, keepdim=True)
                features.append(feat)
                
            if len(features) > 1:
                combined_features = torch.cat(features, dim=-1)
                combined_features = combined_features / combined_features.norm(dim=-1, keepdim=True)
            else:
                combined_features = features[0]
                
            return combined_features