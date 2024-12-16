import torch
import clip

class CLIPModel:
    def __init__(self, device: str):
        """
        Initialize CLIP model with weighted feature extraction including ViT-L/14.
        
        Args:
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.device = device
        self.models = {}
        self.preprocess = None
        self.weights = {
            'ViT-B/32': 0.2,  
            'ViT-B/16': 0.3, 
            'ViT-L/14': 0.5   
        }
        self._load_clip_model()
        
    def _load_clip_model(self):
        """
        Load multiple CLIP models and preprocessing function.
        Handle potential memory issues with ViT-L/14.
        """
        try:
            for model_name in self.weights.keys():
                print(f"Loading {model_name}...")
                try:
                    model, preprocess = clip.load(model_name, self.device)
                    self.models[model_name] = model
                    self.preprocess = preprocess
                except Exception as e:
                    print(f"Error loading {model_name}: {str(e)}")
                    self._adjust_weights(model_name)
                    
            if not self.models:
                raise ValueError("No models were successfully loaded")
                
        except Exception as e:
            print(f"Error in model initialization: {str(e)}")
            model, preprocess = clip.load('ViT-B/32', self.device)
            self.models = {'ViT-B/32': model}
            self.preprocess = preprocess
            self.weights = {'ViT-B/32': 1.0}
    
    def _adjust_weights(self, failed_model: str):
        """
        Adjust weights if a model fails to load.
        
        Args:
            failed_model: Name of the model that failed to load
        """
        weight_to_distribute = self.weights.pop(failed_model)
        
        if self.weights:
            total_remaining = sum(self.weights.values())
            for model in self.weights:
                self.weights[model] += (self.weights[model] / total_remaining) * weight_to_distribute
    
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
                
                attention = torch.nn.functional.softmax(feat.mean(dim=-1), dim=0)
                feat = feat * attention.unsqueeze(-1)
                
                feat = feat * self.weights[model_name]
                features.append(feat)
            
            combined_features = torch.cat(features, dim=-1)
            combined_features = combined_features / combined_features.norm(dim=-1, keepdim=True)
            
            return combined_features