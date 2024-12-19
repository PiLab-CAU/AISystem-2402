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
        self.model, self.preprocess = self._load_clip_model()
        
    def _load_clip_model(self):
        """
        Load the CLIP model and preprocessing function.
        
        Returns:
            tuple: (model, preprocess_function)
        """
        model, preprocess = clip.load('ViT-B/16', self.device)
        return model, preprocess
    
    def extract_features(self, image: torch.Tensor) -> torch.Tensor:
        """
        Extract features from an image using CLIP.
        
        Args:
            image: Input image tensor
            
        Returns:
            torch.Tensor: Normalized feature vector
        """
        with torch.no_grad():
            features = self.model.encode_image(image)
            return features / features.norm(dim=-1, keepdim=True)
        
    def extract_text_features(self, text):
            tokenized_text = clip.tokenize(text).to(self.device)
            with torch.no_grad():
                features = self.model.encode_text(tokenized_text)
            return features / features.norm(dim=-1, keepdim=True)
        
        
        
        

if __name__ == '__main__':
    model = CLIPModel('cuda:0')

    x = torch.randn(3,224,224).unsqueeze(0).to(model.device)
    print(x.shape)
    
    #print(model)
    #print(model.preprocess)
    print(model.device)
    print(model.extract_features(x).min(), model.extract_features(x).max())