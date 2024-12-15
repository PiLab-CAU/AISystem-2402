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
        model, preprocess = clip.load('ViT-B/32', self.device)
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
            features = features.float()  # double -> float 변환
            
            # 차원을 맞추기 위한 선형 변환 추가
            if not hasattr(self, 'projection'):
                self.projection = torch.nn.Linear(
                    features.shape[-1], 
                    512  # CLIP의 기본 임베딩 차원
                ).to(self.device)
                
            projected_features = self.projection(features)
            return projected_features / projected_features.norm(dim=-1, keepdim=True)