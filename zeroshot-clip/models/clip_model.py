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
    
    # def extract_features(self, image: torch.Tensor) -> torch.Tensor:
    #     """
    #     Extract features from an image using CLIP.
        
    #     Args:
    #         image: Input image tensor
            
    #     Returns:
    #         torch.Tensor: Normalized feature vector
    #     """
    #     with torch.no_grad():
    #         features = self.model.encode_image(image)
    #         features = features.float()  # double -> float 변환
            
    #         # 차원을 맞추기 위한 선형 변환 추가
    #         if not hasattr(self, 'projection'):
    #             self.projection = torch.nn.Linear(
    #                 features.shape[-1], 
    #                 512  # CLIP의 기본 임베딩 차원
    #             ).to(self.device)
                
    #         projected_features = self.projection(features)
    #         return projected_features / projected_features.norm(dim=-1, keepdim=True)
    def extract_features(self, image: torch.Tensor) -> torch.Tensor:
        """
        Extract features from an image using CLIP.
        
        Args:
            image: Input image tensor
            
        Returns:
            torch.Tensor: Normalized and scaled feature vector
        """
        with torch.no_grad():
            features = self.model.encode_image(image)
            features = features.float()  # double -> float 변환
            
            # L2 normalization
            features = features / features.norm(dim=-1, keepdim=True)
            
            # Feature scaling: 값의 범위를 조정하여 안정성 향상
            # scale factor를 좀 더 작게 설정하여 미세한 차이를 더 잘 포착
            scale_factor = 20.0  # 기본값보다 작은 값으로 설정
            features = features * scale_factor
            
            return features