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
        self.prompts = {
            "normal": [
                "a photo of a {class_name} in perfect condition",
                "a clear photo showing a standard {class_name}",
                "a photo showing a pristine {class_name}",
                "a photo of a clean, undamaged {class_name}",
                "a typical photo of a {class_name} in good shape"
            ],
            "anomaly": [
                "a photo of a {class_name} with obvious damage or defects",
                "a photo showing a broken {class_name}",
                "a photo of a damaged {class_name}",
                "a clear photo of a defective {class_name}",
                "a photo showing flaws in a {class_name}"
            ]
        }
    
    def encode_text_prompts(self, class_name: str):
        # 각 클래스에 대한 여러 프롬프트 생성
        normal_prompts = [prompt.format(class_name=class_name) 
                         for prompt in self.prompts["normal"]]
        anomaly_prompts = [prompt.format(class_name=class_name) 
                          for prompt in self.prompts["anomaly"]]
        
        # 텍스트 토큰화
        normal_tokens = clip.tokenize(normal_prompts).to(self.device)
        anomaly_tokens = clip.tokenize(anomaly_prompts).to(self.device)
        
        with torch.no_grad():
            normal_features = self.model.encode_text(normal_tokens)
            anomaly_features = self.model.encode_text(anomaly_tokens)
            
            # 평균 특징 벡터 계산
            normal_features = normal_features.mean(dim=0, keepdim=True)
            anomaly_features = anomaly_features.mean(dim=0, keepdim=True)
            
            # 정규화
            normal_features = normal_features / normal_features.norm(dim=-1, keepdim=True)
            anomaly_features = anomaly_features / anomaly_features.norm(dim=-1, keepdim=True)
        
        return normal_features, anomaly_features
    
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
            return features / features.norm(dim=-1, keepdim=True)
