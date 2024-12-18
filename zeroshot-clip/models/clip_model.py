import torch
import clip
import math

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
        features = []
        attention_weights = []
        
        with torch.no_grad():
            for model_name, model in self.models.items():
                # Multi-scale 특징 추출을 위한 이미지 크기 조정
                # clip_model.py에서
                image_scales = [0.5, 0.75, 1.0, 1.25, 1.5]  # 스케일 범위 확대  
                model_features = []
                
                for scale in image_scales:
                    # 이미지 크기 조정
                    scaled_size = [int(x * scale) for x in image.shape[-2:]]
                    scaled_image = torch.nn.functional.interpolate(
                        image, 
                        size=scaled_size, 
                        mode='bilinear'
                    )
                    scaled_image = torch.nn.functional.interpolate(
                        scaled_image, 
                        size=image.shape[-2:], 
                        mode='bilinear'
                    )
                    
                    # 특징 추출
                    feat = model.encode_image(scaled_image)
                    feat = feat / feat.norm(dim=-1, keepdim=True)
                    model_features.append(feat)
                
                # 여러 스케일의 특징을 결합
                multi_scale_feat = torch.mean(torch.stack(model_features), dim=0)
                
                # Attention 계산
                attention = torch.nn.functional.softmax(
                    multi_scale_feat.mean(dim=-1) * math.sqrt(multi_scale_feat.size(-1)), 
                    dim=0
                )
                
                weighted_feat = multi_scale_feat * attention.unsqueeze(-1)
                features.append(weighted_feat * self.weights[model_name])
                attention_weights.append(attention)
            
            # 모든 모델의 특징을 결합
            combined_features = torch.cat(features, dim=-1)
            combined_features = combined_features / combined_features.norm(dim=-1, keepdim=True)
            
            return combined_features