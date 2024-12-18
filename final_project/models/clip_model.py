import torch
import clip
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

class EnhancedCLIPModel:
    def __init__(self, device: str):
        """
        Initialize enhanced CLIP model with multiple feature extraction methods.
        
        Args:
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.device = device
        self.model, self.preprocess = self._load_clip_model()
        self.feature_weights = {
            'global': 0.5,  # 전역 특징의 가중치 증가
            'local': 0.3,
            'attention': 0.2
        }
        
    def _load_clip_model(self) -> Tuple[torch.nn.Module, callable]:
        """
        Load the CLIP model and preprocessing function.
        
        Returns:
            tuple: (model, preprocess_function)
        """
        model, preprocess = clip.load('ViT-L/14', self.device)
        model = model.float()
        return model, preprocess
    
    def _extract_global_features(self, image: torch.Tensor) -> torch.Tensor:
        """
        Extract global features using standard CLIP encoding.
        
        Args:
            image: Input image tensor
            
        Returns:
            torch.Tensor: Normalized global feature vector
        """
        with torch.no_grad():
            image = image.float()
            features = self.model.encode_image(image)
            return features / features.norm(dim=-1, keepdim=True)
    
    def _extract_local_features(self, image: torch.Tensor) -> torch.Tensor:
        """
        Extract local features using intermediate layers.
        
        Args:
            image: Input image tensor
            
        Returns:
            torch.Tensor: Normalized local feature vector
        """
        with torch.no_grad():
            image = image.float()
            
            # Get initial features
            x = self.model.visual.conv1(image)  # [batch_size, width, height, channels]
            
            # Patch embedding
            B, C, H, W = x.shape
            x = x.reshape(B, C, H * W).permute(0, 2, 1)  # [batch_size, patches, channels]
            
            # Process through transformer
            for block in self.model.visual.transformer.resblocks[:-1]:
                x = block(x)
            
            # Extract local features from patches
            patch_features = x[:, 1:, :]  # Remove CLS token
            
            # Pool features in groups to reduce dimension
            n_groups = 4
            group_size = patch_features.size(1) // n_groups
            local_features = []
            
            for i in range(n_groups):
                start_idx = i * group_size
                end_idx = (i + 1) * group_size
                group_features = patch_features[:, start_idx:end_idx, :].mean(dim=1)
                local_features.append(group_features)
            
            # Combine local features
            local_features = torch.cat(local_features, dim=1)
            return F.normalize(local_features, dim=-1)
    
    def _extract_attention_features(self, image: torch.Tensor) -> torch.Tensor:
        """
        Extract attention-based features.
        
        Args:
            image: Input image tensor
            
        Returns:
            torch.Tensor: Normalized attention features
        """
        with torch.no_grad():
            image = image.float()
            
            # Initial processing
            x = self.model.visual.conv1(image)
            B, C, H, W = x.shape
            x = x.reshape(B, C, H * W).permute(0, 2, 1)
            
            # Get features from transformer blocks
            attention_weights = []
            
            for block in self.model.visual.transformer.resblocks:
                # Process through self-attention
                x = block(x)
                # Store attention weights (using the output as a proxy for attention)
                weights = x.mean(dim=-1, keepdim=True)
                attention_weights.append(weights)
            
            # Combine attention weights from different layers
            combined_weights = torch.mean(torch.stack(attention_weights, dim=1), dim=1)
            # Apply attention weights to final features
            attended_features = x * combined_weights
            # Global pooling
            attended_features = attended_features.mean(dim=1)
            
            return F.normalize(attended_features, dim=-1)
    
    def extract_features(self, image: torch.Tensor) -> torch.Tensor:
        """
        Extract combined features using multiple methods.
        
        Args:
            image: Input image tensor
            
        Returns:
            torch.Tensor: Normalized combined feature vector
        """
        try:
            image = image.float()
            
            # Extract features using different methods
            global_features = self._extract_global_features(image)
            local_features = self._extract_local_features(image)
            attention_features = self._extract_attention_features(image)
            
            # Project features to same dimension if needed
            feat_dim = global_features.shape[-1]
            
            if local_features.shape[-1] != feat_dim:
                local_projection = torch.nn.Linear(
                    local_features.shape[-1], 
                    feat_dim,
                    device=self.device
                ).float()
                local_features = local_projection(local_features)
            
            if attention_features.shape[-1] != feat_dim:
                attention_projection = torch.nn.Linear(
                    attention_features.shape[-1], 
                    feat_dim,
                    device=self.device
                ).float()
                attention_features = attention_projection(attention_features)
            
            # Combine features using weights
            combined_features = (
                self.feature_weights['global'] * global_features +
                self.feature_weights['local'] * local_features +
                self.feature_weights['attention'] * attention_features
            )
            
            return F.normalize(combined_features, dim=-1)
            
        except Exception as e:
            print(f"Error in feature extraction: {str(e)}")
            # Fallback to global features
            return self._extract_global_features(image)
    
    def update_feature_weights(self, weights: Dict[str, float]) -> None:
        """
        Update the weights for feature combination.
        
        Args:
            weights: Dictionary of feature weights
        """
        if not all(k in self.feature_weights for k in weights.keys()):
            raise ValueError("Invalid feature weight keys")
        
        self.feature_weights.update(weights)
        total = sum(self.feature_weights.values())
        self.feature_weights = {k: v/total for k, v in self.feature_weights.items()}