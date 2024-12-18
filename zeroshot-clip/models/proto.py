import torch
import torch.nn as nn
import torch.nn.functional as F

class PrototypicalNetwork(nn.Module):
    def __init__(self, clip_feature_dim: int, hidden_dim: int = 256):
        """
        Initialize Prototypical Network for feature adaptation.
        
        Args:
            clip_feature_dim: Dimension of CLIP features
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(clip_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of CLIP features
            
        Returns:
            torch.Tensor: Adapted features
        """
        features = self.adapter(x)
        return F.normalize(features, p=2, dim=-1)  # L2 정규화

def contrastive_loss(
    normal_features: torch.Tensor,
    anomaly_features: torch.Tensor,
    temperature: float = 0.1
) -> torch.Tensor:
    """
    Compute contrastive loss between normal and anomaly features.
    
    Args:
        normal_features: Features of normal samples
        anomaly_features: Features of anomaly samples
        temperature: Temperature parameter for scaling
        
    Returns:
        torch.Tensor: Computed loss
    """
    # 정상 샘플들 간의 유사도
    normal_similarity = torch.matmul(normal_features, normal_features.t())
    
    # 정상-비정상 샘플 간의 유사도
    anomaly_similarity = torch.matmul(normal_features, anomaly_features.t())
    
    # 손실 계산
    logits = torch.cat([
        normal_similarity / temperature,
        anomaly_similarity / temperature
    ], dim=1)
    
    labels = torch.zeros(normal_features.size(0), device=normal_features.device)
    loss = F.cross_entropy(logits, labels)
    
    return loss