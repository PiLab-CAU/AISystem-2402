import torch
import clip
from torch import nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int = 4):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads

        self.qkv = nn.Linear(feature_dim, feature_dim * 3)
        self.proj = nn.Linear(feature_dim, feature_dim)

        # Layer Normalization
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)

        # Feed Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Linear(feature_dim * 4, feature_dim),
            nn.Dropout(0.1)
        )

        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        # Multi-head Self Attention
        shortcut = x
        x = self.norm1(x)

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        x = x + shortcut

        # Feed Forward
        x = x + self.ffn(self.norm2(x))

        return x

class CLIPModel:
    def __init__(self, device: str):
        self.device = device
        self.model, self.preprocess = self._load_clip_model()

        # Initialize self-attention
        self.self_attention = SelfAttention(
            feature_dim=768,  # CLIP ViT-L/14의 feature dimension
            num_heads=4
        ).to(device).float()

        # Layer for feature transformation
        self.feature_transform = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(768, 768)
        ).to(device).float()

        self.temp = 0.07

    def _load_clip_model(self):
        # Set deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        model, preprocess = clip.load('ViT-L/14', self.device)

        # Ensure the model operates in float precision
        model = model.float()
        return model, preprocess

    @torch.no_grad()
    def extract_features(self, image: torch.Tensor) -> torch.Tensor:
        """
        Enhanced feature extraction with self-attention
        """
        # Ensure the image tensor matches the model's precision
        if self.model.dtype == torch.float16:
            image = image.half().to(self.device)
        else:
            image = image.float().to(self.device)

        # Get base features from CLIP
        features = self.model.encode_image(image)
        B = features.shape[0]

        # Reshape for self-attention
        features = features.view(B, 1, -1)  # [B, 1, 768]

        # Apply self-attention
        features = self.self_attention(features)

        # Feature transformation
        features = self.feature_transform(features.squeeze(1))

        # Normalize features
        features = features / features.norm(dim=-1, keepdim=True)

        # Additional feature enhancement
        features = self._enhance_features(features)

        return features

    def _enhance_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        Additional feature enhancement
        """
        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True

        # L2 normalization
        features = features / features.norm(dim=-1, keepdim=True)

        # Global context modeling
        B = features.shape[0]
        context = features.mean(dim=0, keepdim=True).expand(B, -1)

        # Combine with global context
        features = features + 0.1 * context

        # Final normalization
        features = features / features.norm(dim=-1, keepdim=True)

        return features

    def compute_similarity(self, features1: torch.Tensor, features2: torch.Tensor) -> torch.Tensor:
        """
        Compute enhanced similarity between feature vectors
        """
        # Ensure both feature tensors match the model's precision
        if self.model.dtype == torch.float16:
            features1 = features1.half()
            features2 = features2.half()
        else:
            features1 = features1.float()
            features2 = features2.float()

        # Basic cosine similarity
        cos_sim = F.cosine_similarity(features1, features2) / self.temp

        # L2 distance
        l2_dist = torch.norm(features1 - features2, dim=-1)
        l2_sim = 1 / (1 + l2_dist)

        # Combine similarities
        similarity = (cos_sim + l2_sim) / 2

        return similarity
