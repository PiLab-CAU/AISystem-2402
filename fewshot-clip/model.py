import torch
import clip
from torch import nn

class CLIPClassifier(nn.Module):
    def __init__(self, clip_model):
        super(CLIPClassifier, self).__init__()
        self.clip_model = clip_model
        # Output neuron count is 1 for binary classification
        self.fc = nn.Linear(clip_model.visual.output_dim, 1)  

    def forward(self, images):
        # Extract image features using CLIP model without gradient computation
        with torch.no_grad():
            features = self.clip_model.encode_image(images)
        # Convert features to float type
        features = features.float()
        # Pass features through the fully connected layer
        output = self.fc(features)
        # Apply sigmoid activation to get probability output (0-1 range)
        return torch.sigmoid(output)  
        
def load_clip_model(device):
    # Load pre-trained CLIP model (ViT-B/32 architecture)
    clip_model, _ = clip.load("ViT-B/32", device=device)
    
    # Initialize and move the classifier model to specified device
    model = CLIPClassifier(clip_model).to(device)

    return model