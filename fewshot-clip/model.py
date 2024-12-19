import torch
import clip
from torch import nn

import subprocess
import sys

def install_clip():
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "git+https://github.com/openai/CLIP.git"])
    except subprocess.CalledProcessError as e:
        print(f"Failed to install CLIP: {e}")
        sys.exit(1)

#install_clip()


class CLIPClassifier(nn.Module):
    def __init__(self, clip_model):
        super(CLIPClassifier, self).__init__()
        self.clip_model = clip_model
        self.fc = nn.Linear(clip_model.visual.output_dim, 1)  # dim: 512 

    def forward(self, images):
        with torch.no_grad():
            features = self.clip_model.encode_image(images)

        features = features.float() # (batch, 512)

        output = self.fc(features)

        return torch.sigmoid(output)  
        
def load_clip_model(device):
    # Load pre-trained CLIP model (ViT-B/32 architecture)
    clip_model, _ = clip.load("ViT-B/32", device=device)
    #print('??', _)
    
    # Initialize and move the classifier model to specified device
    model = CLIPClassifier(clip_model).to(device)

    return model

if __name__ == '__main__':
    clip_model = load_clip_model('cuda:0')
