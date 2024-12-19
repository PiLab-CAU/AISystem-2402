import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

import random
import math
import numpy as np

import augmentation_type





    
    



class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None, normal_transform=None, mode="train"):
        self.root_dir = root_dir
        self.mode = mode

        print('\n---------------------')
        
        print(f"\nInitializing dataset from: {root_dir}")
        print(f"Mode: {mode}")
        
        if not os.path.exists(root_dir):
            raise ValueError(f"Directory does not exist: {root_dir}")
            
        print(f"Directory contents: {os.listdir(root_dir)}")
        
        if mode == "test":
            self.transform = None
            self.normal_transform = transform
        else:
            self.transform = transform
            self.normal_transform = normal_transform
            
        self.image_paths = []
        
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith('.jpg'):
                    image_path = os.path.join(root, file)
                    self.image_paths.append(image_path)
                
        print(f"Found {len(self.image_paths)} images")


        if len(self.image_paths) > 0:
            print(f"Sample image path: {self.image_paths[0]}")

###############  len  ######################

    def __len__(self):
        if self.mode == "test" or not self.transform:
            return len(self.image_paths)
        return len(self.image_paths) * 3
    
###############  get_item  ######################

    def __getitem__(self, idx):
        num_original_images = len(self.image_paths)

        
                 
        if self.mode == "test" or idx < num_original_images:
            image_path = self.image_paths[idx % num_original_images]
            image = Image.open(image_path).convert("RGB")
            
            if self.normal_transform:
                image = self.normal_transform(image)
            
            # In test mode, determine label based on directory name
            if self.mode == "test":
                label = 0 if "normal" in image_path.lower() else 1
            else:
                label = 0  # Original images are normal in training mode
            
            return image, label
        
        else:  # Transformed images in training mode

            augmentation_type = idx % num_original_images
            image_path = self.image_paths[augmentation_type]
            image = Image.open(image_path).convert("RGB")
            
            if self.transform: # anomaly ? 
                if augmentation_type == 0:
                    image = self.transform.transform(image=image, augmentation_type=0)

                if augmentation_type == 1:
                    image = self.transform.transform(image=image, augmentation_type=1)

                if augmentation_type == 2:
                    image = self.transform.transform(image=image, augmentation_type=2)
            
            return image, 1  # transformed image is anomaly
        

        

def get_data_loader(train_dir, test_dir, batch_size):
    print(f'@@@@@@@@@@@@ Batch: {batch_size}')
    print(f"\nCurrent working directory: {os.getcwd()}")
    print(f"Checking train directory: {train_dir}")
    print(f"Checking test directory: {test_dir}")
    
    if not os.path.exists(train_dir):
        raise ValueError(f"Training directory does not exist: {train_dir}")
    if not os.path.exists(test_dir):
        raise ValueError(f"Test directory does not exist: {test_dir}")
    
    normal_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    """
    anomaly_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ColorJitter(brightness=0.5),
        transforms.ToTensor()
    ])
    """

    anomaly_transform = augmentation_type.anomaly_transform()

    
    train_dataset = CustomDataset(
        train_dir, 
        transform=anomaly_transform,
        normal_transform=normal_transform,
        mode="train"
    )
    
    test_dataset = CustomDataset(
        test_dir, 
        transform=normal_transform,
        mode="test"
    )

    if len(train_dataset) == 0:
        raise ValueError(f"No images found in training directory: {train_dir}")
    if len(test_dataset) == 0:
        raise ValueError(f"No images found in test directory: {test_dir}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


