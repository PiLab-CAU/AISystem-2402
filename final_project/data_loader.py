import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

class CustomAnomalyTransform:
    def __init__(self, p=0.5, size_range=(0.02, 0.4)):
        self.p = p
        self.size_range = size_range
        
    def __call__(self, img):
        if random.random() > self.p:
            return img
            
        h, w = img.size
        area = h * w
        target_area = random.uniform(self.size_range[0], self.size_range[1]) * area
        aspect_ratio = random.uniform(0.3, 1/0.3)
        
        cut_h = int(round(math.sqrt(target_area * aspect_ratio)))
        cut_w = int(round(math.sqrt(target_area / aspect_ratio)))
        
        y = random.randint(0, h - cut_h)
        x = random.randint(0, w - cut_w)
        
        img = np.array(img)
        img[y:y+cut_h, x:x+cut_w] = 0
        return Image.fromarray(img)

def get_data_loader(train_dir, test_dir, batch_size=4):
    print(f"\nCurrent working directory: {os.getcwd()}")
    print(f"Checking train directory: {train_dir}")
    print(f"Checking test directory: {test_dir}")
    
    if not os.path.exists(train_dir):
        raise ValueError(f"Training directory does not exist: {train_dir}")
    if not os.path.exists(test_dir):
        raise ValueError(f"Test directory does not exist: {test_dir}")
    
    # Preprocessing normal images
    normal_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    # Preprocessing anomaly images
    anomaly_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        CustomAnomalyTransform(p=1.0),  # Apply Cutout
        transforms.ToTensor()
    ])

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None, normal_transform=None, mode="train"):
        self.root_dir = root_dir
        self.mode = mode
        
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

    def __len__(self):
        if self.mode == "test" or not self.transform:
            return len(self.image_paths)
        return len(self.image_paths) * 2

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
            image_path = self.image_paths[idx - num_original_images]
            image = Image.open(image_path).convert("RGB")
            
            if self.transform:
                image = self.transform(image)
            
            return image, 1  # transformed image is anomaly

def get_data_loader(train_dir, test_dir, batch_size=4):
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
    
    anomaly_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ColorJitter(brightness=0.5),
        transforms.ToTensor()
    ])
    
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