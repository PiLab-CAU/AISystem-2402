import os
from PIL import Image
from typing import Dict, List, Tuple
import torch
import random

def load_normal_samples(train_path: str, n_samples: int = 5) -> Dict[str, List[str]]:
    """
    Load few-shot samples of normal data.
    
    Args:
        train_path: Path to training data directory
        n_samples: Number of samples to use per class
        
    Returns:
        Dict[str, List[str]]: Dictionary of image paths by class
    """
    normal_samples = {}
    
    for category_name in os.listdir(train_path):
        category_path = os.path.join(train_path, category_name)
        
        if os.path.isdir(category_path):
            image_files = [f for f in os.listdir(category_path) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            selected_files = random.sample(image_files, min(n_samples, len(image_files)))
            normal_samples[category_name] = [os.path.join(category_path, img) 
                                            for img in selected_files]
            
    return normal_samples


def load_test_images(test_path: str) -> Dict[str, List[str]]:
    """
    Load paths of test images.
    
    Args:
        test_path: Path to test data directory
        
    Returns:
        Dict[str, List[str]]: Dictionary of normal and anomaly image paths
    """
    test_images = {'normal': [], 'anomaly': []}
    
    for label in ['normal', 'anomaly']:
        label_path = os.path.join(test_path, label)
        
        if os.path.isdir(label_path):
            image_files = [os.path.join(label_path, f) for f in os.listdir(label_path)
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            test_images[label].extend(image_files)
            
    return test_images


def load_image(image_path: str, preprocess: callable, device: str) -> torch.Tensor:
    """
    Load and preprocess image for model input.
    
    Args:
        image_path: Path to image file
        preprocess: Preprocessing function
        device: Device to use ('cuda' or 'cpu')
        
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    try:
        image = Image.open(image_path).convert('RGB')
        image_input = preprocess(image).unsqueeze(0).to(device)
        return image_input
        
    except Exception as e:
        raise Exception(f"Error loading image {image_path}: {str(e)}")


def verify_data_structure(train_path: str, test_path: str) -> Tuple[bool, str]:
    """
    Verify the correctness of data structure.
    
    Args:
        train_path: Path to training data
        test_path: Path to test data
        
    Returns:
        Tuple[bool, str]: Validation result and error message
    """
    # Check path existence
    if not os.path.exists(train_path):
        return False, f"Train path does not exist: {train_path}"
        
    if not os.path.exists(test_path):
        return False, f"Test path does not exist: {test_path}"
    
    # Check training data structure
    train_classes = [d for d in os.listdir(train_path)
                    if os.path.isdir(os.path.join(train_path, d))]
                    
    if not train_classes:
        return False, f"No class directories found in train path: {train_path}"
    
    # Check for images in each class directory
    for category_name in train_classes:
        category_path = os.path.join(train_path, category_name)
        images = [f for f in os.listdir(category_path)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    
        if not images:
            return False, f"No images found in class directory: {category_path}"
    
    # Check test data structure
    for label in ['normal', 'anomaly']:
        label_path = os.path.join(test_path, label)
        
        if not os.path.exists(label_path):
            return False, f"Missing directory for {label} images in test path: {test_path}"
            
        test_images = [f for f in os.listdir(label_path)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                        
        if not test_images:
            return False, f"No {label} images found in directory: {label_path}"
    
    return True, "Data structure is valid"


def get_class_info(train_path: str) -> Dict[str, int]:
    """
    Get number of images per class.
    
    Args:
        train_path: Path to training data
        
    Returns:
        Dict[str, int]: Dictionary of image counts per class
    """
    class_info = {}
    
    for category_name in os.listdir(train_path):
        category_path = os.path.join(train_path, category_name)
        
        if os.path.isdir(category_path):
            images = [f for f in os.listdir(category_path)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            class_info[category_name] = len(images)
            
    return class_info
