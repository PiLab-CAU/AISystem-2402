import os
import torch
from tqdm import tqdm
from typing import Dict, List, Tuple
from models.clip_model import CLIPModel
from models.anomaly_detector import AnomalyDetector

from utils.data_loader import (
    load_normal_samples,
    load_test_images,
    verify_data_structure,
    load_image
)

from utils.metrics import PerformanceEvaluator
# from utils.visualization import save_predictions
from config import Config

def setup_environment(config: Config) -> Tuple[str, str, str]:
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    
    train_path = config.train_path
    test_path = config.test_path
    
    # Verify data structure
    is_valid, message = verify_data_structure(train_path, test_path)
    if not is_valid:
        raise ValueError(message)
        
    return device, train_path, test_path

def initialize_models(device: str, config: Config) -> Tuple[CLIPModel, AnomalyDetector]:
    """
    Initialize CLIP model and anomaly detector.
    
    Args:
        device: Device to run models on
        config: Configuration object
        
    Returns:
        Tuple[CLIPModel, AnomalyDetector]: Initialized models
    """
    clip_model = CLIPModel(device)
    detector = AnomalyDetector(model=clip_model)  # threshold 파라미터 제거
    
    return clip_model, detector

def process_images(
    detector: AnomalyDetector,
    test_images: Dict[str, List[str]],
    evaluator: PerformanceEvaluator,
    config: Config
) -> None:

    skipped_images = []
    
    for true_label, image_paths in test_images.items():
        for image_path in tqdm(image_paths, desc=f"Processing {true_label} images"):
            try:
                # Load image
                image = load_image(image_path, detector.model.preprocess, detector.model.device)
                if image is None:
                    raise ValueError("Failed to load image")
                    
                # Prediction
                prediction = detector.predict(image)
                if prediction['predicted_label'] == 'error':
                    raise ValueError("Prediction failed")
                    
                # Save results
                evaluator.add_result(true_label, prediction)
                
                # Save visualization
                # if config.save_predictions:
                #     save_predictions(
                #         image_path=image_path,
                #         true_label=true_label,
                #         **prediction,
                #         save_dir=config.results_path
                #     )
                    
            except Exception as e:
                error_msg = f"Error processing image {image_path}: {str(e)}"
                print(error_msg)
                skipped_images.append((image_path, error_msg))
                continue
    
    # Print information about skipped images
    if skipped_images:
        print("\nSkipped images:")
        for img_path, error in skipped_images:
            print(f"- {img_path}: {error}")

def main():
    try:
        # Load configuration
        config = Config()
        
        # Setup environment
        device, train_path, test_path = setup_environment(config)
        
        # Initialize models
        clip_model, detector = initialize_models(device, config)
        
        # Initialize evaluator
        evaluator = PerformanceEvaluator()
        
        # Load data
        print("Loading training samples...")
        normal_samples = load_normal_samples(
            train_path,
            n_samples=config.n_samples
        )
        
        print("Loading test images...")
        test_images = load_test_images(test_path)
        
        # Prepare detector
        print("Preparing anomaly detector...")
        detector.prepare(normal_samples)
        
        # Process test images
        print("Processing test images...")
        process_images(detector, test_images, evaluator, config)
        
        # Print results
        print("\nComputing and displaying metrics...")
        evaluator.print_metrics()
        
        # Save results if enabled
        if config.save_results:
            evaluator.save_metrics(config.results_path)
            
        print(f"\nResults have been saved to: {config.results_path}")
        
    except Exception as e:
        print(f"An error occurred during execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()