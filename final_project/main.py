import os
import torch
from tqdm import tqdm
from typing import Dict, List, Tuple
from models.clip_model import EnhancedCLIPModel
from models.anomaly_detector import EnhancedEnsembleAnomalyDetector

from utils.data_loader import (
    load_normal_samples,
    load_test_images,
    verify_data_structure,
    load_image
)

from utils.metrics import PerformanceEvaluator
from utils.visualization import save_predictions
from config import Config
from utils.seed_utils import set_global_seed

def setup_environment(config: Config) -> Tuple[str, str, str]:
    """
    Setup execution environment and paths.
    
    Args:
        config: Configuration object containing parameters
        
    Returns:
        Tuple[str, str, str]: Device, train path, and test path
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    train_path = config.train_path
    test_path = config.test_path
    
    # Verify data structure
    is_valid, message = verify_data_structure(train_path, test_path)
    if not is_valid:
        raise ValueError(message)
        
    return device, train_path, test_path

def initialize_models(device: str, config: Config):
    clip_model = EnhancedCLIPModel(device)
    detector = EnhancedEnsembleAnomalyDetector(
        model=clip_model,
        thresholds=config.ensemble_thresholds
    )
    return clip_model, detector
def extract_category_from_path(image_path: str) -> str:
    """
    이미지 경로에서 카테고리 추출
    
    Args:
        image_path: 이미지 파일 경로
        
    Returns:
        str: 카테고리 이름
    """
    # 경로 구조에 따라 적절히 수정 필요
    # 예: "path/to/Calculator/normal/image.jpg" -> "Calculator"
    parts = image_path.split(os.sep)
    # 카테고리는 normal/anomaly 폴더의 상위 폴더
    try:
        category_idx = parts.index("normal") - 1
        return parts[category_idx]
    except ValueError:
        try:
            category_idx = parts.index("anomaly") - 1
            return parts[category_idx]
        except ValueError:
            raise ValueError(f"Cannot extract category from path: {image_path}")
def process_images(
    detector: EnhancedEnsembleAnomalyDetector,
    test_images: Dict[str, List[str]],
    evaluator: PerformanceEvaluator,
    config: Config
) -> None:
    """
    Process test images and evaluate results.
    """
    skipped_images = []
    
    for true_label, image_paths in test_images.items():
        for image_path in tqdm(image_paths, desc=f"Processing {true_label} images"):
            try:
                # 이미지의 카테고리 추출 (경로에서)
                category = extract_category_from_path(image_path)
                detector.set_category(category)
                
                # 이미지 로드 및 예측
                image = load_image(image_path, detector.model.preprocess, detector.model.device)
                if image is None:
                    raise ValueError("Failed to load image")
                    
                prediction = detector.predict(image)
                if prediction['predicted_label'] == 'error':
                    raise ValueError("Prediction failed")
                
                # 결과 저장
                evaluator.add_result(true_label, prediction)
                
                # 임계값 업데이트
                detector.update_threshold(
                    prediction['anomaly_score'],
                    true_label
                )
                
                # 예측 결과 시각화 저장
                if config.save_predictions:
                    save_predictions(
                        image_path=image_path,
                        true_label=true_label,
                        **prediction,
                        save_dir=config.results_path
                    )
                    
            except Exception as e:
                error_msg = f"Error processing image {image_path}: {str(e)}"
                print(error_msg)
                skipped_images.append((image_path, error_msg))
                continue

    if skipped_images:
        print("\nSkipped images:")
        for img_path, error in skipped_images:
            print(f"- {img_path}: {error}")
def main():
    set_global_seed(42)
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