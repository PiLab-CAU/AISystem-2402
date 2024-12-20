import torch
from typing import Dict, List, Tuple
from PIL import Image
from tqdm import tqdm
from utils.augmentation.anomaly_augmenter import AnomalyAugmenter

class AnomalyDetector:
    def __init__(self, model, threshold: float = 0.3):
        """
        Initialize anomaly detector.
        
        Args:
            model: CLIP model instance
            threshold: Threshold for anomaly detection (default: 0.2)
        """
        self.model = model
        self.threshold = threshold
        self.class_embeddings = None
        self.anomaly_embeddings = None
        
    def prepare(self, normal_samples: Dict[str, List[str]]) -> None:
        """
        Prepare the detector by computing necessary embeddings.
        
        Args:
            normal_samples: Dictionary containing paths of normal images for each class
        """
        self.class_embeddings = self._compute_class_embeddings(normal_samples)
        self.anomaly_embeddings = self._generate_anomaly_embeddings(normal_samples)

    def predict(self, image: torch.Tensor) -> Dict:
        """
        Predict whether an image is anomalous.
        
        Args:
            image: Input image tensor
            
        Returns:
            Dict: Prediction results including predicted label and scores
        """
        try:
            features = self.model.extract_features(image)
            if features is None:
                raise ValueError("Failed to extract features from image")
                
            score, normal_sim, anomaly_sim = self._compute_anomaly_score(features)
            if any(x is None for x in [score, normal_sim, anomaly_sim]):
                raise ValueError("Failed to compute anomaly score")
                
            is_anomaly = score < self.threshold
            
            return {
                'predicted_label': 'anomaly' if is_anomaly else 'normal',
                'anomaly_score': float(score),
                'normal_similarity': float(normal_sim),
                'anomaly_similarity': float(anomaly_sim),
                'is_anomaly': is_anomaly,
                'threshold': float(self.threshold)
            }
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            # Return default values in case of error
            return {
                'predicted_label': 'error',
                'anomaly_score': 0.0,
                'normal_similarity': 0.0,
                'anomaly_similarity': 0.0,
                'is_anomaly': True,
                'threshold': float(self.threshold)
            }

    def _compute_class_embeddings(
        self, 
        samples_dict: Dict[str, List[str]]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute embeddings for each normal class.
        
        Args:
            samples_dict: Dictionary of normal sample paths
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary of class embeddings
        """
        class_embeddings = {}
        
        for class_name, image_paths in tqdm(samples_dict.items(), 
                                          desc="Computing class embeddings"):
            embeddings = []
            for img_path in image_paths:
                try:
                    image = Image.open(img_path).convert('RGB')
                    image_input = self.model.preprocess(image).unsqueeze(0).to(self.model.device)
                    features = self.model.extract_features(image_input)
                    embeddings.append(features)
                except Exception as e:
                    print(f"Error processing {img_path}: {str(e)}")
                    continue
            
            if embeddings:
                class_embedding = torch.mean(torch.stack(embeddings), dim=0)
                class_embeddings[class_name] = class_embedding / class_embedding.norm(dim=-1, keepdim=True)
        
        return class_embeddings

    def _generate_anomaly_embeddings(
        self, 
        samples_dict: Dict[str, List[str]], 
        n_anomalies_per_class: int = 3
    ) -> torch.Tensor:
        """
        Generate anomaly embeddings using augmentation.
        
        Args:
            samples_dict: Dictionary of normal sample paths
            n_anomalies_per_class: Number of anomaly samples to generate per class
            
        Returns:
            torch.Tensor: Tensor of anomaly embeddings
        """
        anomaly_embeddings = []
        augmenter = AnomalyAugmenter(severity=0.4)
        
        for class_name, image_paths in tqdm(samples_dict.items(), 
                                          desc="Generating anomaly embeddings"):
            for img_path in image_paths[:n_anomalies_per_class]:
                try:
                    image = Image.open(img_path).convert('RGB')
                    anomaly_image = augmenter.generate_anomaly(image)
                    
                    image_input = self.model.preprocess(anomaly_image).unsqueeze(0).to(self.model.device)
                    features = self.model.extract_features(image_input)
                    anomaly_embeddings.append(features)
                except Exception as e:
                    print(f"Error generating anomaly for {img_path}: {str(e)}")
                    continue
        
        if not anomaly_embeddings:
            raise ValueError("Failed to generate any anomaly embeddings")
            
        return torch.cat(anomaly_embeddings, dim=0)

    def _compute_anomaly_score(
        self, 
        image_features: torch.Tensor
    ) -> Tuple[float, float, float]:
        """
        Compute anomaly score for given image features.
        
        Args:
            image_features: Extracted image features
            
        Returns:
            Tuple[float, float, float]: Anomaly score, normal similarity, and anomaly similarity
        """
        try:
            if self.class_embeddings is None or self.anomaly_embeddings is None:
                raise ValueError("Embeddings not initialized. Call prepare() first.")
                
            normal_similarities = []
            for class_embedding in self.class_embeddings.values():
                similarity = torch.cosine_similarity(image_features, class_embedding)
                normal_similarities.append(similarity.item())
                
            if not normal_similarities:
                raise ValueError("No normal similarities computed")
                
            max_normal_similarity = max(normal_similarities)
            
            anomaly_similarities = torch.cosine_similarity(
                image_features.expand(self.anomaly_embeddings.shape[0], -1),
                self.anomaly_embeddings
            )
            mean_anomaly_similarity = anomaly_similarities.mean().item()
            
            anomaly_score = max_normal_similarity - mean_anomaly_similarity
            return anomaly_score, max_normal_similarity, mean_anomaly_similarity
            
        except Exception as e:
            print(f"Error in compute_anomaly_score: {str(e)}")
            return None, None, None