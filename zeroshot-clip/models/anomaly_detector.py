import torch
from typing import Dict, List
from PIL import Image
from tqdm import tqdm
from utils.augmentation.anomaly_augmenter import AnomalyAugmenter

class AnomalyDetector:
    def __init__(self, model, threshold: float = 0.5):
        """
        Initialize anomaly detector.
        
        Args:
            model: CLIP model instance
            threshold: Initial threshold for anomaly detection (default: 0.5)
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
        self._optimize_threshold()

    def _optimize_threshold(self) -> None:
        """Optimize threshold using synthetic anomalies"""
        thresholds = torch.linspace(0.0, 1.0, 100)
        best_f1 = 0
        best_threshold = self.threshold

        # Prepare validation features
        normal_features = torch.stack(list(self.class_embeddings.values()))
        anomaly_features = self.anomaly_embeddings

        for threshold in thresholds:
            # Calculate scores
            normal_scores = self._compute_scores(normal_features)
            anomaly_scores = self._compute_scores(anomaly_features)

            # Apply threshold
            normal_preds = normal_scores > threshold
            anomaly_preds = anomaly_scores > threshold

            # Calculate metrics
            tp = (~anomaly_preds).sum().item()
            fp = (~normal_preds).sum().item()
            fn = anomaly_preds.sum().item()

            # Calculate F1 score
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        self.threshold = float(best_threshold)
        print(f"Optimized threshold: {self.threshold:.3f} (F1: {best_f1:.3f})")

    def _compute_scores(self, features: torch.Tensor) -> torch.Tensor:
        """Compute anomaly scores for given features"""
        # Normalize features
        if len(features.shape) == 1:
            features = features.unsqueeze(0)
        features = features / features.norm(dim=-1, keepdim=True)

        # Calculate similarity with normal class prototypes
        normal_sims = []
        for class_emb in self.class_embeddings.values():
            sim = torch.cosine_similarity(features, class_emb.unsqueeze(0), dim=-1)
            normal_sims.append(sim)
        max_normal_sim = torch.stack(normal_sims).max(dim=0)[0]

        # Calculate similarity with anomaly embeddings
        anomaly_sims = torch.cosine_similarity(
            features.unsqueeze(1),
            self.anomaly_embeddings.unsqueeze(0),
            dim=-1
        )
        # Use average of top-3 anomaly similarities
        k = min(3, anomaly_sims.shape[1])
        top_k_anomaly_sim = torch.topk(anomaly_sims, k, dim=1)[0].mean(dim=1)

        # Compute final score: higher score = more normal
        return max_normal_sim - 0.5 * top_k_anomaly_sim

    def predict(self, image: torch.Tensor) -> Dict:
        """Predict whether an image is anomalous"""
        try:
            features = self.model.extract_features(image)
            if features is None:
                raise ValueError("Failed to extract features")

            # Compute score and similarities
            score = self._compute_scores(features)
            
            # Get similarities for logging
            normal_sims = []
            for class_emb in self.class_embeddings.values():
                sim = torch.cosine_similarity(features, class_emb.unsqueeze(0), dim=-1)
                normal_sims.append(sim)
            max_normal_sim = torch.stack(normal_sims).max()

            anomaly_sims = torch.cosine_similarity(
                features.unsqueeze(1),
                self.anomaly_embeddings.unsqueeze(0),
                dim=-1
            )
            mean_anomaly_sim = anomaly_sims.mean()

            # Make prediction
            is_anomaly = score < self.threshold

            return {
                'predicted_label': 'anomaly' if is_anomaly else 'normal',
                'anomaly_score': float(score),
                'normal_similarity': float(max_normal_sim),
                'anomaly_similarity': float(mean_anomaly_sim),
                'is_anomaly': bool(is_anomaly),
                'threshold': float(self.threshold)
            }

        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return {
                'predicted_label': 'error',
                'anomaly_score': 0.0,
                'normal_similarity': 0.0,
                'anomaly_similarity': 0.0,
                'is_anomaly': True,
                'threshold': float(self.threshold)
            }

    def _compute_class_embeddings(self, samples_dict: Dict[str, List[str]]) -> Dict[str, torch.Tensor]:
        """Compute embeddings for each normal class"""
        class_embeddings = {}
        
        for class_name, image_paths in tqdm(samples_dict.items(), desc="Computing class embeddings"):
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
                # Average embeddings and normalize
                class_embedding = torch.stack(embeddings).mean(dim=0)
                class_embedding = class_embedding / class_embedding.norm(dim=-1, keepdim=True)
                class_embeddings[class_name] = class_embedding
        
        return class_embeddings

    def _generate_anomaly_embeddings(self, samples_dict: Dict[str, List[str]]) -> torch.Tensor:
        """Generate anomaly embeddings using augmentation"""
        anomaly_embeddings = []
        augmenter = AnomalyAugmenter(severity=0.5)
        
        for image_paths in tqdm(samples_dict.values(), desc="Generating anomaly embeddings"):
            for img_path in image_paths[:5]:  # Use first 5 images from each class
                try:
                    image = Image.open(img_path).convert('RGB')
                    # Generate multiple anomalies per image
                    for _ in range(2):
                        anomaly_image = augmenter.generate_anomaly(image)
                        image_input = self.model.preprocess(anomaly_image).unsqueeze(0).to(self.model.device)
                        features = self.model.extract_features(image_input)
                        anomaly_embeddings.append(features)
                except Exception as e:
                    print(f"Error generating anomaly for {img_path}: {str(e)}")
                    continue
        
        if not anomaly_embeddings:
            raise ValueError("Failed to generate any anomaly embeddings")
            
        embeddings = torch.stack(anomaly_embeddings).squeeze(1)
        return embeddings / embeddings.norm(dim=-1, keepdim=True)