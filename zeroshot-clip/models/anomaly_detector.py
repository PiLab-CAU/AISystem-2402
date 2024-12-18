import torch
from typing import Dict, List, Tuple
from PIL import Image
from tqdm import tqdm
import numpy as np
from sklearn.mixture import GaussianMixture
from utils.augmentation.anomaly_augmenter import AnomalyAugmenter

class AnomalyDetector:
    def __init__(self, model, threshold: float = 0.2):
        """
        Initialize anomaly detector with class-specific thresholds.
        
        Args:
            model: CLIP model instance
            base_threshold: Default threshold for initialization
        """
        self.model = model
        self.base_threshold = threshold
        self.class_thresholds = {}  # Dictionary to store class-specific thresholds
        self.class_embeddings = None
        self.anomaly_embeddings = {}  # Dictionary to store class-specific anomaly embeddings
        
    def prepare(self, normal_samples: Dict[str, List[str]]) -> None:
        """
        Prepare the detector by computing class-specific embeddings and thresholds
        """
        print("Computing class embeddings...")
        self.class_embeddings = self._compute_class_embeddings(normal_samples)
        
        print("Generating class-specific anomaly embeddings...")
        for class_name in normal_samples.keys():
            print(f"Processing class: {class_name}")
            self.anomaly_embeddings[class_name] = self._generate_anomaly_embeddings(
                {class_name: normal_samples[class_name]})
        
        print("Optimizing class-specific thresholds...")
        self._optimize_class_thresholds(normal_samples)
        
        # Print optimized thresholds for each class
        for class_name, threshold in self.class_thresholds.items():
            print(f"Optimized threshold for {class_name}: {threshold:.4f}")

    def _optimize_class_thresholds(self, normal_samples: Dict[str, List[str]]) -> None:
        """
        Optimize threshold for each class separately
        """
        for class_name, image_paths in normal_samples.items():
            normal_scores = []
            augmented_scores = []
            augmenter = AnomalyAugmenter(severity=0.4)
            
            # Collect scores for normal and augmented samples
            for img_path in image_paths:
                try:
                    # Normal image scores
                    image = Image.open(img_path).convert('RGB')
                    image_input = self.model.preprocess(image).unsqueeze(0).to(self.model.device)
                    features = self.model.extract_features(image_input)
                    score = self._compute_class_score(features, class_name)
                    if score is not None:
                        normal_scores.append(score)
                    
                    # Generate multiple augmented versions
                    for _ in range(3):
                        augmented_image = augmenter.generate_anomaly(image)
                        aug_input = self.model.preprocess(augmented_image).unsqueeze(0).to(self.model.device)
                        aug_features = self.model.extract_features(aug_input)
                        aug_score = self._compute_class_score(aug_features, class_name)
                        if aug_score is not None:
                            augmented_scores.append(aug_score)
                            
                except Exception as e:
                    continue
            
            # Use GMM to find optimal threshold for this class
            combined_scores = np.array(normal_scores + augmented_scores)
            if len(combined_scores) > 2:  # Need at least 2 samples for GMM
                gmm = GaussianMixture(n_components=2, random_state=42)
                gmm.fit(combined_scores.reshape(-1, 1))
                
                # Use the intersection point of the two Gaussians as threshold
                means = gmm.means_.flatten()
                threshold = np.mean(means)
            else:
                # Fallback to base threshold if not enough samples
                threshold = self.base_threshold
                
            self.class_thresholds[class_name] = float(threshold)

    def predict(self, image: torch.Tensor) -> Dict:
        """
        Predict whether an image is anomalous using class-specific thresholds
        """
        try:
            features = self.model.extract_features(image)
            if features is None:
                raise ValueError("Failed to extract features from image")
            
            # Compute scores for each class
            class_scores = {}
            normal_similarities = []
            anomaly_similarities = []
            
            for class_name in self.class_thresholds.keys():
                # Get class embedding
                class_embedding = self.class_embeddings[class_name]
                
                # Compute normal similarity
                normal_sim = torch.cosine_similarity(features, class_embedding).item()
                normal_similarities.append(normal_sim)
                
                # Compute anomaly similarity
                class_anomaly_embeddings = self.anomaly_embeddings[class_name]
                anom_sims = torch.cosine_similarity(
                    features.expand(class_anomaly_embeddings.shape[0], -1),
                    class_anomaly_embeddings
                )
                mean_anom_sim = anom_sims.mean().item()
                anomaly_similarities.append(mean_anom_sim)
                
                # Compute class score
                score = self._compute_class_score(features, class_name)
                class_scores[class_name] = score
            
            # Find best matching class
            best_class = max(class_scores.items(), key=lambda x: x[1])[0]
            best_score = class_scores[best_class]
            
            # Use class-specific threshold
            threshold = self.class_thresholds[best_class]
            is_anomaly = best_score < threshold
            
            # Get max similarities
            max_normal_sim = max(normal_similarities)
            mean_anomaly_sim = sum(anomaly_similarities) / len(anomaly_similarities)
            
            return {
                'predicted_label': 'anomaly' if is_anomaly else 'normal',
                'predicted_class': best_class,
                'anomaly_score': float(best_score),
                'normal_similarity': float(max_normal_sim),
                'anomaly_similarity': float(mean_anomaly_sim),
                'class_scores': {k: float(v) for k, v in class_scores.items()},
                'threshold': float(threshold),
                'is_anomaly': is_anomaly
            }
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return {
                'predicted_label': 'error',
                'predicted_class': None,
                'anomaly_score': 0.0,
                'normal_similarity': 0.0,
                'anomaly_similarity': 0.0,
                'class_scores': {},
                'threshold': 0.0,
                'is_anomaly': True
            }

    def _compute_class_score(
        self,
        image_features: torch.Tensor,
        class_name: str
    ) -> float:
        """
        Compute anomaly score for a specific class
        """
        # Get class embedding
        class_embedding = self.class_embeddings[class_name]
        
        # Compute similarities
        cosine_sim = torch.cosine_similarity(image_features, class_embedding)
        euc_dist = torch.norm(image_features - class_embedding, dim=-1)
        
        # Get class-specific anomaly embeddings
        class_anomaly_embeddings = self.anomaly_embeddings[class_name]
        
        # Compute similarity with anomaly embeddings
        anomaly_similarities = torch.cosine_similarity(
            image_features.expand(class_anomaly_embeddings.shape[0], -1),
            class_anomaly_embeddings
        )
        mean_anomaly_similarity = anomaly_similarities.mean().item()
        
        # Normalize distance
        max_dist = 2.0  # Maximum possible L2 distance for normalized vectors
        normalized_dist = euc_dist.item() / max_dist
        
        # Compute final score
        normal_score = (cosine_sim.item() * 0.7 + (1 - normalized_dist) * 0.3)
        anomaly_penalty = (mean_anomaly_similarity * 0.5)
        combined_score = normal_score - anomaly_penalty
        
        return combined_score

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
        Generate anomaly embeddings using augmentation with fixed random seed.
        
        Args:
            samples_dict: Dictionary of normal sample paths
            n_anomalies_per_class: Number of anomaly samples to generate per class
            
        Returns:
            torch.Tensor: Tensor of anomaly embeddings
        """
        torch.manual_seed(42)
        np.random.seed(42)
        
        anomaly_embeddings = []
        augmenter = AnomalyAugmenter(severity=0.4)
        
        for class_name, image_paths in sorted(samples_dict.items()):
            print(f"Generating anomalies for class: {class_name}")
            
            selected_paths = sorted(image_paths)[:n_anomalies_per_class]
            
            for img_path in tqdm(selected_paths, desc=f"Processing {class_name}"):
                try:
                    img_seed = hash(img_path) % (2**32)
                    torch.manual_seed(img_seed)
                    np.random.seed(img_seed)
                    
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
        
        torch.manual_seed(42)
        return torch.cat(anomaly_embeddings, dim=0)

'''
    def _compute_anomaly_score(self, image_features: torch.Tensor) -> Tuple[float, float, float]:
        # 코사인 유사도와 유클리디안 거리를 모두 활용
        cosine_similarities = []
        euclidean_distances = []
        
        for class_embedding in self.class_embeddings.values():
            cos_sim = torch.cosine_similarity(image_features, class_embedding)
            cosine_similarities.append(cos_sim.item())
            
            euc_dist = torch.norm(image_features - class_embedding, dim=-1)
            euclidean_distances.append(euc_dist.item())
        
        # 최대 코사인 유사도와 최소 유클리디안 거리 사용
        max_cosine = max(cosine_similarities)
        min_distance = min(euclidean_distances)
        
        # 거리 정규화
        max_dist = max(euclidean_distances)
        min_dist = min(euclidean_distances)
        normalized_dist = (min_distance - min_dist) / (max_dist - min_dist + 1e-6)
        
        # 이상치 클래스와의 유사도
        anomaly_similarities = torch.cosine_similarity(
            image_features.expand(self.anomaly_embeddings.shape[0], -1),
            self.anomaly_embeddings
        )
        mean_anomaly_similarity = anomaly_similarities.mean().item()
        
        # 개선된 점수 계산
        normal_score = (max_cosine * 0.7 + (1 - normalized_dist) * 0.3)
        anomaly_penalty = (mean_anomaly_similarity * 0.5)
        combined_score = normal_score - anomaly_penalty
        
        return combined_score, max_cosine, mean_anomaly_similarity
'''