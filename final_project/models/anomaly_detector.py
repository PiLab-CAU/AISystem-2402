import torch
from typing import Dict, List, Tuple
from PIL import Image
from tqdm import tqdm
from utils.augmentation.anomaly_augmenter import AnomalyAugmenter
from torchvision import transforms

import numpy as np
from scipy.stats import norm

class AnomalyDetector:
    def __init__(self, model):
        """
        Initialize anomaly detector.
        
        Args:
            model: CLIP model instance
            threshold: Threshold for anomaly detection (default: 0.2)
        """
        self.model = model
        self.threshold = None #threshold
        self.class_embeddings = None
        self.anomaly_embeddings = None
        
        self.confidence_level = 0.05
        self.score_mean = None
        self.score_std = None
        
    def prepare(self, normal_samples: Dict[str, List[str]]) -> None:
        """
        Prepare the detector by computing necessary embeddings.
        
        Args:
            normal_samples: Dictionary containing paths of normal images for each class
        """
        self.class_embeddings = self._compute_class_embeddings(normal_samples)
        self.anomaly_embeddings = self._generate_anomaly_embeddings(normal_samples)
        
        scores = []
        for class_name, image_paths in normal_samples.items():
            for img_path in image_paths:
                try:
                    image = Image.open(img_path).convert('RGB')
                    image_input = self.model.preprocess(image).unsqueeze(0).to(self.model.device)
                    features = self.model.extract_features(image_input)
                    
                    similarities = []
                    for class_embedding in self.class_embeddings.values():
                        similarity = torch.cosine_similarity(features, class_embedding)
                        similarities.append(similarity.item())
                    
                    score = max(similarities)
                    scores.append(score)
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {str(e)}")
                    continue
        self.score_mean = np.mean(scores)
        self.score_std = np.std(scores)
        self.threshold = norm.ppf(self.confidence_level, self.score_mean, self.score_std)
        print(f"뚜레쉬홀드: {self.threshold:.3f} (민: {self.score_mean:.3f}, 스탠다드디스트리뷰션: {self.score_std:.3f})")

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
            
            # 모든 클래스와의 유사도 계산
            similarities = []
            for class_embedding in self.class_embeddings.values():
                similarity = torch.cosine_similarity(features, class_embedding)
                similarities.append(similarity.item())
            
            # 최대 유사도를 스코어로 사용
            max_similarity = max(similarities)
            
            # 가우시안 분포 기반 예측
            is_anomaly = max_similarity < self.threshold
            return {
                'predicted_label': 'anomaly' if is_anomaly else 'normal',
                'anomaly_score': float(max_similarity),
                'normal_similarity': float(max_similarity),
                'anomaly_similarity': 0.0,  # 이전 버전과의 호환성을 위해 유지
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
        augmenter = AnomalyAugmenter(severity=0.6)
        
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
    
    def _compute_adaptive_threshold(self, embeddings: torch.Tensor) -> float:
        """
        클래스별 적응형 임계값을 계산합니다.
        
        Args:
            embeddings: 클래스의 임베딩 텐서
            
        Returns:
            float: 계산된 적응형 임계값
        """
        # 임베딩 간의 유사도 행렬 계산
        similarities = torch.matmul(embeddings, embeddings.T)
        # 평균과 표준편차를 사용하여 임계값 계산
        threshold = float(similarities.mean() - similarities.std())
        return threshold


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
            
            # 각 클래스별 임계값 계산
            class_thresholds = {
                class_name: self._compute_adaptive_threshold(embeddings)
                for class_name, embeddings in self.class_embeddings.items()
            }
            
            # 각 클래스별 유사도 및 임계값 기반 점수 계산
            normal_similarities = []
            threshold_scores = []
            
            normal_similarities = []
            for class_embedding in self.class_embeddings.values():
                similarity = torch.cosine_similarity(image_features, class_embedding)
                normal_similarities.append(similarity.item())
            
            max_normal_similarity = max(normal_similarities)
            
            anomaly_similarities = torch.cosine_similarity(
                image_features.expand(self.anomaly_embeddings.shape[0], -1),
                self.anomaly_embeddings
            )
            mean_anomaly_similarity = anomaly_similarities.mean().item()
            
            # 점수는 단순히 normal과 anomaly의 유사도 차이로 계산
            score = max_normal_similarity - mean_anomaly_similarity
            
            return score, max_normal_similarity, mean_anomaly_similarity
            
        except Exception as e:
            print(f"Error in compute_anomaly_score: {str(e)}")
            return None, None, None