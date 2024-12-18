import torch
from typing import Dict, List, Tuple
from PIL import Image
from tqdm import tqdm
from utils.augmentation.anomaly_augmenter import AnomalyAugmenter
from collections import defaultdict

class AnomalyDetector:
    def __init__(self, model, threshold: float = 0.2):
        """
        Initialize anomaly detector.
        
        Args:
            model: CLIP model instance
            threshold: Threshold for anomaly detection (default: 0.2)
        """
        self.model = model
        self.threshold = threshold
        self.class_embeddings = None
        self.normal_embeddings = None
        self.anomaly_embeddings = None
        
    def prepare(self, normal_samples: Dict[str, List[str]]) -> None:
        """
        Prepare the detector by computing necessary embeddings.
        
        Args:
            normal_samples: Dictionary containing paths of normal images for each class
        """
        self.class_embeddings, self.normal_embeddings = self._compute_class_embeddings(normal_samples)
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
        normal_embeddings = []
        for class_name, image_paths in tqdm(samples_dict.items(), 
                                          desc="Computing class embeddings"):
            embeddings = []
            for img_path in image_paths:
                try:
                    image = Image.open(img_path).convert('RGB')
                    image_input = self.model.preprocess(image).unsqueeze(0).to(self.model.device)
                    features = self.model.extract_features(image_input)
                    embeddings.append(features)
                    normal_embeddings.append(features)
                except Exception as e:
                    print(f"Error processing {img_path}: {str(e)}")
                    continue
            
            if embeddings:
                class_embedding = torch.mean(torch.stack(embeddings), dim=0)
                class_embeddings[class_name] = class_embedding / class_embedding.norm(dim=-1, keepdim=True)
        
        return class_embeddings, torch.cat(normal_embeddings, dim=0)


    def _generate_anomaly_embeddings(
        self, 
        samples_dict: Dict[str, List[str]], 
        n_anomalies_per_class: int = 5
    ) -> torch.Tensor:
        """
        Generate anomaly embeddings using augmentation.
        
        Args:
            samples_dict: Dictionary of normal sample paths
            n_anomalies_per_class: Number of anomaly samples to generate per class
            
        Returns:
            torch.Tensor: Tensor of anomaly embeddings
        """
        anomaly_embeddings = defaultdict(list)
        augmenter = AnomalyAugmenter(severity=0.4)
        
        for class_name, image_paths in tqdm(samples_dict.items(), 
                                          desc="Generating anomaly embeddings"):
            for img_path in image_paths[:n_anomalies_per_class]:
                try:
                    image = Image.open(img_path).convert('RGB')
                    anomaly_image, augmentations = augmenter.generate_anomaly(image)
                    
                    image_input = self.model.preprocess(anomaly_image).unsqueeze(0).to(self.model.device)
                    features = self.model.extract_features(image_input)
                    for aug in augmentations:
                        anomaly_embeddings[aug].append(features)
                except Exception as e:
                    print(f"Error generating anomaly for {img_path}: {str(e)}")
                    continue
        
        # 정규화된 평균을 저장할 새 딕셔너리
        normalized_abnomaly_embeddings = {}

        # 정규화된 평균 계산 및 저장
        for aug_name, embeddings in anomaly_embeddings.items():
            stacked_embeddings = torch.stack(embeddings)
            class_embedding = torch.mean(stacked_embeddings, dim=0)
            normalized_embedding = class_embedding / class_embedding.norm(dim=-1, keepdim=True)
            normalized_abnomaly_embeddings[aug_name] = normalized_embedding
        if not normalized_abnomaly_embeddings:
            raise ValueError("Failed to generate any anomaly embeddings")
            
        return normalized_abnomaly_embeddings

    def _compute_anomaly_score(self, image_features: torch.Tensor) -> Tuple[float, float, float]:
        """
        Compute anomaly score for given image features.
        image_features: [1,512]
        normal_embedding: [512]
        anomaly_embeddings: [M,512] (M개의 anomaly 이미지 임베딩)
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
            
            max_normal_similarity = sum(sorted(normal_similarities, reverse=True)[:2]) / 2
            print("max_normal_similarity: ", max_normal_similarity)
            # max값이 충분히 크다면 그대로, 작다면 전체 평균
            anomaly_similarities = []
            for aug_embedding in self.anomaly_embeddings.values():
                similarity = torch.cosine_similarity(image_features, aug_embedding)
                anomaly_similarities.append(similarity.item())
                
            if not anomaly_similarities:
                raise ValueError("No anormal similarities computed")
            
            max_anomaly_similarity = max(anomaly_similarities)
            if max_normal_similarity < 0.8:
                max_normal_similarity = torch.cosine_similarity(
                image_features.expand(self.normal_embeddings.shape[0], -1),
                self.normal_embeddings
                ).mean().item()
                max_anomaly_similarity = sum(sorted(anomaly_similarities, reverse=True)[:2]) / 2
            # normal과 augmented를 한번에 비교하여서 선택. 10개중 큰 순으로 5개를 뽑고, anomaly가 3개 이상인 경우 선택
            # 한번에 비교하지만, 10개의 similarity를 softmax함수 형태로 치환. 이후 anomaly에 대한 term을 모두 더해서 0.5 이상일 시 선택.
            # normal과 anomaly 모두 class별 분류 후 max term 추출 후 빼기
            anomaly_score = max_normal_similarity - max_anomaly_similarity
            return anomaly_score, max_normal_similarity, max_anomaly_similarity

        except Exception as e:
            print(f"Error in compute_anomaly_score: {str(e)}")
            return None, None, None