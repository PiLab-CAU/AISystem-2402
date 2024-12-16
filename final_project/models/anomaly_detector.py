import torch
from typing import Dict, List, Tuple
from PIL import Image
from tqdm import tqdm
from utils.augmentation.anomaly_augmenter import AnomalyAugmenter
from typing import Dict, List, Tuple
import numpy as np
import torch
from models.clip_model import CLIPModel

class EnsembleAnomalyDetector:
    def __init__(self, model: CLIPModel, thresholds: List[float] = [0.15, 0.2, 0.25]):
        """
        여러 임계값과 특성을 가진 detector들의 앙상블
        
        Args:
            model: CLIP 모델 인스턴스
            thresholds: 각 detector의 임계값 리스트
        """
        self.model = model
        self.thresholds = thresholds
        self.class_embeddings = None
        self.anomaly_embeddings = None
        self.weights = None
        
    def prepare(self, normal_samples: Dict[str, List[str]]) -> None:
        """기본 임베딩 준비"""
        self.class_embeddings = self._compute_class_embeddings(normal_samples)
        self.anomaly_embeddings = self._generate_anomaly_embeddings(normal_samples)
        # 초기 가중치는 동일하게 설정
        self.weights = [1.0 / len(self.thresholds)] * len(self.thresholds)

    def _compute_ensemble_score(
        self, 
        image_features: torch.Tensor
    ) -> Tuple[float, float, float]:
        """
        앙상블 스코어 계산
        
        Args:
            image_features: 입력 이미지의 특징 벡터
            
        Returns:
            Tuple[float, float, float]: 최종 이상 점수, 정상 유사도, 이상 유사도
        """
        ensemble_scores = []
        ensemble_normal_sims = []
        ensemble_anomaly_sims = []
        
        # 각 임계값별로 스코어 계산
        for threshold, weight in zip(self.thresholds, self.weights):
            try:
                normal_similarities = []
                for class_embedding in self.class_embeddings.values():
                    similarity = torch.cosine_similarity(
                        image_features, 
                        class_embedding
                    )
                    normal_similarities.append(similarity.item())
                
                max_normal_similarity = max(normal_similarities)
                
                anomaly_similarities = torch.cosine_similarity(
                    image_features.expand(self.anomaly_embeddings.shape[0], -1),
                    self.anomaly_embeddings
                )
                mean_anomaly_similarity = anomaly_similarities.mean().item()
                
                # 임계값 기반 스코어 계산
                anomaly_score = max_normal_similarity - mean_anomaly_similarity
                
                ensemble_scores.append(anomaly_score * weight)
                ensemble_normal_sims.append(max_normal_similarity * weight)
                ensemble_anomaly_sims.append(mean_anomaly_similarity * weight)
                
            except Exception as e:
                print(f"Error in ensemble score computation: {str(e)}")
                continue
        
        if not ensemble_scores:
            return None, None, None
            
        # 최종 앙상블 스코어 계산
        final_score = sum(ensemble_scores)
        final_normal_sim = sum(ensemble_normal_sims)
        final_anomaly_sim = sum(ensemble_anomaly_sims)
        
        return final_score, final_normal_sim, final_anomaly_sim

    def predict(self, image: torch.Tensor) -> Dict:
        """
        이미지에 대한 앙상블 예측 수행
        
        Args:
            image: 입력 이미지 텐서
            
        Returns:
            Dict: 예측 결과 딕셔너리
        """
        try:
            features = self.model.extract_features(image)
            if features is None:
                raise ValueError("Failed to extract features")
                
            score, normal_sim, anomaly_sim = self._compute_ensemble_score(features)
            if any(x is None for x in [score, normal_sim, anomaly_sim]):
                raise ValueError("Failed to compute ensemble score")
            
            # 앙상블 기반 최종 판정
            # 각 detector의 판정을 종합하여 최종 결정
            anomaly_votes = sum(1 for threshold in self.thresholds 
                              if score < threshold)
            is_anomaly = anomaly_votes >= len(self.thresholds) / 2
            
            return {
                'predicted_label': 'anomaly' if is_anomaly else 'normal',
                'anomaly_score': float(score),
                'normal_similarity': float(normal_sim),
                'anomaly_similarity': float(anomaly_sim),
                'is_anomaly': is_anomaly,
                'threshold': float(np.mean(self.thresholds)),  # 평균 임계값
                'anomaly_votes': anomaly_votes
            }
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return {
                'predicted_label': 'error',
                'anomaly_score': 0.0,
                'normal_similarity': 0.0,
                'anomaly_similarity': 0.0,
                'is_anomaly': True,
                'threshold': float(np.mean(self.thresholds)),
                'anomaly_votes': 0
            }
    
    def update_weights(self, validation_results: List[Dict]) -> None:
        """
        검증 결과를 기반으로 가중치 업데이트
        
        Args:
            validation_results: 검증 데이터에 대한 예측 결과 리스트
        """
        if not validation_results:
            return
            
        accuracies = []
        for threshold in self.thresholds:
            correct = sum(1 for result in validation_results 
                        if (result['anomaly_score'] < threshold) == result['is_anomaly'])
            accuracy = correct / len(validation_results)
            accuracies.append(accuracy)
        
        # 정확도 기반 가중치 계산
        total_accuracy = sum(accuracies)
        if total_accuracy > 0:
            self.weights = [acc / total_accuracy for acc in accuracies]

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
