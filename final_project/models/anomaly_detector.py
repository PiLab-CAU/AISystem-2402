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
    def __init__(self, model: CLIPModel, thresholds: List[float] = [0.15, 0.175, 0.2, 0.225, 0.25]):
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
    
    def _compute_adaptive_threshold(self, image_features: torch.Tensor) -> float:
        """
        이미지별 Adaptive Threshold 계산.
        
        Args:
            image_features: 입력 이미지의 특징 벡터
            
        Returns:
            float: Adaptive Threshold 값
        """
        try:
            if self.class_embeddings is None:
                raise ValueError("Class embeddings not initialized. Call prepare() first.")
            
            # 정상 클래스와의 유사도 평균 및 표준 편차 계산
            normal_similarities = []
            for class_embedding in self.class_embeddings.values():
                similarity = torch.cosine_similarity(image_features, class_embedding)
                normal_similarities.append(similarity.item())
            
            if not normal_similarities:
                raise ValueError("No normal similarities computed")
            
            # Adaptive Threshold 계산 (평균 유사도 - 표준 편차)
            adaptive_threshold = max(normal_similarities) - np.std(normal_similarities)
            return adaptive_threshold
            
        except Exception as e:
            print(f"Error computing adaptive threshold: {str(e)}")
            return float(np.mean(self.thresholds))  # 기본 Threshold로 fallback

    def predict(self, image: torch.Tensor) -> Dict:
        """
        이미지에 대한 앙상블 예측 수행 (Adaptive Threshold 적용).
        
        Args:
            image: 입력 이미지 텐서
            
        Returns:
            Dict: 기존 출력 포맷 유지
        """
        try:
            features = self.model.extract_features(image)
            if features is None:
                raise ValueError("Failed to extract features")
                
            # Adaptive Threshold 계산
            adaptive_threshold = self._compute_adaptive_threshold(features)
            
            # 기존 이상 점수 계산
            score, normal_sim, anomaly_sim = self._compute_ensemble_score(features)
            if any(x is None for x in [score, normal_sim, anomaly_sim]):
                raise ValueError("Failed to compute ensemble score")
            
            # Adaptive Threshold와 Static Threshold 비교 후 결정
            final_threshold = min(adaptive_threshold, np.mean(self.thresholds))  # 더 엄격한 Threshold 선택
            is_anomaly = score < final_threshold
            
            return {
                'predicted_label': 'anomaly' if is_anomaly else 'normal',
                'anomaly_score': float(score),
                'normal_similarity': float(normal_sim),
                'anomaly_similarity': float(anomaly_sim),
                'is_anomaly': is_anomaly,
                'threshold': float(np.mean(self.thresholds)),  # 출력은 기존 Static Threshold 사용
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
            }
    
    def update_weights(self, validation_results: List[Dict]) -> None:
        """
        Validation 결과를 기반으로 앙상블 가중치를 최적화.
        
        Args:
            validation_results: Validation 데이터에 대한 예측 결과 리스트
        """
        if not validation_results:
            return
        
        accuracies = []
        for threshold in self.thresholds:
            correct = sum(1 for result in validation_results if (result['anomaly_score'] < threshold) == result['is_anomaly'])
            accuracies.append(correct / len(validation_results))
        
        total_accuracy = sum(accuracies)
        if total_accuracy > 0:
            self.weights = [accuracy / total_accuracy for accuracy in accuracies]
        
        print(f"Updated weights: {self.weights}")

    def _compute_class_embeddings(self, samples_dict: Dict[str, List[str]]) -> Dict[str, torch.Tensor]:
        """
        Normal 클래스의 임베딩을 중간 레이어 출력으로 계산.
        
        Args:
            samples_dict: Normal 샘플 경로의 딕셔너리
            
        Returns:
            Dict[str, torch.Tensor]: 각 클래스의 중간 레이어 기반 임베딩
        """
        class_embeddings = {}
        
        for class_name, image_paths in tqdm(samples_dict.items(), desc="Computing class embeddings"):
            embeddings = []
            for img_path in image_paths:
                try:
                    image = Image.open(img_path).convert('RGB')
                    image_input = self.model.preprocess(image).unsqueeze(0).to(self.model.device)
                    # 중간 레이어에서 특성 추출
                    with torch.no_grad():
                        features = self.model.extract_features(image_input)
                    embeddings.append(features)
                except Exception as e:
                    print(f"Error processing {img_path}: {str(e)}")
                    continue
            
            if embeddings:
                class_embedding = torch.mean(torch.stack(embeddings), dim=0)
                class_embeddings[class_name] = class_embedding / class_embedding.norm(dim=-1, keepdim=True)
        
        return class_embeddings

    def _generate_anomaly_embeddings(self, samples_dict: Dict[str, List[str]], n_anomalies_per_class: int = 3) -> torch.Tensor:
        """
        Augmented 이미지로부터 이상 임베딩 생성 (중간 레이어 활용).
        
        Args:
            samples_dict: Normal 샘플 경로의 딕셔너리
            n_anomalies_per_class: 클래스별 생성할 이상 샘플 수
            
        Returns:
            torch.Tensor: 이상 샘플 임베딩
        """
        anomaly_embeddings = []
        augmenter = AnomalyAugmenter(severity=0.4)
        
        for class_name, image_paths in tqdm(samples_dict.items(), desc="Generating anomaly embeddings"):
            for img_path in image_paths[:n_anomalies_per_class]:
                try:
                    image = Image.open(img_path).convert('RGB')
                    anomaly_image = augmenter.generate_anomaly(image)
                    
                    image_input = self.model.preprocess(anomaly_image).unsqueeze(0).to(self.model.device)
                    # 중간 레이어에서 특성 추출
                    with torch.no_grad():
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
    def optimize_threshold(self, validation_results: List[Dict]) -> None:
        """
        검증 데이터를 기반으로 최적 Threshold를 동적으로 최적화.
        
        Args:
            validation_results: Validation 데이터에 대한 예측 결과 리스트
        """
        best_f1 = 0
        best_threshold = None
        for threshold in np.linspace(0.1, 0.5, num=50):  # Threshold 범위
            tp = sum(1 for result in validation_results if result['anomaly_score'] < threshold and result['is_anomaly'])
            fp = sum(1 for result in validation_results if result['anomaly_score'] < threshold and not result['is_anomaly'])
            fn = sum(1 for result in validation_results if result['anomaly_score'] >= threshold and result['is_anomaly'])
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        self.thresholds = [best_threshold] * len(self.thresholds)  # 최적 Threshold를 모든 탐지기에 설정
        print(f"Optimized threshold: {best_threshold:.2f} with F1 score: {best_f1:.2f}")

