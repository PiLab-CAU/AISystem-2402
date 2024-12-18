import torch
from typing import Dict, List, Tuple
from PIL import Image
from tqdm import tqdm
from utils.augmentation.anomaly_augmenter import AnomalyAugmenter
from typing import Dict, List, Tuple
import numpy as np
import torch
from models.clip_model import EnhancedCLIPModel

from typing import Dict, List, Tuple
import torch
import numpy as np
from models.clip_model import EnhancedCLIPModel
from .adaptive_threshold import AdaptiveThresholdManager

class EnhancedEnsembleAnomalyDetector:
    def __init__(
        self, 
        model: EnhancedCLIPModel,
        thresholds: List[float] = [0.15, 0.2, 0.25],
        window_size: int = 100,
        adaptation_rate: float = 0.1
    ):
        """
        Args:
            model: CLIP 모델 인스턴스
            thresholds: 초기 임계값 리스트
            window_size: 적응을 위한 관찰 윈도우 크기
            adaptation_rate: 임계값 조정 속도 (0-1)
        """
        self.model = model
        self.threshold_manager = AdaptiveThresholdManager(
            base_thresholds=thresholds,
            window_size=window_size,
            adaptation_rate=adaptation_rate
        )
        self.class_embeddings = None
        self.anomaly_embeddings = None
        self.weights = None
        self.current_category = None
        # 기존 코드와의 호환성을 위해 thresholds 속성 추가
        self.thresholds = thresholds
    
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
        
        # 현재 카테고리의 임계값 가져오기
        current_thresholds = (self.threshold_manager.get_thresholds(self.current_category) 
                            if self.current_category else self.thresholds)
        
        # 각 임계값별로 스코어 계산
        for threshold, weight in zip(current_thresholds, self.weights):
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
            
            # 현재 카테고리의 임계값 가져오기
            current_thresholds = (self.threshold_manager.get_thresholds(self.current_category) 
                                if self.current_category else self.thresholds)
            
            # 앙상블 기반 최종 판정
            anomaly_votes = sum(1 for threshold in current_thresholds 
                              if score < threshold)
            is_anomaly = anomaly_votes >= len(current_thresholds) / 2
            
            optimal_threshold = (self.threshold_manager.get_optimal_threshold(self.current_category)
                               if self.current_category else np.mean(self.thresholds))
            
            return {
                'predicted_label': 'anomaly' if is_anomaly else 'normal',
                'anomaly_score': float(score),
                'normal_similarity': float(normal_sim),
                'anomaly_similarity': float(anomaly_sim),
                'is_anomaly': is_anomaly,
                'threshold': float(optimal_threshold),
                'anomaly_votes': anomaly_votes,
                'category': self.current_category
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
                'anomaly_votes': 0,
                'category': self.current_category
            }
    
    def prepare(self, normal_samples: Dict[str, List[str]]) -> None:
        """기본 임베딩 준비 및 카테고리 설정"""
        self.class_embeddings = self._compute_class_embeddings(normal_samples)
        self.anomaly_embeddings = self._generate_anomaly_embeddings(normal_samples)
        # 초기 가중치는 동일하게 설정
        self.weights = [1.0 / 3] * 3  # 3개의 임계값 사용
        
        # 첫 번째 카테고리를 현재 카테고리로 설정
        if normal_samples:
            self.current_category = list(normal_samples.keys())[0]
    
    def set_category(self, category: str) -> None:
        """현재 처리할 카테고리 설정"""
        self.current_category = category
    
    def update_threshold(self, score: float, true_label: str) -> None:
        """
        예측 결과를 바탕으로 임계값 업데이트
        
        Args:
            score: 이상 점수
            true_label: 실제 레이블 ('normal' 또는 'anomaly')
        """
        if self.current_category is not None:
            self.threshold_manager.update_scores(
                self.current_category,
                score,
                true_label
            )

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
