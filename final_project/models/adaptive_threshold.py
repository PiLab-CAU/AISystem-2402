import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats
from collections import defaultdict
import torch
from sklearn.mixture import GaussianMixture
from scipy.stats import norm

class RobustAdaptiveThresholdManager:
    """클래스 특성을 고려한 강건한 적응형 임계값 관리 클래스"""
    
    def __init__(
        self,
        base_thresholds: List[float] = [0.15, 0.2, 0.25],
        window_size: int = 100,
        adaptation_rate: float = 0.1,
        min_samples: int = 20,
        n_components: int = 2,  # GMM 컴포넌트 수
        warmup_period: int = 50  # 초기 학습 기간
    ):
        self.base_thresholds = base_thresholds
        self.window_size = window_size
        self.adaptation_rate = adaptation_rate
        self.min_samples = min_samples
        self.n_components = n_components
        self.warmup_period = warmup_period
        
        # 카테고리별 데이터 저장
        self.category_thresholds: Dict[str, List[float]] = {}
        self.score_history: Dict[str, List[float]] = defaultdict(list)
        self.label_history: Dict[str, List[str]] = defaultdict(list)
        
        # 클래스별 특성 저장
        self.class_characteristics: Dict[str, Dict] = {}
        self.gmm_models: Dict[str, GaussianMixture] = {}
        
        # 성능 메트릭 저장
        self.performance_history: Dict[str, List[Dict]] = defaultdict(list)
    
    def _initialize_class_characteristics(self, category: str) -> None:
        """클래스별 특성 초기화"""
        if category not in self.class_characteristics:
            self.class_characteristics[category] = {
                'score_mean': None,
                'score_std': None,
                'normal_distribution': None,
                'anomaly_distribution': None,
                'optimal_threshold_history': [],
                'performance_metrics': {
                    'false_positives': [],
                    'false_negatives': [],
                    'precision_history': [],
                    'recall_history': []
                }
            }
    
    def _fit_gmm(self, scores: np.ndarray, category: str) -> None:
        """가우시안 혼합 모델 학습"""
        try:
            if len(scores) >= self.min_samples:
                gmm = GaussianMixture(
                    n_components=self.n_components,
                    covariance_type='full',
                    random_state=42
                )
                scores_reshaped = scores.reshape(-1, 1)
                gmm.fit(scores_reshaped)
                self.gmm_models[category] = gmm
        except Exception as e:
            print(f"GMM fitting error for category {category}: {str(e)}")
    
    def _compute_robust_threshold(
        self, 
        scores: np.ndarray, 
        labels: np.ndarray, 
        category: str
    ) -> float:
        """강건한 임계값 계산"""
        try:
            normal_scores = scores[labels == 'normal']
            anomaly_scores = scores[labels == 'anomaly']
            
            if len(normal_scores) < self.min_samples or len(anomaly_scores) < self.min_samples:
                return self.base_thresholds[1]
            
            # 정상/이상 분포 추정
            normal_kde = stats.gaussian_kde(normal_scores)
            anomaly_kde = stats.gaussian_kde(anomaly_scores)
            
            # GMM 기반 임계값 계산
            if category in self.gmm_models:
                gmm = self.gmm_models[category]
                x = np.linspace(min(scores), max(scores), 1000).reshape(-1, 1)
                gmm_probs = gmm.predict_proba(x)
                gmm_threshold = x[np.argmax(np.abs(np.diff(gmm_probs, axis=1)))]
            else:
                gmm_threshold = np.mean(scores)
            
            # 분포 기반 임계값 계산
            x_range = np.linspace(min(scores), max(scores), 1000)
            normal_pdf = normal_kde(x_range)
            anomaly_pdf = anomaly_kde(x_range)
            
            # 교차점 찾기
            intersections = x_range[np.where(np.diff(np.signbit(normal_pdf - anomaly_pdf)))[0]]
            if len(intersections) > 0:
                dist_threshold = float(np.median(intersections))
            else:
                dist_threshold = np.mean([np.mean(normal_scores), np.mean(anomaly_scores)])
            
            # 클래스 특성 기반 가중치 계산
            if self.class_characteristics[category]['score_mean'] is not None:
                historical_weight = 0.3
                current_weight = 0.7
                historical_mean = self.class_characteristics[category]['score_mean']
                historical_std = self.class_characteristics[category]['score_std']
                
                # z-score 기반 이상치 제거
                z_scores = np.abs((scores - historical_mean) / historical_std)
                valid_scores = scores[z_scores < 3]
                if len(valid_scores) > self.min_samples:
                    current_mean = np.mean(valid_scores)
                else:
                    current_mean = np.mean(scores)
                
                # 가중 평균 계산
                weighted_mean = (historical_mean * historical_weight + 
                               current_mean * current_weight)
            else:
                weighted_mean = np.mean(scores)
            
            # 최종 임계값 계산 (GMM, 분포, 가중평균 결합)
            final_threshold = (0.4 * gmm_threshold + 
                             0.4 * dist_threshold + 
                             0.2 * weighted_mean)
            
            return float(final_threshold)
            
        except Exception as e:
            print(f"Error in robust threshold computation: {str(e)}")
            return self.base_thresholds[1]
    
    def _update_class_characteristics(
        self, 
        category: str, 
        scores: np.ndarray, 
        labels: np.ndarray
    ) -> None:
        """클래스 특성 업데이트"""
        try:
            normal_scores = scores[labels == 'normal']
            anomaly_scores = scores[labels == 'anomaly']
            
            if len(normal_scores) >= self.min_samples:
                # 통계치 업데이트
                self.class_characteristics[category]['score_mean'] = np.mean(normal_scores)
                self.class_characteristics[category]['score_std'] = np.std(normal_scores)
                
                # 분포 업데이트
                self.class_characteristics[category]['normal_distribution'] = \
                    stats.gaussian_kde(normal_scores)
                
                if len(anomaly_scores) >= self.min_samples:
                    self.class_characteristics[category]['anomaly_distribution'] = \
                        stats.gaussian_kde(anomaly_scores)
                
        except Exception as e:
            print(f"Error updating class characteristics: {str(e)}")
    
    def _evaluate_performance(
        self, 
        category: str, 
        threshold: float, 
        scores: np.ndarray, 
        labels: np.ndarray
    ) -> None:
        """성능 평가 및 기록"""
        try:
            predictions = scores > threshold
            true_labels = labels == 'anomaly'
            
            tp = np.sum((predictions == True) & (true_labels == True))
            fp = np.sum((predictions == True) & (true_labels == False))
            fn = np.sum((predictions == False) & (true_labels == True))
            tn = np.sum((predictions == False) & (true_labels == False))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            self.class_characteristics[category]['performance_metrics']['false_positives'].append(fp)
            self.class_characteristics[category]['performance_metrics']['false_negatives'].append(fn)
            self.class_characteristics[category]['performance_metrics']['precision_history'].append(precision)
            self.class_characteristics[category]['performance_metrics']['recall_history'].append(recall)
            
        except Exception as e:
            print(f"Error evaluating performance: {str(e)}")
    
    def _update_category_thresholds(self, category: str) -> None:
        """카테고리별 임계값 업데이트"""
        scores = np.array(self.score_history[category])
        labels = np.array(self.label_history[category])
        
        if len(np.unique(labels)) < 2:
            self.category_thresholds[category] = self.base_thresholds
            return
        
        try:
            # 클래스 특성 초기화 및 업데이트
            self._initialize_class_characteristics(category)
            self._update_class_characteristics(category, scores, labels)
            
            # GMM 모델 학습
            self._fit_gmm(scores, category)
            
            # 강건한 임계값 계산
            optimal_threshold = self._compute_robust_threshold(scores, labels, category)
            
            # 성능 평가
            self._evaluate_performance(category, optimal_threshold, scores, labels)
            
            # 임계값 범위 설정
            spread = self.class_characteristics[category]['score_std'] \
                    if self.class_characteristics[category]['score_std'] is not None \
                    else np.std(scores)
            spread = min(spread * 0.5, 0.1)  # 범위 제한
            
            new_thresholds = [
                optimal_threshold - spread,
                optimal_threshold,
                optimal_threshold + spread
            ]
            
            # 점진적 업데이트
            if category in self.category_thresholds:
                current_thresholds = self.category_thresholds[category]
                updated_thresholds = [
                    current * (1 - self.adaptation_rate) + new * self.adaptation_rate
                    for current, new in zip(current_thresholds, new_thresholds)
                ]
            else:
                updated_thresholds = new_thresholds
            
            self.category_thresholds[category] = updated_thresholds
            
            # 최적 임계값 히스토리 업데이트
            self.class_characteristics[category]['optimal_threshold_history'].append(
                optimal_threshold
            )
            
        except Exception as e:
            print(f"Error updating thresholds for category {category}: {str(e)}")
            self.category_thresholds[category] = self.base_thresholds
    
    def update_scores(
        self, 
        category: str, 
        score: float, 
        true_label: str
    ) -> None:
        """새로운 스코어 추가 및 히스토리 관리"""
        self.score_history[category].append(score)
        self.label_history[category].append(true_label)
        
        if len(self.score_history[category]) > self.window_size:
            self.score_history[category].pop(0)
            self.label_history[category].pop(0)
        
        # 초기 학습 기간 이후 임계값 업데이트
        if len(self.score_history[category]) >= self.min_samples and \
           len(self.score_history[category]) >= self.warmup_period:
            self._update_category_thresholds(category)
    
    def get_thresholds(self, category: str) -> List[float]:
        """카테고리별 현재 임계값 반환"""
        return self.category_thresholds.get(category, self.base_thresholds)
    
    def get_optimal_threshold(self, category: str) -> float:
        """카테고리별 최적 임계값 반환"""
        thresholds = self.get_thresholds(category)
        return thresholds[len(thresholds)//2]
    
    def get_class_performance(self, category: str) -> Dict:
        """클래스별 성능 메트릭 반환"""
        if category in self.class_characteristics:
            return self.class_characteristics[category]['performance_metrics']
        return None