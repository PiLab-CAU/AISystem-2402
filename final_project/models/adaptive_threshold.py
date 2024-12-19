import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats
from collections import defaultdict
from scipy.optimize import minimize
from scipy.stats import gaussian_kde

class EnhancedAdaptiveThresholdManager:
    """향상된 적응형 임계값 관리 클래스"""
    
    def __init__(
        self,
        base_thresholds: List[float] = [0.15, 0.2, 0.25],
        window_size: int = 100,
        adaptation_rate: float = 0.1,
        min_samples: int = 20,
        optimization_interval: int = 50  # 최적화 수행 간격
    ):
        self.base_thresholds = base_thresholds
        self.window_size = window_size
        self.adaptation_rate = adaptation_rate
        self.min_samples = min_samples
        self.optimization_interval = optimization_interval
        
        # 카테고리별 임계값 및 히스토리 저장
        self.category_thresholds: Dict[str, List[float]] = {}
        self.score_history: Dict[str, List[float]] = defaultdict(list)
        self.label_history: Dict[str, List[str]] = defaultdict(list)
        
        # 최적화 메트릭 저장
        self.optimization_history: Dict[str, List[float]] = defaultdict(list)
        
    def _compute_otsu_threshold(self, scores: np.ndarray) -> float:
        """Otsu's method를 사용한 임계값 계산"""
        if len(scores) == 0:
            return self.base_thresholds[1]  # 기본값 반환
            
        # 히스토그램 생성
        hist, bin_edges = np.histogram(scores, bins=50)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Otsu's method
        total = len(scores)
        sum_total = sum(hist * bin_centers)
        
        max_variance = 0
        optimal_threshold = bin_centers[0]
        
        weight_1 = 0
        sum_1 = 0
        
        for i, center in enumerate(bin_centers[:-1]):
            weight_1 += hist[i]
            weight_2 = total - weight_1
            
            if weight_1 == 0 or weight_2 == 0:
                continue
                
            sum_1 += hist[i] * center
            mean_1 = sum_1 / weight_1
            mean_2 = (sum_total - sum_1) / weight_2
            
            variance = weight_1 * weight_2 * (mean_1 - mean_2) ** 2
            
            if variance > max_variance:
                max_variance = variance
                optimal_threshold = center
                
        return optimal_threshold
    
    def _optimize_threshold_bayesian(
        self, 
        normal_scores: np.ndarray, 
        anomaly_scores: np.ndarray
    ) -> float:
        """베이지안 최적화를 통한 임계값 최적화"""
        if len(normal_scores) == 0 or len(anomaly_scores) == 0:
            return self.base_thresholds[1]
            
        def objective(threshold):
            # 임계값 기준 분류 성능 계산
            tp = np.sum(anomaly_scores > threshold)
            tn = np.sum(normal_scores <= threshold)
            fp = np.sum(normal_scores > threshold)
            fn = np.sum(anomaly_scores <= threshold)
            
            # F1 점수 계산
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            return -f1  # 최소화 문제로 변환
        
        # 최적화 범위 설정
        bounds = [(min(np.min(normal_scores), np.min(anomaly_scores)), 
                  max(np.max(normal_scores), np.max(anomaly_scores)))]
        
        # 베이지안 최적화 수행
        result = minimize(
            objective,
            x0=np.mean(bounds[0]),
            bounds=bounds,
            method='L-BFGS-B'
        )
        
        return result.x[0]
    
    def _update_category_thresholds(self, category: str) -> None:
        """카테고리별 임계값 업데이트"""
        scores = np.array(self.score_history[category])
        labels = np.array(self.label_history[category])
        
        if len(np.unique(labels)) < 2:
            self.category_thresholds[category] = self.base_thresholds
            return
            
        normal_scores = scores[labels == 'normal']
        anomaly_scores = scores[labels == 'anomaly']
        
        if len(normal_scores) < self.min_samples or len(anomaly_scores) < self.min_samples:
            return
            
        try:
            # Otsu's method로 초기 임계값 계산
            otsu_threshold = self._compute_otsu_threshold(scores)
            
            # 베이지안 최적화로 미세 조정
            optimal_threshold = self._optimize_threshold_bayesian(normal_scores, anomaly_scores)
            
            # 두 방법의 결과를 결합
            final_threshold = (otsu_threshold + optimal_threshold) / 2
            
            # 새로운 임계값 세트 생성 (spread 조정)
            spread = np.std(scores) * 0.2  # 데이터 분포에 따른 동적 spread
            new_thresholds = [
                final_threshold - spread,
                final_threshold,
                final_threshold + spread
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
            
            # 최적화 히스토리 저장
            self.optimization_history[category].append(final_threshold)
            
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
        
        # 일정 간격으로 임계값 최적화
        if len(self.score_history[category]) >= self.min_samples and \
           len(self.score_history[category]) % self.optimization_interval == 0:
            self._update_category_thresholds(category)
    
    def get_thresholds(self, category: str) -> List[float]:
        """카테고리별 현재 임계값 반환"""
        return self.category_thresholds.get(category, self.base_thresholds)
    
    def get_optimal_threshold(self, category: str) -> float:
        """카테고리별 최적 임계값 반환"""
        thresholds = self.get_thresholds(category)
        return thresholds[len(thresholds)//2]
    
    def get_optimization_history(self, category: str) -> List[float]:
        """카테고리별 임계값 최적화 히스토리 반환"""
        return self.optimization_history.get(category, [])