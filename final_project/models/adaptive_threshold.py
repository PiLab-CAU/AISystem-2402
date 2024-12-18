# adaptive_threshold.py

import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from scipy import stats
from collections import defaultdict

class AdaptiveThresholdManager:
    """적응형 임계값을 관리하는 클래스"""
    
    def __init__(
        self,
        base_thresholds: List[float] = [0.15, 0.2, 0.25],
        window_size: int = 100,
        adaptation_rate: float = 0.1,
        min_samples: int = 20
    ):
        """
        Args:
            base_thresholds: 기본 임계값 리스트
            window_size: 적응을 위한 관찰 윈도우 크기
            adaptation_rate: 임계값 조정 속도 (0-1)
            min_samples: 적응형 임계값 계산에 필요한 최소 샘플 수
        """
        self.base_thresholds = base_thresholds
        self.window_size = window_size
        self.adaptation_rate = adaptation_rate
        self.min_samples = min_samples
        
        # 카테고리별 임계값 및 히스토리 저장
        self.category_thresholds: Dict[str, List[float]] = {}
        self.score_history: Dict[str, List[float]] = defaultdict(list)
        self.label_history: Dict[str, List[str]] = defaultdict(list)
        
    def update_scores(
        self, 
        category: str, 
        score: float, 
        true_label: str
    ) -> None:
        """
        새로운 스코어 추가 및 히스토리 관리
        
        Args:
            category: 이미지 카테고리
            score: 이상 점수
            true_label: 실제 레이블 ('normal' 또는 'anomaly')
        """
        # 스코어 및 레이블 히스토리 업데이트
        self.score_history[category].append(score)
        self.label_history[category].append(true_label)
        
        # 윈도우 크기 유지
        if len(self.score_history[category]) > self.window_size:
            self.score_history[category].pop(0)
            self.label_history[category].pop(0)
        
        # 충분한 샘플이 모이면 임계값 업데이트
        if len(self.score_history[category]) >= self.min_samples:
            self._update_category_thresholds(category)
    
    def _update_category_thresholds(self, category: str) -> None:
        """
        카테고리별 임계값 업데이트
        
        Args:
            category: 업데이트할 카테고리
        """
        scores = np.array(self.score_history[category])
        labels = np.array(self.label_history[category])
        
        if len(np.unique(labels)) < 2:
            # 레이블이 한 종류만 있으면 기본 임계값 사용
            self.category_thresholds[category] = self.base_thresholds
            return
        
        # 정상과 이상 샘플의 스코어 분리
        normal_scores = scores[labels == 'normal']
        anomaly_scores = scores[labels == 'anomaly']
        
        if len(normal_scores) < self.min_samples or len(anomaly_scores) < self.min_samples:
            return
            
        try:
            # 정상과 이상 분포의 교차점 찾기
            normal_kde = stats.gaussian_kde(normal_scores)
            anomaly_kde = stats.gaussian_kde(anomaly_scores)
            
            x = np.linspace(min(scores), max(scores), 1000)
            normal_pdf = normal_kde(x)
            anomaly_pdf = anomaly_kde(x)
            
            # 두 분포의 교차점 찾기
            intersections = x[np.where(np.diff(np.signbit(normal_pdf - anomaly_pdf)))[0]]
            
            if len(intersections) > 0:
                optimal_threshold = float(np.median(intersections))
                
                # 새로운 임계값 세트 생성
                spread = 0.05  # 임계값 간격
                new_thresholds = [
                    optimal_threshold - spread,
                    optimal_threshold,
                    optimal_threshold + spread
                ]
                
                # 점진적으로 임계값 업데이트
                if category in self.category_thresholds:
                    current_thresholds = self.category_thresholds[category]
                    updated_thresholds = [
                        current * (1 - self.adaptation_rate) + new * self.adaptation_rate
                        for current, new in zip(current_thresholds, new_thresholds)
                    ]
                else:
                    updated_thresholds = new_thresholds
                
                self.category_thresholds[category] = updated_thresholds
                
        except Exception as e:
            print(f"Error updating thresholds for category {category}: {str(e)}")
            # 에러 발생시 기본 임계값 사용
            self.category_thresholds[category] = self.base_thresholds
    
    def get_thresholds(self, category: str) -> List[float]:
        """
        카테고리별 현재 임계값 반환
        
        Args:
            category: 이미지 카테고리
            
        Returns:
            List[float]: 임계값 리스트
        """
        return self.category_thresholds.get(category, self.base_thresholds)
    
    def get_optimal_threshold(self, category: str) -> float:
        """
        카테고리별 최적 임계값 반환 (중간 임계값 사용)
        
        Args:
            category: 이미지 카테고리
            
        Returns:
            float: 최적 임계값
        """
        thresholds = self.get_thresholds(category)
        return thresholds[len(thresholds)//2]