import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance
import numpy as np
from typing import Dict, List, Tuple
from tqdm import tqdm
from utils.augmentation.anomaly_augmenter import AnomalyAugmenter
import random

# Set random seeds for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
random.seed(42)

class AnomalyDetector:
    def __init__(self, model):
        self.model = model
        self.threshold = None
        self.reference_embeddings = None
        self.anomaly_embeddings = None

    def prepare(self, normal_samples: Dict[str, List[str]]) -> None:
        print("Computing reference embeddings...")
        self.reference_embeddings = self._compute_reference_embeddings(normal_samples)

        print("Generating anomaly embeddings...")
        self.anomaly_embeddings = self._generate_anomaly_embeddings(normal_samples)

        print("Finding optimal threshold...")
        validation_scores = self._generate_validation_scores(normal_samples)
        self.threshold = self._find_optimal_threshold(validation_scores)
        print(f"Optimal threshold: {self.threshold:.3f}")

    def predict(self, image: torch.Tensor) -> Dict:
        try:
            image = image.to(self.model.device).float()
            if self.model.model.dtype == torch.float16:
                image = image.half()

            features = self.model.extract_features(image)

            if features is None:
                raise ValueError("Failed to extract features from image")

            score, normal_sim, anomaly_sim = self._compute_anomaly_score(features)
            is_anomaly = score > self.threshold

            return {
                'predicted_label': 'anomaly' if is_anomaly else 'normal',
                'anomaly_score': float(score),
                'normal_similarity': float(normal_sim),
                'anomaly_similarity': float(anomaly_sim),
                'is_anomaly': bool(is_anomaly),
                'threshold': float(self.threshold)
            }

        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return {
                'predicted_label': 'error',
                'anomaly_score': 1.0,
                'normal_similarity': 0.0,
                'anomaly_similarity': 1.0,
                'is_anomaly': True,
                'threshold': 0.5
            }

    def _apply_normal_augmentation(self, image: Image.Image) -> Image.Image:
        try:
            img = image.copy()
            
            # Brightness
            if random.random() < 0.8:
                enhancer = ImageEnhance.Brightness(img)
                factor = random.uniform(0.7, 1.3)
                img = enhancer.enhance(factor)
            
            # Contrast
            if random.random() < 0.8:
                enhancer = ImageEnhance.Contrast(img)
                factor = random.uniform(0.7, 1.3)
                img = enhancer.enhance(factor)
            
            # Color
            if random.random() < 0.5:
                enhancer = ImageEnhance.Color(img)
                factor = random.uniform(0.8, 1.2)
                img = enhancer.enhance(factor)
            
            # Random rotation
            if random.random() < 0.6:
                angle = random.uniform(-15, 15)
                img = img.rotate(angle, Image.BILINEAR, expand=False)
            
            # Random horizontal flip
            if random.random() < 0.3:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            
            # Random crop and resize
            if random.random() < 0.4:
                width, height = img.size
                crop_size = min(width, height)
                crop_size = int(crop_size * random.uniform(0.8, 1.0))
                left = random.randint(0, width - crop_size)
                top = random.randint(0, height - crop_size)
                img = img.crop((left, top, left + crop_size, top + crop_size))
                img = img.resize((width, height), Image.BILINEAR)
                
            return img
        except Exception as e:
            print(f"Error in normal augmentation: {str(e)}")
            return image
    
    def _generate_anomaly_embeddings(self, samples_dict: Dict[str, List[str]], max_embeddings: int = 300) -> torch.Tensor:
        """
        Generate limited number of anomaly embeddings using augmentation
        """
        print("Starting anomaly embeddings generation...")
        anomaly_embeddings = []
        augmenter = AnomalyAugmenter(severity=0.65)
        
        batch_images = []
        batch_size = 8  # 배치 크기 감소
        
        all_image_paths = [path for paths in samples_dict.values() for path in paths]
        samples_per_class = 5  # 클래스당 사용할 이미지 수
        augmentations_per_sample = 2  # 이미지당 augmentation 수 감소
        
        # 전체 샘플 수 제한
        total_samples = min(
            len(all_image_paths), 
            samples_per_class * len(samples_dict)
        )
        
        with tqdm(total=total_samples, desc="Generating anomaly embeddings") as pbar:
            for img_path in all_image_paths[:total_samples]:
                try:
                    image = Image.open(img_path).convert('RGB')
                    
                    # 각 이미지당 2개의 anomaly 생성
                    for _ in range(augmentations_per_sample):
                        if len(anomaly_embeddings) * batch_size >= max_embeddings:
                            break
                            
                        anomaly_image = augmenter.generate_anomaly(image.copy())
                        if anomaly_image is not None:
                            batch_images.append(
                                self.model.preprocess(anomaly_image)
                            )
                            
                            if len(batch_images) >= batch_size:
                                batch = torch.stack(batch_images).to(self.model.device).float()
                                features = self.model.extract_features(batch)
                                anomaly_embeddings.append(features)
                                batch_images.clear()  # 메모리 효율을 위해 clear 사용
                                
                except Exception as e:
                    print(f"Error processing {img_path}: {str(e)}")
                    continue
                    
                pbar.update(1)
        
        # 남은 배치 처리
        if batch_images:
            batch = torch.stack(batch_images).to(self.model.device).float()
            features = self.model.extract_features(batch)
            anomaly_embeddings.append(features)
        
        # 모든 임베딩 결합
        all_embeddings = torch.cat(anomaly_embeddings, dim=0)
        
        # 최대 개수로 제한
        if len(all_embeddings) > max_embeddings:
            all_embeddings = all_embeddings[:max_embeddings]
        
        return F.normalize(all_embeddings, dim=1)

    def _compute_reference_embeddings(self, samples_dict: Dict[str, List[str]]) -> torch.Tensor:
        all_embeddings = []
        
        for image_paths in tqdm(samples_dict.values(), desc="Computing reference embeddings"):
            for img_path in image_paths:
                try:
                    image = Image.open(img_path).convert('RGB')
                    image_input = self.model.preprocess(image).unsqueeze(0).to(self.model.device).float()
                    if self.model.model.dtype == torch.float16:
                        image_input = image_input.half()

                    features = self.model.extract_features(image_input)
                    all_embeddings.append(features)

                except Exception as e:
                    print(f"Error processing {img_path}: {str(e)}")
                    continue

        if all_embeddings:
            stacked_embeddings = torch.cat(all_embeddings, dim=0)
            return F.normalize(torch.mean(stacked_embeddings, dim=0), dim=-1)
        
        raise ValueError("No valid embeddings computed")

    def _compute_final_score(self, normal_similarity: float, anomaly_similarity: float) -> float:
        # 정상 샘플과의 유사도에 더 큰 가중치 부여
        weighted_normal = normal_similarity * 1.2
        weighted_anomaly = anomaly_similarity * 0.8
        
        # 점수 계산 방식 개선
        diff_score = weighted_normal - weighted_anomaly
        
        # Sigmoid 함수의 스케일링 팩터 조정
        final_score = torch.sigmoid(torch.tensor(diff_score * 0.1)).item()
        
        # 점수 범위를 0.2-0.8로 확장
        final_score = 0.2 + (final_score * 0.6)
        
        return float(final_score)

    def _compute_anomaly_score(self, image_features: torch.Tensor) -> Tuple[float, float, float]:
        try:
            device = self.model.device
            # 모든 텐서를 동일한 device로 이동
            image_features = image_features.to(device)
            image_features = F.normalize(image_features, dim=-1).float()

            normal_similarity = self._compute_normal_similarity(image_features)
            anomaly_similarity = self._compute_anomaly_similarity(image_features)

            final_score = self._compute_final_score(
                normal_similarity,
                anomaly_similarity
            )

            return final_score, normal_similarity, anomaly_similarity

        except Exception as e:
            print(f"Error in compute_anomaly_score: {str(e)}")
            return None, None, None

    def _compute_normal_similarity(self, image_features: torch.Tensor) -> float:
        device = self.model.device
        # 모든 텐서를 동일한 device로 이동
        reference_embedding = self.reference_embeddings.to(device)
        image_features = image_features.to(device)
        
        # 정규화
        reference_embedding = F.normalize(reference_embedding, dim=-1).float()
        image_features = F.normalize(image_features, dim=-1).float()
        
        cos_sim = F.cosine_similarity(image_features, reference_embedding.unsqueeze(0))
        l2_dist = torch.cdist(image_features, reference_embedding.unsqueeze(0))
        l2_sim = torch.exp(-l2_dist.squeeze() * 2.0).to(device)
        
        combined_sim = (cos_sim * 0.8 + l2_sim * 0.2).to(device)
        scale_factor = 15.0
        combined_sim = combined_sim * scale_factor
        
        return combined_sim.item()

    def _compute_anomaly_similarity(self, image_features: torch.Tensor) -> float:
        device = self.model.device
        # 모든 텐서를 동일한 device로 이동
        image_features = image_features.to(device)
        anomaly_embeddings = self.anomaly_embeddings.to(device)
        
        # 정규화
        image_features = F.normalize(image_features, dim=-1).float()
        anomaly_embeddings = F.normalize(anomaly_embeddings, dim=-1)
        
        cos_similarities = F.cosine_similarity(
            image_features.unsqueeze(1),
            anomaly_embeddings.unsqueeze(0),
            dim=2
        ).squeeze(0)
        
        k = min(5, len(cos_similarities))
        top_k_sims = torch.topk(cos_similarities, k)[0]
        
        weights = torch.softmax(torch.arange(k, 0, -1).float().to(device), dim=0)
        weighted_sim = torch.sum(top_k_sims * weights) * 15.0
        
        return weighted_sim.item()

    def _find_optimal_threshold(self, scores: Dict[str, List[float]]) -> float:
        if not scores['normal'] or not scores['anomaly']:
            return 0.5
        
        normal_scores = np.array(scores['normal'])
        anomaly_scores = np.array(scores['anomaly'])
        
        normal_mean = np.mean(normal_scores)
        normal_std = np.std(normal_scores)
        anomaly_mean = np.mean(anomaly_scores)
        anomaly_std = np.std(anomaly_scores)
        
        overlap = min(normal_mean + normal_std, anomaly_mean + anomaly_std) - \
                 max(normal_mean - normal_std, anomaly_mean - anomaly_std)
                 
        if overlap <= 0:
            threshold = (normal_mean + anomaly_mean) / 2
        else:
            threshold = normal_mean + (0.5 * normal_std)
        
        threshold = np.clip(threshold, 0.45, 0.55)
        
        return float(threshold)

    def prepare(self, normal_samples: Dict[str, List[str]]) -> None:
        print("Computing reference embeddings...")
        self.reference_embeddings = self._compute_reference_embeddings(normal_samples)
        self.reference_embeddings = self.reference_embeddings.to(self.model.device)

        print("Generating anomaly embeddings...")
        self.anomaly_embeddings = self._generate_anomaly_embeddings(normal_samples)
        self.anomaly_embeddings = self.anomaly_embeddings.to(self.model.device)

        print("Finding optimal threshold...")
        validation_scores = self._generate_validation_scores(normal_samples)
        self.threshold = self._find_optimal_threshold(validation_scores)
        print(f"Optimal threshold: {self.threshold:.3f}")

    def _generate_validation_scores(self, normal_samples: Dict[str, List[str]]) -> Dict[str, List[float]]:
        scores = {'normal': [], 'anomaly': []}
        augmenter = AnomalyAugmenter(severity=0.8)
        
        max_samples = 20
        n_normal_augs = 8
        n_anomaly_augs = 8
        
        for image_paths in normal_samples.values():
            for sample_path in image_paths[:max_samples]:
                try:
                    image = Image.open(sample_path).convert('RGB')
                    
                    # 원본 이미지
                    image_input = self.model.preprocess(image).unsqueeze(0).to(self.model.device).float()
                    features = self.model.extract_features(image_input)
                    score, _, _ = self._compute_anomaly_score(features)
                    if score is not None:
                        scores['normal'].append(score)
                    
                    # 정상 증강
                    for strength in np.linspace(0.3, 1.0, n_normal_augs):
                        normal_aug = self._apply_normal_augmentation(image)
                        aug_input = self.model.preprocess(normal_aug).unsqueeze(0).to(self.model.device).float()
                        aug_features = self.model.extract_features(aug_input)
                        aug_score, _, _ = self._compute_anomaly_score(aug_features)
                        if aug_score is not None:
                            scores['normal'].append(aug_score)
                    
                    # 이상 증강
                    for severity in np.linspace(0.6, 1.0, n_anomaly_augs):
                        augmenter.severity = severity
                        anomaly_image = augmenter.generate_anomaly(image)
                        if anomaly_image is not None:
                            anomaly_input = self.model.preprocess(anomaly_image).unsqueeze(0).to(self.model.device).float()
                            anomaly_features = self.model.extract_features(anomaly_input)
                            anomaly_score, _, _ = self._compute_anomaly_score(anomaly_features)
                            if anomaly_score is not None:
                                scores['anomaly'].append(anomaly_score)
                                
                except Exception as e:
                    print(f"Error processing validation sample {sample_path}: {str(e)}")
                    continue
        
        return scores