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
        self.normal_prompt = None
        self.anomaly_prompt = None
        self.class_embeddings = None
        self.normal_embeddings = None
        self.anomaly_embeddings = None
        
    def prepare(self, normal_samples: Dict[str, List[str]]) -> None:
        """
        Prepare the detector by computing necessary embeddings.
        
        Args:
            normal_samples: Dictionary containing paths of normal images for each class
        """
        self.class_embeddings, self.normal_embeddings, self.normal_prompt, self.anomaly_prompt = self._compute_class_embeddings(normal_samples)
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
                
            score, normal_sim, anomaly_sim, is_anomaly = self._compute_anomaly_score(features)
            if any(x is None for x in [score, normal_sim, anomaly_sim]):
                raise ValueError("Failed to compute anomaly score")
                
            
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
        class_prompt_embeddings = {}
        a_class_prompt_embeddings = {}
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
            
            for class_name in class_embeddings.keys():
                prompt = f"a photo of a {class_name}"
                a_prompt = f"a photo of a damaged {class_name}"
                text_feat = self.model.extract_text_features([prompt])
                a_text_feat = self.model.extract_text_features([a_prompt])
                class_prompt_embeddings[class_name] = text_feat
                a_class_prompt_embeddings[class_name] = a_text_feat

        return class_embeddings, torch.cat(normal_embeddings, dim=0), class_prompt_embeddings, a_class_prompt_embeddings


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
        anomaly_embeddings = []
        augmenter = AnomalyAugmenter(severity=0.4)
        
        for class_name, image_paths in tqdm(samples_dict.items(), 
                                          desc="Generating anomaly embeddings"):
            for img_path in image_paths[:n_anomalies_per_class]:
                try:
                    image = Image.open(img_path).convert('RGB')
                    anomaly_image, _ = augmenter.generate_anomaly(image)
                    
                    image_input = self.model.preprocess(anomaly_image).unsqueeze(0).to(self.model.device)
                    features = self.model.extract_features(image_input)
                    anomaly_embeddings.append(features)
                except Exception as e:
                    print(f"Error generating anomaly for {img_path}: {str(e)}")
                    continue
        
        if not anomaly_embeddings:
            raise ValueError("Failed to generate any anomaly embeddings")
            
        return torch.cat(anomaly_embeddings, dim=0)

    def _compute_anomaly_score(self, image_features: torch.Tensor) -> Tuple[float, float, float]:
        """
        Compute anomaly score for given image features.
        image_features: [1,512]
        normal_embedding: [M,512]
        anomaly_embeddings: [M,512] (M개의 anomaly 이미지 임베딩)
        prompts: [512]
        """
        try:
            if self.normal_embeddings is None or self.anomaly_embeddings is None:
                raise ValueError("Embeddings not initialized. Call prepare() first.")
            
            image_similarities = []
            for class_embedding in self.class_embeddings.values():
                similarity = torch.cosine_similarity(image_features, class_embedding)
                image_similarities.append(similarity.item())
            if not image_similarities:
                raise ValueError("No image similarities computed")
            
            text_similarities = []
            for prompt_embedding in self.normal_prompt.values():
                similarity = torch.cosine_similarity(image_features, prompt_embedding)
                text_similarities.append(similarity.item())
            if not text_similarities:
                raise ValueError("No text similarities computed")
            
            max_image = max(image_similarities)
            image_class = image_similarities.index(max_image)
            max_text = max(text_similarities)
            text_class = text_similarities.index(max_text)
            print("text: ", max_text)
            if image_class == text_class and max_image + max_text >= 0.9: # Seen Class
                anomaly = list(self.anomaly_prompt.values())[text_class]
                a_max_text = torch.cosine_similarity(image_features, anomaly)
                anomaly_score = max_text - a_max_text
                normal_similarity = max_text
                anomaly_similarity = a_max_text
                anomaly_bool = anomaly_score < 0 or abs(anomaly_score) > 0.3
                
            else: # Unseen Class
                normal_similarities = torch.cosine_similarity(
                    image_features.expand(self.normal_embeddings.shape[0], -1),
                    self.normal_embeddings
                )
                mean_normal_similarity = normal_similarities.mean().item()
            
                anomaly_similarities = torch.cosine_similarity(
                    image_features.expand(self.anomaly_embeddings.shape[0], -1),
                    self.anomaly_embeddings
                )
                mean_anomaly_similarity = anomaly_similarities.mean().item()
                
                anomaly_score = mean_normal_similarity - mean_anomaly_similarity
                normal_similarity = mean_normal_similarity
                anomaly_similarity = mean_anomaly_similarity
                anomaly_bool = anomaly_score < 0.1
            
            return anomaly_score, normal_similarity, anomaly_similarity, anomaly_bool

        except Exception as e:
            print(f"Error in compute_anomaly_score: {str(e)}")
            return None, None, None