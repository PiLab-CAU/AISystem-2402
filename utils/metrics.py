import numpy as np
from typing import Dict
import json
import os
from datetime import datetime

class PerformanceEvaluator:
    def __init__(self):
        """Initialize the evaluator with empty results list"""
        self.results = []
    
    def add_result(self, true_label: str, prediction: Dict) -> None:
        """
        Add a prediction result to the evaluator.
        
        Args:
            true_label: Ground truth label
            prediction: Dictionary containing prediction results
        """
        self.results.append({**prediction, 'true_label': true_label})
    
    def compute_metrics(self) -> Dict:
        """
        Calculate detailed performance metrics.
        
        Returns:
            Dict: Dictionary containing various performance metrics
        """
        metrics = {
            'total': len(self.results),
            'correct': sum(1 for r in self.results 
                            if r['predicted_label'] == r['true_label']),
            'normal': {
                'total': sum(1 for r in self.results 
                            if r['true_label'] == 'normal'),
                'correct': sum(1 for r in self.results 
                                if r['true_label'] == 'normal' 
                                and not r['is_anomaly']),
                'incorrect': sum(1 for r in self.results 
                                if r['true_label'] == 'normal' 
                                and r['is_anomaly'])
            },
            'anomaly': {
                'total': sum(1 for r in self.results 
                            if r['true_label'] == 'anomaly'),
                'correct': sum(1 for r in self.results 
                                if r['true_label'] == 'anomaly' 
                                and r['is_anomaly']),
                'incorrect': sum(1 for r in self.results 
                                if r['true_label'] == 'anomaly' 
                                and not r['is_anomaly'])
            }
        }
        
        # Calculate accuracies
        metrics['accuracy'] = (metrics['correct'] / metrics['total'] 
                                if metrics['total'] > 0 else 0)
        metrics['normal']['accuracy'] = (metrics['normal']['correct'] / 
                                        metrics['normal']['total'] 
                                        if metrics['normal']['total'] > 0 else 0)
        metrics['anomaly']['accuracy'] = (metrics['anomaly']['correct'] / 
                                        metrics['anomaly']['total']
                                        if metrics['anomaly']['total'] > 0 else 0)
        
        # Calculate precision, recall, and F1 score
        tp = metrics['anomaly']['correct']  # True Positives
        fp = metrics['normal']['incorrect']  # False Positives
        fn = metrics['anomaly']['incorrect']  # False Negatives
        
        metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['f1'] = (2 * metrics['precision'] * metrics['recall'] / 
                        (metrics['precision'] + metrics['recall'])
                        if (metrics['precision'] + metrics['recall']) > 0 else 0)
        
        # Calculate average scores
        metrics['average_scores'] = {
            'normal': {
                'anomaly_score': sum(r['anomaly_score'] for r in self.results 
                                    if r['true_label'] == 'normal') / 
                                metrics['normal']['total'] 
                                if metrics['normal']['total'] > 0 else 0,
                'normal_similarity': sum(r['normal_similarity'] for r in self.results 
                                        if r['true_label'] == 'normal') / 
                                    metrics['normal']['total'] 
                                    if metrics['normal']['total'] > 0 else 0,
                'anomaly_similarity': sum(r['anomaly_similarity'] for r in self.results 
                                        if r['true_label'] == 'normal') / 
                                    metrics['normal']['total'] 
                                    if metrics['normal']['total'] > 0 else 0
            },
            'anomaly': {
                'anomaly_score': sum(r['anomaly_score'] for r in self.results 
                                    if r['true_label'] == 'anomaly') / 
                                metrics['anomaly']['total'] 
                                if metrics['anomaly']['total'] > 0 else 0,
                'normal_similarity': sum(r['normal_similarity'] for r in self.results 
                                        if r['true_label'] == 'anomaly') / 
                                    metrics['anomaly']['total'] 
                                    if metrics['anomaly']['total'] > 0 else 0,
                'anomaly_similarity': sum(r['anomaly_similarity'] for r in self.results 
                                        if r['true_label'] == 'anomaly') / 
                                    metrics['anomaly']['total'] 
                                    if metrics['anomaly']['total'] > 0 else 0
            }
        }
        
        return metrics
    
    def print_metrics(self) -> None:
        """Print all computed metrics in a formatted way"""
        metrics = self.compute_metrics()
        
        print("\n" + "="*50)
        print("Performance Metrics:")
        print("="*50)
        
        print(f"\nOverall Performance:")
        print(f"Total images: {metrics['total']}")
        print(f"Correct predictions: {metrics['correct']}")
        print(f"Accuracy: {metrics['accuracy']*100:.2f}%")
        
        print(f"\nNormal Class Performance:")
        print(f"Total: {metrics['normal']['total']}")
        print(f"Correct: {metrics['normal']['correct']}")
        print(f"Incorrect: {metrics['normal']['incorrect']}")
        print(f"Accuracy: {metrics['normal']['accuracy']*100:.2f}%")
        print(f"Average Anomaly Score: {metrics['average_scores']['normal']['anomaly_score']:.3f}")
        print(f"Average Normal Similarity: {metrics['average_scores']['normal']['normal_similarity']:.3f}")
        print(f"Average Anomaly Similarity: {metrics['average_scores']['normal']['anomaly_similarity']:.3f}")
        
        print(f"\nAnomaly Class Performance:")
        print(f"Total: {metrics['anomaly']['total']}")
        print(f"Correct: {metrics['anomaly']['correct']}")
        print(f"Incorrect: {metrics['anomaly']['incorrect']}")
        print(f"Accuracy: {metrics['anomaly']['accuracy']*100:.2f}%")
        print(f"Average Anomaly Score: {metrics['average_scores']['anomaly']['anomaly_score']:.3f}")
        print(f"Average Normal Similarity: {metrics['average_scores']['anomaly']['normal_similarity']:.3f}")
        print(f"Average Anomaly Similarity: {metrics['average_scores']['anomaly']['anomaly_similarity']:.3f}")
        
        print(f"\nDetailed Metrics:")
        print(f"Precision: {metrics['precision']*100:.2f}%")
        print(f"Recall: {metrics['recall']*100:.2f}%")
        print(f"F1 Score: {metrics['f1']*100:.2f}%")
        print("="*50)
    
    def save_metrics(self, save_dir: str) -> None:
        """
        Save metrics to a JSON file.
        
        Args:
            save_dir: Directory to save the metrics file
        """
        metrics = self.compute_metrics()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(save_dir, f"metrics_{timestamp}.json")
        
        # Convert float32/64 to regular float for JSON serialization
        metrics = {k: float(v) if isinstance(v, (np.float32, np.float64)) 
                    else v for k, v in metrics.items()}
        
        with open(filename, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        print(f"\nMetrics saved to: {filename}")