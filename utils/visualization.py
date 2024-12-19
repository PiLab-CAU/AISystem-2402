import os
import matplotlib.pyplot as plt
from PIL import Image

def save_predictions(
    image_path: str,
    true_label: str,
    predicted_label: str,
    anomaly_score: float,
    normal_similarity: float,
    anomaly_similarity: float,
    threshold: float,
    is_anomaly: bool,
    save_dir: str = "./results"
) -> None:
    """
    Save the image with its prediction details.
    
    Args:
        image_path: Path to the input image
        true_label: Ground truth label
        predicted_label: Predicted label
        anomaly_score: Computed anomaly score
        normal_similarity: Similarity with normal class
        anomaly_similarity: Similarity with anomaly class
        threshold: Threshold used for classification
        is_anomaly: Whether the image was classified as anomaly
        save_dir: Directory to save the results (default: "./results")
    """
    try:
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Read the image
        image = Image.open(image_path)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot image
        ax1.imshow(image)
        ax1.axis('off')
        ax1.set_title('Image')
        
        # Create text box with prediction details
        text = (
            f"Prediction Details\n"
            f"True Label: {true_label}\n"
            f"Predicted Label: {predicted_label}\n"
            f"Anomaly Score: {anomaly_score:.3f}\n"
            f"Normal Similarity: {normal_similarity:.3f}\n"
            f"Anomaly Similarity: {anomaly_similarity:.3f}\n"
            f"Threshold: {threshold}"
        )
        
        # Display text
        ax2.text(0.1, 0.5, text, fontsize=10, verticalalignment='center')
        ax2.axis('off')
        
        # Set title color based on prediction correctness
        title_color = 'green' if predicted_label == true_label else 'red'
        plt.suptitle(f"Prediction Result", color=title_color, fontsize=14)
        
        # Create filename for saving
        base_name = os.path.basename(image_path)
        name_without_ext = os.path.splitext(base_name)[0]
        result_path = os.path.join(
            save_dir, 
            f"{name_without_ext}_result_{predicted_label}.png"
        )
        
        # Save figure
        plt.savefig(result_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        
    except Exception as e:
        print(f"Error saving visualization for {image_path}: {str(e)}")

def plot_predictions(
    image_path: str,
    true_label: str,
    predicted_label: str,
    anomaly_score: float,
    normal_similarity: float,
    anomaly_similarity: float,
    threshold: float,
    is_anomaly: bool
) -> None:
    """
    Plot the image with its prediction details (for display, not saving).
    
    Args:
        image_path: Path to the input image
        true_label: Ground truth label
        predicted_label: Predicted label
        anomaly_score: Computed anomaly score
        normal_similarity: Similarity with normal class
        anomaly_similarity: Similarity with anomaly class
        threshold: Threshold used for classification
        is_anomaly: Whether the image was classified as anomaly
    """
    try:
        # Read the image
        image = Image.open(image_path)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot image
        ax1.imshow(image)
        ax1.axis('off')
        ax1.set_title('Image')
        
        # Create text box with prediction details
        text = (
            f"Prediction Details\n"
            f"True Label: {true_label}\n"
            f"Predicted Label: {predicted_label}\n"
            f"Anomaly Score: {anomaly_score:.3f}\n"
            f"Normal Similarity: {normal_similarity:.3f}\n"
            f"Anomaly Similarity: {anomaly_similarity:.3f}\n"
            f"Threshold: {threshold}"
        )
        
        # Display text
        ax2.text(0.1, 0.5, text, fontsize=10, verticalalignment='center')
        ax2.axis('off')
        
        # Set title color based on prediction correctness
        title_color = 'green' if predicted_label == true_label else 'red'
        plt.suptitle(f"Prediction Result", color=title_color, fontsize=14)
        
        plt.show()
        plt.close(fig)
        
    except Exception as e:
        print(f"Error displaying visualization for {image_path}: {str(e)}")