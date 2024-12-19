import torch
import time
import os

from model import EnsembleCLIPClassifier
from data_loader import CustomDataset
from torchvision import transforms
from collections import defaultdict
from utils.visualize import save_prediction_image


def inference(image_dir, model_path, device):
    # Define transformation for resizing and converting images to tensors
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    # Define directories for normal and anomaly test datasets
    normal_dir = os.path.join(image_dir, "normal")
    anomaly_dir = os.path.join(image_dir, "anomaly")

    # Load datasets for normal and anomaly images
    normal_dataset = CustomDataset(normal_dir, transform=transform, mode="test")
    anomaly_dataset = CustomDataset(anomaly_dir, transform=transform, mode="test")

    # Create data loaders for normal and anomaly images with batch size of 1
    normal_loader = torch.utils.data.DataLoader(normal_dataset, batch_size=1, shuffle=False)
    anomaly_loader = torch.utils.data.DataLoader(anomaly_dataset, batch_size=1, shuffle=False)

    # Load pre-trained CLIP model and set it to evaluation mode
    model = EnsembleCLIPClassifier(num_models=3, device=device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Initialize counters for tracking correct predictions and total counts per class
    correct_counts = defaultdict(int)  # Counts of correct predictions by label
    total_counts = defaultdict(int)    # Total image counts by label
    total_correct = 0                  # Total correct predictions across all images
    total_images = 0                   # Total number of test images
    start_time = time.time()           # Record start time for inference duration

    # Predict on normal dataset
    for idx, (images, _) in enumerate(normal_loader):
        images = images.to(device)               # Move images to specified device (CPU or GPU)
        outputs = model(images)                  # Model prediction
        _, predicted = torch.max(outputs, 1)     # Get predicted class (0 for normal, 1 for anomaly)
        
        label_name = "normal"                    # Actual label for normal dataset
        predicted_name = "anomaly" if predicted.item() == 1 else "normal"  # Convert prediction to label name

        # Save image with prediction overlay for visualization
        save_prediction_image(images[0], label_name, predicted_name, output_dir="results", image_name=f"normal_{idx + 1}.jpg")

        # Update counts if prediction matches actual label
        if predicted_name == label_name:
            correct_counts[label_name] += 1
            total_correct += 1
        total_counts[label_name] += 1
        total_images += 1

    # Predict on anomaly dataset
    for idx, (images, _) in enumerate(anomaly_loader):
        images = images.to(device)               # Move images to specified device (CPU or GPU)
        outputs = model(images)                  # Model prediction
        _, predicted = torch.max(outputs, 1)     # Get predicted class (0 for normal, 1 for anomaly)
        
        label_name = "anomaly"                   # Actual label for anomaly dataset
        predicted_name = "anomaly" if predicted.item() == 1 else "normal"  # Convert prediction to label name

        # Save image with prediction overlay for visualization
        save_prediction_image(images[0], label_name, predicted_name, output_dir="results", image_name=f"anomaly_{idx + 1}.jpg")

        # Update counts if prediction matches actual label
        if predicted_name == label_name:
            correct_counts[label_name] += 1
            total_correct += 1
        total_counts[label_name] += 1
        total_images += 1

    end_time = time.time()              # Record end time for calculating inference duration
    elapsed_time = end_time - start_time  # Calculate total time taken for inference

    # Handle case where there are no images in test dataset
    if total_images == 0:
        print("No images found in the test dataset.")
        return

    # Print inference summary, including accuracy per class and overall accuracy
    print("\n--- Inference Summary ---")
    for label in total_counts:
        accuracy = 100 * correct_counts[label] / total_counts[label]  # Calculate accuracy for each label
        print(f"{label.capitalize()} - Correct: {correct_counts[label]}, Total: {total_counts[label]}, Accuracy: {accuracy:.2f}%")
    
    # Calculate overall accuracy across both normal and anomaly classes
    overall_accuracy = 100 * total_correct / total_images
    print(f"\nOverall Accuracy: {overall_accuracy:.2f}%")
    print(f"Total Time Taken: {elapsed_time:.2f} seconds")
    print(f"Average Time per Image: {elapsed_time / total_images:.4f} seconds")

# Run inference on specified test dataset and model path
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inference("./test", "./saved_models/best_model.pth", device)
