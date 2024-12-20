import os
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

def save_prediction_image(image_tensor, label, prediction, output_dir="results", image_name="prediction.jpg"):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Convert the tensor to a PIL image
    image = transforms.ToPILImage()(image_tensor.squeeze().cpu())
    
    # Set up the visualization with two subplots
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))  # Create two side-by-side subplots
    ax[0].imshow(image)                           # Display the image in the first subplot
    ax[0].axis("off")                             # Remove axis for the image display
    
    # Display text with label and prediction in the second subplot
    result_text = f"Label: {label}\nPrediction: {prediction}"
    ax[1].text(0.1, 0.5, result_text, fontsize=12, va="center", color="red")  # Show text
    ax[1].axis("off")                             # Remove axis for the text display
    
    # Save the visualization
    save_path = os.path.join(output_dir, image_name)
    plt.savefig(save_path, bbox_inches="tight")    # Save the figure with minimal whitespace
    plt.close(fig)                                 # Close the figure to save memory
    print(f"Saved prediction image at {save_path}")
