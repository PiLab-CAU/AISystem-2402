import torch
import torch.optim as optim
from torch.nn import BCEWithLogitsLoss
from data_loader import get_data_loader
from model import load_clip_model
from tqdm import tqdm
import os
import matplotlib.pyplot as plt


def train_model(train_loader, model, criterion, optimizer, device):
    model.train()  # Set model to training mode
    running_loss = 0.0
    correct = 0
    total = 0
    progress_bar = tqdm(train_loader, desc="Training", leave=False)

    for images, labels in progress_bar:
        # Move data to specified device (GPU/CPU)
        images, labels = images.to(device), labels.to(device).float()
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images).squeeze()
        
        # Calculate loss and perform backpropagation
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        preds = (outputs > 0.5).float()  # Convert probability to binary prediction using 0.5 threshold
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        # Update running loss and progress bar
        running_loss += loss.item()
        progress_bar.set_postfix(loss=(running_loss / len(train_loader)))

    # Calculate and print epoch statistics
    avg_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total
    print(f"Train Loss: {avg_loss:.4f}, Train Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy



def evaluate_model(test_loader, model, device):    
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # Disable gradient computation during evaluation
        progress_bar = tqdm(test_loader, desc="Evaluating", leave=False)
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device).float()
            outputs = model(images).squeeze()
            
            # Calculate accuracy
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    print(f'Accuracy on test images: {accuracy:.2f}%')
    return accuracy






def main():
    # Set device (GPU if available, otherwise CPU)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    ## @@@@@@@@@@@@@@
    batch_size = 20
    print(f'@@@@@@@@@@@@ Batch: {batch_size}')
    
    # Initialize data loaders and model
    train_loader, test_loader = get_data_loader("./train", "./test", batch_size=batch_size)
    model = load_clip_model(device)

    # Define loss function and optimizer
    criterion = BCEWithLogitsLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=1e-4)
    
    num_epochs = 20
    best_accuracy = 0.0

    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Train and evaluate model
        train_loss, train_accuracy = train_model(train_loader, model, criterion, optimizer, device)
        print(f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")

        epoch_range = list(range(1, num_epochs+1))
        train_loss_graph(epoch_range, train_loss)



        

        if epoch == num_epochs-1:
            accuracy = evaluate_model(test_loader, model, device)
            
            # Save model if it achieves better accuracy
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                model_save_path = os.path.join("saved_models", "best_model.pth")
                os.makedirs("saved_models", exist_ok=True)
                torch.save(model.state_dict(), model_save_path)
                print(f"Model saved at {model_save_path} with accuracy: {best_accuracy:.2f}%")


def train_loss_graph(epoch_range, train_loss):
    # Plotting train loss
    plt.figure(figsize=(10, 5))
    plt.plot(epoch_range, train_loss, label='Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

def train_acc_graph(epoch_range, train_accuracy):
    # Plotting train accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(epoch_range, train_accuracy, label='Train Accuracy', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Train Accuracy over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()



if __name__ == "__main__":
    print("cuda:0" if torch.cuda.is_available() else "cpu")
    main()