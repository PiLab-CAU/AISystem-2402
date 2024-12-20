import torch
import torch.optim as optim
from torch.nn import BCEWithLogitsLoss
from data_loader import get_data_loader
from model import load_clip_model, EnsembleCLIPClassifier
from tqdm import tqdm
import os
import numpy as np
from utils.losses import FocalLoss

def train_model(train_loader, model, criterion, optimizer, scheduler, device):
    model.train()  # Set model to training mode
    running_loss = 0.0
    correct = 0
    total = 0
    progress_bar = tqdm(train_loader, desc="Training", leave=False)
    
    # Gradient Accumulation 설정 (더 작은 값으로 조정)
    accumulation_steps = 2
    optimizer.zero_grad()
    
    for i, (images, labels) in enumerate(progress_bar):
        images, labels = images.to(device), labels.to(device).float()
        
        # Forward pass
        outputs = model(images).squeeze()
        loss = criterion(outputs, labels)
        
        # Loss scaling for gradient accumulation
        loss = loss / accumulation_steps
        loss.backward()
        
        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Gradient Accumulation
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # Calculate accuracy
        preds = (outputs > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        # Update running loss and progress bar
        running_loss += loss.item() * accumulation_steps
        progress_bar.set_postfix(loss=(running_loss / (i + 1)))

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
        for images, labels in test_loader:
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize data loaders and model with larger batch size
    train_loader, test_loader = get_data_loader("./train", "./test", batch_size=16)
    model = EnsembleCLIPClassifier(num_models=3, device=device)

    # 학습 파라미터 설정
    num_epochs = 5
    best_accuracy = 0.0

    # Define loss function and optimizer with adjusted learning rate
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    
    # 단일 스케줄러 사용 (OneCycleLR)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=5e-5,  # 최대 학습률 조정
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,  # warm-up 비율 증가
        div_factor=25.0,  # 초기 학습률 = max_lr/div_factor
        final_div_factor=1000.0  # 최종 학습률 = max_lr/final_div_factor
    )

    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Train and evaluate model (scheduler 전달)
        train_loss, train_accuracy = train_model(train_loader, model, criterion, optimizer, scheduler, device)
        print(f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
        
        accuracy = evaluate_model(test_loader, model, device)
        
        # Save model if it achieves better accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            model_save_path = os.path.join("saved_models", "best_model.pth")
            os.makedirs("saved_models", exist_ok=True)
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved at {model_save_path} with accuracy: {best_accuracy:.2f}%")

if __name__ == "__main__":
    main()