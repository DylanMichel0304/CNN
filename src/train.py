import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
import pandas as pd
import matplotlib.pyplot as plt
import os # Added for directory creation

from . import config
from .dataset import get_dataloaders
from .model import SimpleCNN

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train() # Set model to training mode
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

        progress_bar.set_postfix(loss=loss.item(), acc=f"{(predicted == labels).sum().item()/labels.size(0):.4f}")

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    return epoch_loss, epoch_acc

def validate_one_epoch(model, dataloader, criterion, device):
    model.eval() # Set model to evaluation mode
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    progress_bar = tqdm(dataloader, desc="Validation", leave=False)
    with torch.no_grad(): # No need to track gradients during validation
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            progress_bar.set_postfix(loss=loss.item(), acc=f"{(predicted == labels).sum().item()/labels.size(0):.4f}")


    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    return epoch_loss, epoch_acc

def plot_metrics(history, save_path):
    """Plots training and validation loss and accuracy."""
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'bo-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'bo-', label='Training Accuracy')
    plt.plot(epochs, history['val_acc'], 'ro-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Training curves plot saved to {save_path}")
    # plt.show() # Optionally display the plot

def train_model():
    print(f"Using device: {config.DEVICE}")

    # Create output directory if it doesn't exist
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, "training_metrics.csv")
    plot_path = os.path.join(output_dir, "training_curves.png")
    model_save_path = config.MODEL_SAVE_PATH # Use path from config

    # Get DataLoaders
    print("Loading data...")
    train_loader, valid_loader, _, class_map = get_dataloaders()
    print("Data loaded.")

    # Initialize model, criterion, optimizer
    model = SimpleCNN(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    best_val_acc = 0.0
    start_time = time.time()
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []} # Initialize history

    print("Starting training...")
    for epoch in range(config.NUM_EPOCHS):
        epoch_start_time = time.time()
        print(f"\n--- Epoch {epoch+1}/{config.NUM_EPOCHS} ---")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        val_loss, val_acc = validate_one_epoch(model, valid_loader, criterion, config.DEVICE)

        epoch_duration = time.time() - epoch_start_time

        # Store metrics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Valid Loss: {val_loss:.4f}, Valid Acc: {val_acc:.4f}")
        print(f"  Duration: {epoch_duration:.2f} seconds")

        # Save the model if validation accuracy improves
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Ensure the directory for the model exists (if it includes subdirs)
            model_dir = os.path.dirname(model_save_path)
            if model_dir: # Check if path includes a directory
                 os.makedirs(model_dir, exist_ok=True)
            torch.save(model.state_dict(), model_save_path)
            print(f"  Validation accuracy improved. Model saved to {model_save_path}")

    total_training_time = time.time() - start_time
    print(f"\nTraining finished in {total_training_time // 60:.0f}m {total_training_time % 60:.0f}s")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")

    # Save metrics to CSV
    df = pd.DataFrame(history)
    df.to_csv(metrics_path, index=False)
    print(f"Training metrics saved to {metrics_path}")

    # Plot metrics
    plot_metrics(history, plot_path)


if __name__ == '__main__':
    train_model()
