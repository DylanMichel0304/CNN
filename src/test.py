import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os # Added for output directory

from . import config
from .dataset import get_dataloaders
from .model import SimpleCNN

def test_model(model_path=config.MODEL_SAVE_PATH):
    print(f"Using device: {config.DEVICE}")

    # Create output directory if it doesn't exist
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    cm_plot_path = os.path.join(output_dir, "confusion_matrix.png") # Save plot in outputs

    # Get DataLoaders (we only need the test loader and class map here)
    print("Loading data...")
    _, _, test_loader, class_map = get_dataloaders()
    # Invert class_map for reporting
    idx_to_class = {v: k for k, v in class_map.items()}
    class_names = list(idx_to_class.values()) # Get class names in order
    num_classes = len(class_names)
    print("Data loaded.")

    # Load model
    print(f"Loading model from {model_path}...")
    model = SimpleCNN(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    try:
        # Ensure model exists before loading
        if not os.path.exists(model_path):
             print(f"Error: Model file not found at {model_path}. Train the model first.")
             return
        model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        return
    model.eval() # Set model to evaluation mode
    print("Model loaded.")

    all_labels = []
    all_predictions = []
    running_loss = 0.0
    total_samples = 0
    criterion = nn.CrossEntropyLoss() # To calculate loss

    progress_bar = tqdm(test_loader, desc="Testing", leave=False)
    with torch.no_grad():
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs.data, 1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            total_samples += labels.size(0)

    test_loss = running_loss / total_samples
    print(f"\n--- Test Results ---")
    print(f"Overall Test Loss: {test_loss:.4f}")

    # Convert numeric labels/predictions back to class names for reporting
    all_labels_named = [idx_to_class[i] for i in all_labels]
    all_predictions_named = [idx_to_class[i] for i in all_predictions]

    # Classification Report (provides precision, recall, f1-score per class)
    print("\nClassification Report:")
    print(classification_report(all_labels_named, all_predictions_named, target_names=class_names, zero_division=0))

    # Confusion Matrix
    print("Confusion Matrix:")
    cm = confusion_matrix(all_labels_named, all_predictions_named, labels=class_names)
    print(cm)

    # Calculate and print TP, TN, FP, FN, and Accuracy per class
    print("\nPer-Class Metrics:")
    metrics_data = []
    for i, class_name in enumerate(class_names):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        TN = cm.sum() - (TP + FP + FN)
        accuracy = (TP + TN) / cm.sum() if cm.sum() > 0 else 0

        print(f"\nMetrics for class: {class_name}")
        print(f"  True Positives (TP): {TP}")
        print(f"  False Positives (FP): {FP}")
        print(f"  False Negatives (FN): {FN}")
        print(f"  True Negatives (TN): {TN}")
        print(f"  Accuracy: {accuracy:.4f}")
        metrics_data.append({'Class': class_name, 'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN, 'Accuracy': accuracy})

    # Optional: Save per-class metrics to CSV
    metrics_df = pd.DataFrame(metrics_data)
    metrics_csv_path = os.path.join(output_dir, "test_per_class_metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"\nPer-class metrics saved to {metrics_csv_path}")


    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout() # Adjust layout
    plt.savefig(cm_plot_path) # Use defined path
    print(f"\nConfusion matrix plot saved to {cm_plot_path}")
    # plt.show() # Optionally display

if __name__ == '__main__':
    test_model()
