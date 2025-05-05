import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import numpy as np
import os

from . import config
from .dataset import get_dataloaders
from .model import SimpleCNN

def test_model(model_path=config.MODEL_SAVE_PATH):
    print(f"Using device: {config.DEVICE}")

    # Create output directory if it doesn't exist
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "test_summary.txt") # Path for summary file

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
        # Add weights_only=True for security best practice
        model.load_state_dict(torch.load(model_path, map_location=config.DEVICE, weights_only=True))
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        return
    model.eval() # Set model to evaluation mode
    print("Model loaded.")

    all_labels = []
    all_predictions = []

    progress_bar = tqdm(test_loader, desc="Testing", leave=False)
    with torch.no_grad():
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)

            outputs = model(inputs)

            _, predicted = torch.max(outputs.data, 1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # Convert numeric labels/predictions back to class names for confusion matrix labels
    all_labels_named = [idx_to_class[i] for i in all_labels]
    all_predictions_named = [idx_to_class[i] for i in all_predictions]

    # Calculate Confusion Matrix
    print("\nCalculating Confusion Matrix...")
    cm = confusion_matrix(all_labels_named, all_predictions_named, labels=class_names)
    print("Confusion Matrix:")
    print(cm)

    # Calculate Overall Accuracy from Confusion Matrix
    overall_accuracy = np.trace(cm) / np.sum(cm)
    print(f"\nOverall Accuracy: {overall_accuracy:.4f}")

    # Save Accuracy and Confusion Matrix to text file
    try:
        with open(summary_path, 'w') as f:
            f.write(f"Overall Accuracy: {overall_accuracy:.4f}\n\n")
            f.write("Confusion Matrix:\n")
            # Add class names as header for clarity
            header = "Predicted -> " + " | ".join(class_names) + "\n"
            f.write(header)
            f.write("-" * len(header) + "\n")
            for i, class_name in enumerate(class_names):
                 # Format the row string for better alignment
                 row_str = f"True {class_name:<10}| " + " | ".join(f"{x:>4}" for x in cm[i]) + "\n"
                 f.write(row_str)
        print(f"Accuracy and Confusion Matrix saved to {summary_path}")
    except Exception as e:
        print(f"Error saving summary file: {e}")

if __name__ == '__main__':
    test_model()
