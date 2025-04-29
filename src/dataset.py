import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
import numpy as np
from . import config


# Define transform for training data (with augmentation)
train_transform = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5), # 50% chance of horizontal flip
    transforms.RandomRotation(10),         # Rotate by up to +/- 10 degrees
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Keep commented out for now
])

# Define transform for validation and testing (no augmentation)
test_val_transform = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Keep commented out for now
])


def get_dataloaders():

    # Load the training dataset WITH augmentation transforms
    train_dataset_aug = datasets.ImageFolder(config.TRAIN_DIR, transform=train_transform)
    class_to_idx = train_dataset_aug.class_to_idx # Get class mapping from one instance

    # Load the training dataset again but WITH standard transforms for the validation split
    # This ensures validation data doesn't have augmentation applied
    train_dataset_std = datasets.ImageFolder(config.TRAIN_DIR, transform=test_val_transform)

    # Load the test dataset using the standard (non-augmented) transform
    test_dataset = datasets.ImageFolder(config.TEST_DIR, transform=test_val_transform)

    # Split the training dataset indices into training and validation sets
    num_train = len(train_dataset_aug) # Use length from one instance
    indices = list(range(num_train))
    split = int(np.floor(config.VALID_SPLIT * num_train))

    # Ensure reproducibility if needed (optional)
    # np.random.seed(42)
    np.random.shuffle(indices) # Shuffle indices before splitting

    train_idx, valid_idx = indices[split:], indices[:split]

    # Create Subsets using the appropriate dataset instance and indices
    train_subset = Subset(train_dataset_aug, train_idx) # Training subset uses augmented data
    valid_subset = Subset(train_dataset_std, valid_idx) # Validation subset uses standard transformed data


    # Create DataLoaders
    # Consider adjusting num_workers based on your system
    num_workers = 4 if config.DEVICE == "cuda" else 0 # Use workers for GPU, 0 for CPU often better
    train_loader = DataLoader(train_subset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True if config.DEVICE == "cuda" else False)
    valid_loader = DataLoader(valid_subset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True if config.DEVICE == "cuda" else False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True if config.DEVICE == "cuda" else False)

    print(f"Train samples: {len(train_subset)}, Validation samples: {len(valid_subset)}, Test samples: {len(test_dataset)}")
    print(f"Classes: {class_to_idx}")

    return train_loader, valid_loader, test_loader, class_to_idx

if __name__ == '__main__':
    # Example usage: Load data and print batch shape
    train_dl, valid_dl, test_dl, class_map = get_dataloaders()
    print("DataLoaders created.")
    for images, labels in train_dl:
        print("Train batch shape:", images.shape, labels.shape)
        break
    for images, labels in valid_dl:
        print("Validation batch shape:", images.shape, labels.shape)
        break
    for images, labels in test_dl:
        print("Test batch shape:", images.shape, labels.shape)
        break
