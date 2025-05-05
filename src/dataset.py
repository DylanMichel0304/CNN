import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
import numpy as np
from . import config


# Define transform for training data 
train_transform = transforms.Compose([
     transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
     transforms.RandomHorizontalFlip(p=0.5), # 50% chance 
     transforms.RandomRotation(10),         # Rotate by max +/- 10 deg
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
 ])

# Define transform for validation and testing 
test_val_transform = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])


def get_dataloaders():

    train_dataset_aug = datasets.ImageFolder(config.TRAIN_DIR, transform=train_transform) 
    #train_dataset_no_aug = datasets.ImageFolder(config.TRAIN_DIR, transform=test_val_transform)
    class_to_idx = train_dataset_no_aug.class_to_idx # Get class mapping from one instance

    test_dataset = datasets.ImageFolder(config.TEST_DIR, transform=test_val_transform)

    num_train = len(train_dataset_aug) # Use length from the non-augmented instance
    indices = list(range(num_train))
    split = int(np.floor(config.VALID_SPLIT * num_train))

    np.random.shuffle(indices) 

    train_idx, valid_idx = indices[split:], indices[:split]

    train_subset = Subset(train_dataset_aug, train_idx)
    valid_subset = Subset(train_dataset_aug, valid_idx) 

    num_workers = 4 if config.DEVICE == "cuda" else 0 # Use workers for GPU, 0 for CPU often better
    train_loader = DataLoader(train_subset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True if config.DEVICE == "cuda" else False)
    valid_loader = DataLoader(valid_subset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True if config.DEVICE == "cuda" else False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True if config.DEVICE == "cuda" else False)

    print(f"Train samples: {len(train_subset)}, Validation samples: {len(valid_subset)}, Test samples: {len(test_dataset)}")
    print(f"Classes: {class_to_idx}")

    return train_loader, valid_loader, test_loader, class_to_idx

if __name__ == '__main__':
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
