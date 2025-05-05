import torch

# Data paths
TRAIN_DIR = 
TEST_DIR = #plus le meme 

#parameters
IMAGE_SIZE = 224
BATCH_SIZE = 128
VALID_SPLIT = 0.2 # 20% for validation
NUM_CLASSES = 3 # Normal, Virus, Bacteria

# Training parameters
LEARNING_RATE = 0.0001
NUM_EPOCHS = 50 #
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model saving
MODEL_SAVE_PATH = 'pneumonia_cnn_model.pth'

# Results saving
RESULTS_SAVE_PATH = 'training_results.json'
