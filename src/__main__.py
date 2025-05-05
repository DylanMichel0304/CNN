import argparse

from .train import train_model
from .test import test_model

def main():
    parser = argparse.ArgumentParser(description="Train or test the Pneumonia CNN classifier.")
    parser.add_argument('mode', choices=['train', 'test'], help="'train' to train the model, 'test' to evaluate the model.")
    parser.add_argument('--model_path', type=str, default=None, help="Path to the model file (required for testing, optional for training if continuing).")

    args = parser.parse_args()

    if args.mode == 'train':
        print("--- Starting Training Mode ---")
        train_model()
    elif args.mode == 'test':
        print("--- Starting Testing Mode ---")
        model_to_test = args.model_path if args.model_path else config.MODEL_SAVE_PATH
        if not model_to_test:
             print("Error: Model path must be specified for testing using --model_path or defined in config.py")
             return
        test_model(model_path=model_to_test)

if __name__ == '__main__':
    main()
